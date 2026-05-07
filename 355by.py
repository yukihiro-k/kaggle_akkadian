#!/usr/bin/env python3
"""
Deep Past Challenge — Akkadian-to-English Translation
Ensemble MBR inference — DUAL T4 OPTIMIZED

Optimizations vs original:
  1. Model A → cuda:0, Model B → cuda:1 (true parallel inference)
  2. FP16 AMP instead of BF16 (T4 does not support BF16 natively)
  3. ThreadPoolExecutor runs both models simultaneously, halving wall-clock time
  4. batch_size raised to 8 (T4 16 GB VRAM is sufficient for ByT5-base)
  5. torch.compile (torch >= 2.0) with mode="reduce-overhead"
  6. torch.cuda.Stream overlap for encode→GPU transfer during generation
  7. Per-device memory management + explicit stream synchronisation
  8. DataLoader pinned memory uses device-specific allocators
  9. Adaptive beam sizing kept but threshold recalibrated for T4 throughput
  10. All original postprocessing corrections preserved (v3 fixes 1-7)
"""

import os, gc, re, json, math, random, logging, warnings, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from dataclasses import dataclass, field
from contextlib import nullcontext
from typing import List, Optional, Tuple, Dict

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm.auto import tqdm
import sacrebleu

warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

# ---------------------------------------------------------------------------
# 1. GPU / precision helpers
# ---------------------------------------------------------------------------

def _num_gpus() -> int:
    return torch.cuda.device_count() if torch.cuda.is_available() else 0

def _fp16_supported(device: torch.device) -> bool:
    """T4 supports FP16; use it instead of BF16."""
    if device.type != "cuda":
        return False
    try:
        props = torch.cuda.get_device_properties(device)
        # FP16 is safe on all NVIDIA GPUs with compute capability >= 6.0
        return props.major >= 6
    except Exception:
        return False

def _bf16_supported(device: torch.device) -> bool:
    """BF16 requires Ampere (compute >= 8.0) — T4 is 7.5, so False."""
    if device.type != "cuda":
        return False
    try:
        return bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    except Exception:
        return False

def _amp_ctx(device: torch.device, enabled: bool):
    """Return best available AMP context for the device."""
    if not enabled or device.type != "cuda":
        return nullcontext()
    if _bf16_supported(device):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if _fp16_supported(device):
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()

# ---------------------------------------------------------------------------
# 2. Configuration
# ---------------------------------------------------------------------------

@dataclass
class EnsembleMBRConfig:
    test_data_path: str = "/kaggle/input/competitions/deep-past-initiative-machine-translation/test.csv"
    output_dir:     str = "/kaggle/working/"
    model_a_path:   str = "/kaggle/input/datasets/assiaben/final-byt5/byt5-akkadian-optimized-34x"
    model_b_path:   str = "/kaggle/input/models/mattiaangeli/byt5-akkadian-mbr-v2/pytorch/default/1"

    # --- Dual-GPU assignment -------------------------------------------
    # Model A → GPU 0, Model B → GPU 1 when two GPUs are available.
    # Falls back to single-GPU / CPU automatically.
    gpu_a: int = 0
    gpu_b: int = 1

    max_input_length: int = 512
    # OPT: T4 16 GB lets us push batch_size to 8 for ByT5-base
    max_new_tokens:   int = 384
    batch_size:       int = 8          # was 2
    num_workers:      int = 4          # was 2 — more prefetch threads
    num_buckets:      int = 6

    num_beam_cands:      int = 4
    num_beams:           int = 8
    length_penalty:      float = 1.3
    early_stopping:      bool = True
    repetition_penalty:  float = 1.2

    use_diverse_beam:    bool = False
    num_diverse_cands:   int = 4
    num_diverse_beams:   int = 8
    num_beam_groups:     int = 4
    diversity_penalty:   float = 0.8

    use_sampling:        bool = True
    sample_temperatures: List[float] = field(default_factory=lambda: [0.60, 0.80, 1.05])
    num_sample_per_temp: int = 2
    mbr_top_p:           float = 0.92

    @property
    def num_sample_cands(self) -> int:
        return len(self.sample_temperatures) * self.num_sample_per_temp

    mbr_pool_cap: int = 32

    mbr_w_chrf:    float = 0.55
    mbr_w_bleu:    float = 0.25
    mbr_w_jaccard: float = 0.20
    mbr_w_length:  float = 0.10

    use_mixed_precision:    bool = True
    use_better_transformer: bool = True
    use_bucket_batching:    bool = True
    use_adaptive_beams:     bool = True
    # OPT: torch.compile accelerates ByT5 on T4 (requires torch >= 2.0)
    use_torch_compile:      bool = True
    aggressive_postprocessing: bool = True
    checkpoint_freq: int = 200

    def __post_init__(self):
        n_gpus = _num_gpus()

        # Assign devices
        if n_gpus == 0:
            self.device_a = torch.device("cpu")
            self.device_b = torch.device("cpu")
            self.dual_gpu = False
        elif n_gpus == 1:
            self.device_a = torch.device("cuda:0")
            self.device_b = torch.device("cuda:0")
            self.dual_gpu = False
        else:
            self.device_a = torch.device(f"cuda:{self.gpu_a}")
            self.device_b = torch.device(f"cuda:{self.gpu_b}")
            self.dual_gpu = True

        # Legacy single-device reference kept for compatibility
        self.device = self.device_a

        Path(self.output_dir).mkdir(exist_ok=True, parents=True)

        if n_gpus == 0:
            self.use_mixed_precision    = False
            self.use_better_transformer = False
            self.use_torch_compile      = False

        # Per-device precision flags
        self.use_amp_a = self.use_mixed_precision and self.device_a.type == "cuda"
        self.use_amp_b = self.use_mixed_precision and self.device_b.type == "cuda"

        # Informational
        self.amp_dtype_a = (
            "bf16" if _bf16_supported(self.device_a) else
            "fp16" if _fp16_supported(self.device_a) else
            "none"
        )
        self.amp_dtype_b = (
            "bf16" if _bf16_supported(self.device_b) else
            "fp16" if _fp16_supported(self.device_b) else
            "none"
        )

        assert self.num_beams >= self.num_beam_cands
        if self.use_diverse_beam:
            assert self.num_diverse_beams % self.num_beam_groups == 0
            assert self.num_diverse_beams >= self.num_diverse_cands

# ---------------------------------------------------------------------------
# 3. Logging
# ---------------------------------------------------------------------------

def setup_logging(output_dir: str) -> logging.Logger:
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(Path(output_dir) / "ensemble_mbr.log"),
        ],
    )
    return logging.getLogger("ensemble_mbr")

# ---------------------------------------------------------------------------
# 4. Preprocessing  (unchanged from v3)
# ---------------------------------------------------------------------------

_V2 = re.compile(r"([aAeEiIuU])(?:2|₂)")
_V3 = re.compile(r"([aAeEiIuU])(?:3|₃)")
_ACUTE = str.maketrans({"a":"á","e":"é","i":"í","u":"ú","A":"Á","E":"É","I":"Í","U":"Ú"})
_GRAVE = str.maketrans({"a":"à","e":"è","i":"ì","u":"ù","A":"À","E":"È","I":"Ì","U":"Ù"})

def _ascii_to_diacritics(s: str) -> str:
    s = s.replace("sz","š").replace("SZ","Š")
    s = s.replace("s,","ṣ").replace("S,","Ṣ")
    s = s.replace("t,","ṭ").replace("T,","Ṭ")
    s = _V2.sub(lambda m: m.group(1).translate(_ACUTE), s)
    s = _V3.sub(lambda m: m.group(1).translate(_GRAVE), s)
    return s

_ALLOWED_FRACS = [
    (1/6,"0.16666"),(1/4,"0.25"),(1/3,"0.33333"),(1/2,"0.5"),
    (2/3,"0.66666"),(3/4,"0.75"),(5/6,"0.83333"),
]
_FRAC_TOL = 2e-3
_FLOAT_RE = re.compile(r"(?<![\w/])(\d+\.\d{4,})(?![\w/])")

def _canon_decimal(x: float) -> str:
    ip = int(math.floor(x + 1e-12))
    frac = x - ip
    best = min(_ALLOWED_FRACS, key=lambda t: abs(frac - t[0]))
    if abs(frac - best[0]) <= _FRAC_TOL:
        dec = best[1]
        if ip == 0:
            return dec
        return f"{ip}{dec[1:]}" if dec.startswith("0.") else f"{ip}+{dec}"
    return f"{x:.5f}".rstrip("0").rstrip(".")

_WS_RE = re.compile(r"\s+")
_GAP_UNIFIED_RE = re.compile(
    r"<\s*big[\s_\-]*gap\s*>"
    r"|<\s*gap\s*>"
    r"|\bbig[\s_\-]*gap\b"
    r"|\bx(?:\s+x)+\b"
    r"|\.{3,}|…+|\[\.+\]"
    r"|\[\s*x\s*\]|\(\s*x\s*\)"
    r"|(?<!\w)x{2,}(?!\w)"
    r"|(?<!\w)x(?!\w)"
    r"|\(\s*large\s+break\s*\)"
    r"|\(\s*break\s*\)"
    r"|\(\s*\d+\s+broken\s+lines?\s*\)",
    re.I
)

def _normalize_gaps_vec(ser: pd.Series) -> pd.Series:
    return ser.str.replace(_GAP_UNIFIED_RE, "<gap>", regex=True)

_CHAR_TRANS = str.maketrans({
    "ḫ":"h","Ḫ":"H","ʾ":"",
    "₀":"0","₁":"1","₂":"2","₃":"3","₄":"4",
    "₅":"5","₆":"6","₇":"7","₈":"8","₉":"9",
    "—":"-","–":"-",
})
_SUB_X = "ₓ"

_UNICODE_UPPER = r"A-ZŠṬṢḪ\u00C0-\u00D6\u00D8-\u00DE\u0160\u1E00-\u1EFF"
_UNICODE_LOWER = r"a-zšṭṣḫ\u00E0-\u00F6\u00F8-\u00FF\u0161\u1E01-\u1EFF"
_DET_UPPER_RE  = re.compile(r"\(([" + _UNICODE_UPPER + r"0-9]{1,6})\)")
_DET_LOWER_RE  = re.compile(r"\(([" + _UNICODE_LOWER + r"]{1,4})\)")

_PN_RE         = re.compile(r"\bPN\b")
_KUBABBAR_RE   = re.compile(r"KÙ\.B\.")
_EXACT_FRAC_RE = re.compile(r"0\.8333|0\.6666|0\.3333|0\.1666|0\.625|0\.75|0\.25|0\.5")
_EXACT_FRAC_MAP = {
    "0.8333":"⅚","0.6666":"⅔","0.3333":"⅓","0.1666":"⅙",
    "0.625":"⅝","0.75":"¾","0.25":"¼","0.5":"½",
}

def _frac_repl(m: re.Match) -> str:
    return _EXACT_FRAC_MAP[m.group(0)]

class OptimizedPreprocessor:
    def preprocess_batch(self, texts: List[str]) -> List[str]:
        ser = pd.Series(texts).fillna("").astype(str)
        ser = ser.apply(_ascii_to_diacritics)
        ser = ser.str.replace(_DET_UPPER_RE, r"\1", regex=True)
        ser = ser.str.replace(_DET_LOWER_RE, r"{\1}", regex=True)
        ser = _normalize_gaps_vec(ser)
        ser = ser.str.translate(_CHAR_TRANS)
        ser = ser.str.replace(_SUB_X, "", regex=False)
        ser = ser.str.replace(_KUBABBAR_RE, "KÙ.BABBAR", regex=True)
        ser = ser.str.replace(_EXACT_FRAC_RE, _frac_repl, regex=True)
        ser = ser.str.replace(_FLOAT_RE, lambda m: _canon_decimal(float(m.group(1))), regex=True)
        ser = ser.str.replace(_WS_RE, " ", regex=True).str.strip()
        return ser.tolist()

# ---------------------------------------------------------------------------
# 5. Postprocessing  (all v3 fixes preserved)
# ---------------------------------------------------------------------------

_SOFT_GRAM_RE  = re.compile(
    r"\(\s*(?:fem|plur|pl|sing|singular|plural|\?|\!)"
    r"(?:\.\s*(?:plur|plural|sing|singular))?"
    r"\.?\s*[^)]*\)", re.I
)
_BARE_GRAM_RE  = re.compile(r"(?<!\w)(?:fem|sing|pl|plural)\.?(?!\w)\s*", re.I)
_UNCERTAIN_RE  = re.compile(r"\(\?\)")
_CURLY_DQ_RE   = re.compile("[\u201c\u201d]")
_CURLY_SQ_RE   = re.compile("[\u2018\u2019]")
_MONTH_RE      = re.compile(r"\bMonth\s+(XII|XI|X|IX|VIII|VII|VI|V|IV|III|II|I)\b", re.I)
_ROMAN2INT     = {"I":1,"II":2,"III":3,"IV":4,"V":5,"VI":6,"VII":7,"VIII":8,"IX":9,"X":10,"XI":11,"XII":12}
_REPEAT_WORD_RE  = re.compile(r"\b(\w+)(?:\s+\1\b)+")
_REPEAT_PUNCT_RE = re.compile(r"([.,])\1+")
_PUNCT_SPACE_RE  = re.compile(r"\s+([.,:])") 
_FORBIDDEN_TRANS = str.maketrans("", "", '——<>⌈⌋⌊[]+ʾ;')
_COMMODITY_RE    = re.compile(r'(?<=\s)-(gold|tax|textiles)\b')
_COMMODITY_REPL  = {"gold":"pašallum gold","tax":"šadduātum tax","textiles":"kutānum textiles"}

def _commodity_repl(m: re.Match) -> str:
    return _COMMODITY_REPL[m.group(1)]

_SHEKEL_REPLS = [
    (re.compile(r'5\s+11\s*/\s*12\s+shekels?', re.I), '6 shekels less 15 grains'),
    (re.compile(r'5\s*/\s*12\s+shekels?', re.I),      '⅓ shekel 15 grains'),
    (re.compile(r'7\s*/\s*12\s+shekels?', re.I),      '½ shekel 15 grains'),
    (re.compile(r'1\s*/\s*12\s*(?:\(shekel\)|\bshekel)?', re.I), '15 grains'),
]

_SLASH_ALT_RE   = re.compile(r'(?<![0-9/])\s+/\s+(?![0-9])\S+')
_STRAY_MARKS_RE = re.compile(r'<<[^>]*>>|<(?!gap\b)[^>]*>')
_MULTI_GAP_RE   = re.compile(r'(?:<gap>\s*){2,}')
_EXTRA_STRAY_RE = re.compile(r'(?<!\w)(?:\.\.+|xx+)(?!\w)')
_HACEK_TRANS    = str.maketrans({"ḫ":"h","Ḫ":"H"})

def _month_repl(m: re.Match) -> str:
    return f"Month {_ROMAN2INT.get(m.group(1).upper(), m.group(1))}"

class VectorizedPostprocessor:
    def postprocess_batch(self, translations: List[str]) -> List[str]:
        s = pd.Series(translations).fillna("").astype(str)
        s = _normalize_gaps_vec(s)
        s = s.str.replace(_PN_RE, "<gap>", regex=True)
        s = s.str.replace(_COMMODITY_RE, _commodity_repl, regex=True)
        for pat, repl in _SHEKEL_REPLS:
            s = s.str.replace(pat, repl, regex=True)
        s = s.str.replace(_EXACT_FRAC_RE, _frac_repl, regex=True)
        s = s.str.replace(_FLOAT_RE, lambda m: _canon_decimal(float(m.group(1))), regex=True)
        s = s.str.replace(_SOFT_GRAM_RE, " ", regex=True)
        s = s.str.replace(_BARE_GRAM_RE, " ", regex=True)
        s = s.str.replace(_UNCERTAIN_RE, "", regex=True)
        s = s.str.replace(_STRAY_MARKS_RE, "", regex=True)
        s = s.str.replace(_EXTRA_STRAY_RE, "", regex=True)
        s = s.str.replace(_SLASH_ALT_RE, "", regex=True)
        s = s.str.replace(_CURLY_DQ_RE, '"', regex=True)
        s = s.str.replace(_CURLY_SQ_RE, "'", regex=True)
        s = s.str.replace(_MONTH_RE, _month_repl, regex=True)
        s = s.str.replace(_MULTI_GAP_RE, "<gap>", regex=True)
        s = s.str.replace("<gap>", "\x00GAP\x00", regex=False)
        s = s.str.translate(_FORBIDDEN_TRANS)
        s = s.str.replace("\x00GAP\x00", " <gap> ", regex=False)
        s = s.str.translate(_HACEK_TRANS)
        s = s.str.replace(_REPEAT_WORD_RE, r"\1", regex=True)
        for n in range(4, 1, -1):
            pat = r"\b((?:\w+\s+){" + str(n-1) + r"}\w+)(?:\s+\1\b)+"
            s = s.str.replace(pat, r"\1", regex=True)
        s = s.str.replace(_PUNCT_SPACE_RE, r"\1", regex=True)
        s = s.str.replace(_REPEAT_PUNCT_RE, r"\1", regex=True)
        s = s.str.replace(_WS_RE, " ", regex=True).str.strip()
        return s.tolist()

# ---------------------------------------------------------------------------
# 6. Dataset + bucket batching
# ---------------------------------------------------------------------------

class AkkadianDataset(Dataset):
    def __init__(self, df: pd.DataFrame, preprocessor: OptimizedPreprocessor, logger: logging.Logger):
        self.sample_ids = df["id"].tolist()
        proc = preprocessor.preprocess_batch(df["transliteration"].tolist())
        self.input_texts = ["translate Akkadian to English: " + t for t in proc]
        logger.info(f"Dataset: {len(self.sample_ids)} samples")

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        return self.sample_ids[idx], self.input_texts[idx]

class BucketBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, num_buckets, logger, shuffle=False):
        self.batch_size = batch_size
        self.shuffle    = shuffle
        lengths    = [len(t.split()) for _, t in dataset]
        sorted_idx = sorted(range(len(lengths)), key=lambda i: lengths[i])
        bsize = max(1, len(sorted_idx) // max(1, num_buckets))
        self.buckets = [
            sorted_idx[i*bsize : None if i == num_buckets-1 else (i+1)*bsize]
            for i in range(num_buckets)
        ]
        for i, b in enumerate(self.buckets):
            if b:
                bl = [lengths[x] for x in b]
                logger.info(f"  Bucket {i}: {len(b)} samples, len [{min(bl)}, {max(bl)}]")

    def __iter__(self):
        for bucket in self.buckets:
            b = list(bucket)
            if self.shuffle:
                random.shuffle(b)
            for i in range(0, len(b), self.batch_size):
                yield b[i:i+self.batch_size]

    def __len__(self):
        return sum(math.ceil(len(b) / self.batch_size) for b in self.buckets)

# ---------------------------------------------------------------------------
# 7. Model wrapper — T4 aware
# ---------------------------------------------------------------------------

class ModelWrapper:
    def __init__(
        self,
        model_path: str,
        cfg: EnsembleMBRConfig,
        logger: logging.Logger,
        label: str,
        device: torch.device,
        use_amp: bool,
    ):
        self.cfg     = cfg
        self.logger  = logger
        self.label   = label
        self.device  = device
        self.use_amp = use_amp

        logger.info(f"[{label}] Loading → {device}  (path: {model_path})")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

        # OPT: cast to fp16 on T4 to halve VRAM and speed up matmuls
        if device.type == "cuda" and _fp16_supported(device) and not _bf16_supported(device):
            model = model.half()
            logger.info(f"[{label}] Weights cast to FP16 for T4")

        self.model = model.to(device).eval()

        if device.type == "cuda":
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

        # OPT: torch.compile for T4 (reduces Python overhead per forward pass)
        if cfg.use_torch_compile and device.type == "cuda":
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead", fullgraph=False)
                logger.info(f"[{label}] torch.compile applied (reduce-overhead)")
            except Exception as e:
                logger.warning(f"[{label}] torch.compile skipped: {e}")

        if cfg.use_better_transformer and device.type == "cuda":
            try:
                from optimum.bettertransformer import BetterTransformer
                self.model = BetterTransformer.transform(self.model)
                logger.info(f"[{label}] BetterTransformer applied")
            except Exception as e:
                logger.warning(f"[{label}] BetterTransformer skipped: {e}")

        n = sum(p.numel() for p in self.model.parameters() if hasattr(p, "numel"))
        logger.info(f"[{label}] ~{n:,} parameters")
        if device.type == "cuda":
            used = torch.cuda.memory_allocated(device) / 1e9
            logger.info(f"[{label}] GPU mem used after load: {used:.2f} GB")

        # OPT: dedicated CUDA stream per model for async overlap
        self.stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None

    def collate(self, batch_samples):
        ids   = [s[0] for s in batch_samples]
        texts = [s[1] for s in batch_samples]
        enc   = self.tokenizer(
            texts,
            max_length=self.cfg.max_input_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        return ids, enc

    def generate_candidates(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, beam_size: int) -> List[List[str]]:
        cfg = self.cfg
        B   = input_ids.shape[0]
        ctx = _amp_ctx(self.device, self.use_amp)

        Rb = cfg.num_beam_cands
        Rd = cfg.num_diverse_cands if cfg.use_diverse_beam else 0
        Rs = cfg.num_sample_per_temp

        with ctx:
            nb = max(beam_size, Rb)
            beam_out = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                num_beams=nb,
                num_return_sequences=Rb,
                max_new_tokens=cfg.max_new_tokens,
                length_penalty=cfg.length_penalty,
                early_stopping=cfg.early_stopping,
                repetition_penalty=cfg.repetition_penalty,
                use_cache=True,
            )
            beam_texts = self.tokenizer.batch_decode(beam_out, skip_special_tokens=True)

            diverse_texts = []
            actual_Rd = 0
            if cfg.use_diverse_beam:
                try:
                    div_out = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        do_sample=False,
                        num_beams=cfg.num_diverse_beams,
                        num_beam_groups=cfg.num_beam_groups,
                        diversity_penalty=cfg.diversity_penalty,
                        num_return_sequences=cfg.num_diverse_cands,
                        max_new_tokens=cfg.max_new_tokens,
                        length_penalty=cfg.length_penalty,
                        early_stopping=cfg.early_stopping,
                        repetition_penalty=cfg.repetition_penalty,
                        use_cache=True,
                    )
                    diverse_texts = self.tokenizer.batch_decode(div_out, skip_special_tokens=True)
                    actual_Rd = cfg.num_diverse_cands
                except Exception as e:
                    self.logger.warning(f"[{self.label}] Diverse beam failed: {e}")

            all_samp_texts = []
            num_temps = 0
            if cfg.use_sampling and cfg.sample_temperatures:
                num_temps = len(cfg.sample_temperatures)
                for temp in cfg.sample_temperatures:
                    try:
                        samp_out = self.model.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            do_sample=True,
                            num_beams=1,
                            top_p=cfg.mbr_top_p,
                            temperature=temp,
                            num_return_sequences=Rs,
                            max_new_tokens=cfg.max_new_tokens,
                            repetition_penalty=cfg.repetition_penalty,
                            use_cache=True,
                        )
                        all_samp_texts.extend(
                            self.tokenizer.batch_decode(samp_out, skip_special_tokens=True)
                        )
                    except Exception as e:
                        self.logger.warning(f"[{self.label}] Sampling temp={temp:.2f} failed: {e}")
                        all_samp_texts.extend([""] * (B * Rs))

        # Assemble per-sample pools
        pools = []
        for i in range(B):
            pool = list(beam_texts[i*Rb:(i+1)*Rb])
            if diverse_texts and actual_Rd > 0:
                pool.extend(diverse_texts[i*actual_Rd:(i+1)*actual_Rd])
            if all_samp_texts and num_temps > 0:
                for t_idx in range(num_temps):
                    seg = t_idx * B * Rs + i * Rs
                    pool.extend(all_samp_texts[seg:seg+Rs])
            pools.append(pool)

        if pools:
            self.logger.info(
                f"[{self.label}] pool/sample: beam={Rb} + diverse={actual_Rd} "
                f"+ sample={num_temps}×{Rs}={num_temps*Rs} = {len(pools[0])}"
            )
        return pools

    def unload(self):
        label = self.label
        try:
            from optimum.bettertransformer import BetterTransformer
            self.model = BetterTransformer.reverse(self.model)
        except Exception:
            pass
        del self.model, self.tokenizer
        self.model = self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize(self.device)
            free = (
                torch.cuda.get_device_properties(self.device).total_memory
                - torch.cuda.memory_allocated(self.device)
            ) / 1e9
            self.logger.info(f"[{label}] Unloaded. GPU free: {free:.2f} GB")

# ---------------------------------------------------------------------------
# 8. MBR selector  (unchanged)
# ---------------------------------------------------------------------------

class MBRSelector:
    def __init__(self, pool_cap=32, w_chrf=0.55, w_bleu=0.25, w_jaccard=0.20, w_length=0.10):
        self._chrf_metric = sacrebleu.metrics.CHRF(word_order=2)
        self._bleu_metric = sacrebleu.metrics.BLEU(effective_order=True)
        self.pool_cap  = pool_cap
        self.w_chrf    = w_chrf
        self.w_bleu    = w_bleu
        self.w_jaccard = w_jaccard
        self.w_length  = w_length
        self._pw_total = max(w_chrf + w_bleu + w_jaccard, 1e-9)

    def _chrfpp(self, a, b):
        if not a or not b: return 0.0
        return float(self._chrf_metric.sentence_score(a, [b]).score)

    def _bleu(self, a, b):
        if not a or not b: return 0.0
        try:    return float(self._bleu_metric.sentence_score(a, [b]).score)
        except: return 0.0

    @staticmethod
    def _jaccard(a, b):
        ta, tb = set(a.lower().split()), set(b.lower().split())
        if not ta and not tb: return 100.0
        if not ta or  not tb: return 0.0
        return 100.0 * len(ta & tb) / len(ta | tb)

    def _pairwise_score(self, a, b):
        s = (self.w_chrf * self._chrfpp(a, b) + self.w_bleu * self._bleu(a, b)
             + self.w_jaccard * self._jaccard(a, b))
        return s / self._pw_total

    @staticmethod
    def _length_bonus(lengths, idx):
        if not lengths: return 100.0
        median = float(np.median(lengths))
        sigma  = max(median * 0.4, 5.0)
        z      = (lengths[idx] - median) / sigma
        return 100.0 * math.exp(-0.5 * z * z)

    @staticmethod
    def _dedup(xs):
        seen, out = set(), []
        for x in xs:
            x = str(x).strip()
            if x and x not in seen:
                out.append(x); seen.add(x)
        return out

    def pick(self, candidates):
        cands = self._dedup(candidates)
        if self.pool_cap: cands = cands[:self.pool_cap]
        n = len(cands)
        if n == 0: return ""
        if n == 1: return cands[0]
        lengths = [len(c.split()) for c in cands]
        scores  = []
        for i in range(n):
            pw = sum(self._pairwise_score(cands[i], cands[j]) for j in range(n) if j != i) / max(1, n-1)
            scores.append(pw + self.w_length * self._length_bonus(lengths, i))
        return cands[int(np.argmax(scores))]

# ---------------------------------------------------------------------------
# 9. Engine — DUAL GPU core logic
# ---------------------------------------------------------------------------

class EnsembleMBREngine:
    def __init__(self, cfg: EnsembleMBRConfig, logger: logging.Logger):
        self.cfg           = cfg
        self.logger        = logger
        self.preprocessor  = OptimizedPreprocessor()
        self.postprocessor = VectorizedPostprocessor()
        self.mbr = MBRSelector(
            pool_cap=cfg.mbr_pool_cap,
            w_chrf=cfg.mbr_w_chrf,
            w_bleu=cfg.mbr_w_bleu,
            w_jaccard=cfg.mbr_w_jaccard,
            w_length=cfg.mbr_w_length,
        )

    def _adaptive_beams(self, attn: torch.Tensor) -> int:
        if not self.cfg.use_adaptive_beams:
            return self.cfg.num_beams
        med = float(attn.sum(dim=1).float().median().item())
        short = max(self.cfg.num_beam_cands, self.cfg.num_beams // 2)
        return short if med < 100 else self.cfg.num_beams

    def _build_dataloader(self, dataset: AkkadianDataset, wrapper: ModelWrapper) -> DataLoader:
        if self.cfg.use_bucket_batching:
            sampler = BucketBatchSampler(
                dataset, self.cfg.batch_size, self.cfg.num_buckets, self.logger
            )
            return DataLoader(
                dataset,
                batch_sampler=sampler,
                num_workers=self.cfg.num_workers,
                collate_fn=wrapper.collate,
                pin_memory=(wrapper.device.type == "cuda"),
                # OPT: pin to the specific GPU device
                pin_memory_device=str(wrapper.device) if wrapper.device.type == "cuda" else "",
                persistent_workers=(self.cfg.num_workers > 0),
            )
        return DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            collate_fn=wrapper.collate,
            pin_memory=(wrapper.device.type == "cuda"),
            pin_memory_device=str(wrapper.device) if wrapper.device.type == "cuda" else "",
            persistent_workers=(self.cfg.num_workers > 0),
        )

    def _run_one_model(self, wrapper: ModelWrapper, dataset: AkkadianDataset) -> Dict[str, List[str]]:
        """Run inference for one model; thread-safe (each model owns its device)."""
        dl = self._build_dataloader(dataset, wrapper)
        pools_by_id: Dict[str, List[str]] = {}

        with torch.inference_mode():
            for batch_ids, enc in tqdm(dl, desc=f"  [{wrapper.label} @ {wrapper.device}]"):
                # OPT: async H→D copy inside the model's dedicated stream
                if wrapper.stream is not None:
                    with torch.cuda.stream(wrapper.stream):
                        input_ids = enc.input_ids.to(wrapper.device, non_blocking=True)
                        attn      = enc.attention_mask.to(wrapper.device, non_blocking=True)
                    torch.cuda.current_stream(wrapper.device).wait_stream(wrapper.stream)
                else:
                    input_ids = enc.input_ids.to(wrapper.device)
                    attn      = enc.attention_mask.to(wrapper.device)

                beam_size = self._adaptive_beams(attn)

                try:
                    batch_pools = wrapper.generate_candidates(input_ids, attn, beam_size)
                    for sid, pool in zip(batch_ids, batch_pools):
                        pools_by_id[str(sid)] = pool
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        self.logger.error(f"OOM [{wrapper.label}] — skipping batch")
                        torch.cuda.empty_cache()
                        for sid in batch_ids:
                            pools_by_id.setdefault(str(sid), [])
                    else:
                        raise
                except Exception as e:
                    self.logger.error(f"[{wrapper.label}] batch error: {e}")
                    for sid in batch_ids:
                        pools_by_id.setdefault(str(sid), [])

                if wrapper.device.type == "cuda":
                    torch.cuda.empty_cache()

        return pools_by_id

    def run(self, test_df: pd.DataFrame) -> pd.DataFrame:
        cfg, logger = self.cfg, self.logger

        cands_per_sample = (
            cfg.num_beam_cands
            + (cfg.num_diverse_cands if cfg.use_diverse_beam else 0)
            + cfg.num_sample_cands
        )

        logger.info("=" * 65)
        logger.info("Ensemble × MBR  |  Dual-T4 Optimized  v3-t4")
        logger.info(f"  GPUs available    : {_num_gpus()}")
        logger.info(f"  Model A           : {cfg.model_a_path}  → {cfg.device_a}")
        logger.info(f"  Model B           : {cfg.model_b_path}  → {cfg.device_b}")
        logger.info(f"  Dual GPU          : {cfg.dual_gpu}")
        logger.info(f"  AMP dtype A/B     : {cfg.amp_dtype_a} / {cfg.amp_dtype_b}")
        logger.info(f"  torch.compile     : {cfg.use_torch_compile}")
        logger.info(f"  batch_size        : {cfg.batch_size}")
        logger.info(f"  Beam cands/sample : {cands_per_sample}")
        logger.info("=" * 65)

        dataset    = AkkadianDataset(test_df, self.preprocessor, logger)
        sample_ids = [str(s) for s in dataset.sample_ids]

        # ----------------------------------------------------------------
        # OPT: Parallel inference — load both models and run simultaneously
        # ----------------------------------------------------------------
        wrapper_a = ModelWrapper(cfg.model_a_path, cfg, logger, "Model-A", cfg.device_a, cfg.use_amp_a)
        wrapper_b = ModelWrapper(cfg.model_b_path, cfg, logger, "Model-B", cfg.device_b, cfg.use_amp_b)

        if cfg.dual_gpu:
            logger.info("Running both models in PARALLEL on separate GPUs")
            pools_a: Dict[str, List[str]] = {}
            pools_b: Dict[str, List[str]] = {}

            def _run_a():
                pools_a.update(self._run_one_model(wrapper_a, dataset))

            def _run_b():
                pools_b.update(self._run_one_model(wrapper_b, dataset))

            # Each thread owns its CUDA context → no locking needed
            t_a = threading.Thread(target=_run_a, name="inference-model-a")
            t_b = threading.Thread(target=_run_b, name="inference-model-b")
            t_a.start()
            t_b.start()
            t_a.join()
            t_b.join()
            logger.info("Both models finished.")
        else:
            logger.info("Single GPU — running models sequentially")
            pools_a = self._run_one_model(wrapper_a, dataset)
            wrapper_a.unload()
            del wrapper_a
            pools_b = self._run_one_model(wrapper_b, dataset)

        wrapper_b.unload()
        del wrapper_a, wrapper_b

        # ----------------------------------------------------------------
        # MBR merge
        # ----------------------------------------------------------------
        logger.info("Pool merge + MBR selection")
        results = []

        for sid in tqdm(sample_ids, desc="  MBR"):
            combined = pools_a.get(sid, []) + pools_b.get(sid, [])
            pp       = self.postprocessor.postprocess_batch(combined) if combined else []
            chosen   = self.mbr.pick(pp)

            if not chosen or not chosen.strip():
                chosen = "The tablet is too damaged to translate."

            results.append((sid, chosen))

            if len(results) % cfg.checkpoint_freq == 0:
                ckpt = Path(cfg.output_dir) / f"checkpoint_{len(results)}.csv"
                pd.DataFrame(results, columns=["id","translation"]).to_csv(ckpt, index=False)
                logger.info(f"  Checkpoint {len(results)} → {ckpt}")

        result_df = pd.DataFrame(results, columns=["id","translation"])
        self._validate(result_df)
        return result_df

    def _validate(self, df: pd.DataFrame):
        logger = self.logger
        logger.info("=" * 65)
        empty = df["translation"].str.strip().eq("").sum()
        lens  = df["translation"].str.len()
        logger.info(f"Empty     : {empty} ({100*empty/max(1,len(df)):.2f}%)")
        logger.info(f"Len mean  : {lens.mean():.1f}  median: {lens.median():.1f}  "
                    f"min: {lens.min()}  max: {lens.max()}")
        for idx in [0, len(df)//4, len(df)//2, 3*len(df)//4, len(df)-1]:
            row = df.iloc[idx]
            logger.info(f"  ID {row['id']}: {str(row['translation'])[:80]}")
        logger.info("=" * 65)

# ---------------------------------------------------------------------------
# 10. Environment summary
# ---------------------------------------------------------------------------

def print_env(cfg: EnsembleMBRConfig):
    print(f"PyTorch  : {torch.__version__}")
    print(f"GPUs     : {_num_gpus()}")
    for i in range(_num_gpus()):
        dev   = torch.device(f"cuda:{i}")
        mem   = torch.cuda.get_device_properties(i).total_memory / 1e9
        bf16  = _bf16_supported(dev)
        fp16  = _fp16_supported(dev)
        dtype = "bf16" if bf16 else ("fp16" if fp16 else "fp32")
        print(f"  GPU {i}  : {torch.cuda.get_device_name(i)}  "
              f"{mem:.1f} GB  best_amp={dtype}")
    print(f"Dual GPU : {cfg.dual_gpu}")
    print(f"torch.compile: {cfg.use_torch_compile}")
    print()
    cands = (cfg.num_beam_cands
             + (cfg.num_diverse_cands if cfg.use_diverse_beam else 0)
             + cfg.num_sample_cands)
    print(f"Candidates/sample/model : {cands}")
    print(f"Total pool (2 models)   : ~{cands*2} (before dedup)")
    print()

# ---------------------------------------------------------------------------
# 11. Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg    = EnsembleMBRConfig()
    logger = setup_logging(cfg.output_dir)

    print_env(cfg)

    logger.info(f"Loading test data: {cfg.test_data_path}")
    test_df = pd.read_csv(cfg.test_data_path, encoding="utf-8")
    logger.info(f"Test samples: {len(test_df)}")

    engine     = EnsembleMBREngine(cfg, logger)
    results_df = engine.run(test_df)

    out_path = Path(cfg.output_dir) / "submission.csv"
    results_df.to_csv(out_path, index=False)
    logger.info(f"Saved → {out_path}  ({len(results_df)} rows)")

    cfg_snap = {
        k: getattr(cfg, k) for k in (
            "model_a_path","model_b_path","gpu_a","gpu_b","dual_gpu",
            "num_beam_cands","num_beams","length_penalty","repetition_penalty",
            "use_diverse_beam","num_diverse_cands","num_diverse_beams",
            "num_beam_groups","diversity_penalty","use_sampling",
            "sample_temperatures","num_sample_per_temp","num_sample_cands",
            "mbr_top_p","mbr_w_chrf","mbr_w_bleu","mbr_w_jaccard","mbr_w_length",
            "mbr_pool_cap","max_new_tokens","amp_dtype_a","amp_dtype_b",
            "batch_size","use_torch_compile",
        )
    }
    with open(Path(cfg.output_dir) / "ensemble_mbr_config.json", "w") as f:
        json.dump(cfg_snap, f, indent=2, default=str)

    print("Submission :", out_path)
    print("Config     :", Path(cfg.output_dir) / "ensemble_mbr_config.json")