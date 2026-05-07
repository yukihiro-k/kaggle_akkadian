#!/usr/bin/env python3
"""
Deep Past Challenge — Akkadian-to-English Translation
Ensemble MBR inference — v3.4 SPEED-GUARANTEED

All v3.2 quality improvements + aggressive speed for 9h Kaggle limit:
  1. Dual pass DISABLED (halves generation time — the real bottleneck)
  2. Sampling reduced to 2 temps (fewer generate calls)
  3. MBR: chrF-only pairwise, pool cap 30
  4. All quality wins preserved: PN repair, postprocessing, pre-filter
"""

import os, gc, re, json, math, random, logging, warnings, threading
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
    if device.type != "cuda":
        return False
    try:
        return torch.cuda.get_device_properties(device).major >= 6
    except Exception:
        return False

def _bf16_supported(device: torch.device) -> bool:
    if device.type != "cuda":
        return False
    try:
        return bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    except Exception:
        return False

def _amp_ctx(device: torch.device, enabled: bool):
    if not enabled or device.type != "cuda":
        return nullcontext()
    if _bf16_supported(device):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if _fp16_supported(device):
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()

def _best_amp_dtype(device: torch.device) -> str:
    if _bf16_supported(device):
        return "bf16"
    if _fp16_supported(device):
        return "fp16"
    return "none"

# ---------------------------------------------------------------------------
# 2. MBR weight presets
# ---------------------------------------------------------------------------

MBR_PRESETS = {
    "balanced":   {"chrf": 0.55, "bleu": 0.25, "jaccard": 0.20, "length": 0.10},
    "geomean":    {"chrf": 0.45, "bleu": 0.40, "jaccard": 0.10, "length": 0.05},
    "chrf_heavy": {"chrf": 0.70, "bleu": 0.15, "jaccard": 0.10, "length": 0.05},
    "consensus":  {"chrf": 0.40, "bleu": 0.20, "jaccard": 0.30, "length": 0.10},
    "pure_chrf":  {"chrf": 1.00, "bleu": 0.00, "jaccard": 0.00, "length": 0.00},
}

# ---------------------------------------------------------------------------
# 3. Configuration
# ---------------------------------------------------------------------------

@dataclass
class EnsembleMBRConfig:
    test_data_path: str = "/kaggle/input/competitions/deep-past-initiative-machine-translation/test.csv"
    output_dir:     str = "/kaggle/working/"
    lexicon_path:   str = "/kaggle/input/competitions/deep-past-initiative-machine-translation/OA_Lexicon_eBL.csv"

    # --- N-model paths (add as many as you want) -----------------------
    model_paths: List[str] = field(default_factory=lambda: [
        "/kaggle/input/datasets/assiaben/final-byt5/byt5-akkadian-optimized-34x",
        "/kaggle/input/models/mattiaangeli/byt5-akkadian-mbr-v2/pytorch/default/1",
    ])

    max_input_length: int = 512
    max_new_tokens:   int = 384
    batch_size:       int = 8
    num_workers:      int = 4
    num_buckets:      int = 6

    # --- Beam search (v3 proven values) --------------------------------
    num_beam_cands:      int   = 4
    num_beams:           int   = 8
    length_penalty:      float = 1.3
    early_stopping:      bool  = True
    repetition_penalty:  float = 1.2

    # --- Diverse beam search -------------------------------------------
    use_diverse_beam:    bool  = True
    num_diverse_cands:   int   = 4
    num_diverse_beams:   int   = 8
    num_beam_groups:     int   = 4
    diversity_penalty:   float = 0.8

    # --- Sampling (tighter temperatures for quality) -------------------
    use_sampling:        bool        = True
    sample_temperatures: List[float] = field(default_factory=lambda: [0.6, 0.8])
    num_sample_per_temp: int         = 1    # v3.4: keep minimal
    mbr_top_p:           float       = 0.92

    @property
    def num_sample_cands(self) -> int:
        return len(self.sample_temperatures) * self.num_sample_per_temp

    # --- Dual-pass generation (per model) ------------------------------
    use_dual_pass:          bool  = False   # v3.4: DISABLED to halve generation time
    dual_pass_length_penalty: float = 1.15

    # --- MBR selection -------------------------------------------------
    mbr_preset:    str   = "geomean"    # optimised for competition metric (geomean of BLEU + chrF++)
    mbr_pool_cap:  int   = 30          # v3.3: reduced from 48 for 2.6× MBR speedup

    # --- Final polishing -----------------------------------------------
    use_final_polish:       bool  = True
    badness_threshold:      float = 3.0   # raised from 2.5 to only cleanup truly bad outputs
    min_words_fallback:     int   = 3

    # --- Runtime -------------------------------------------------------
    use_mixed_precision:    bool = True
    use_better_transformer: bool = True
    use_bucket_batching:    bool = True
    use_adaptive_beams:     bool = True
    use_torch_compile:      bool = True
    checkpoint_freq: int = 200

    def __post_init__(self):
        n_gpus = _num_gpus()
        n_models = len(self.model_paths)

        # Assign each model to a GPU (round-robin)
        if n_gpus == 0:
            self.model_devices = [torch.device("cpu")] * n_models
            self.use_mixed_precision    = False
            self.use_better_transformer = False
            self.use_torch_compile      = False
        elif n_gpus == 1:
            self.model_devices = [torch.device("cuda:0")] * n_models
        else:
            self.model_devices = [
                torch.device(f"cuda:{i % n_gpus}") for i in range(n_models)
            ]

        # Check if we can run any models in parallel (on different GPUs)
        unique_devices = set(str(d) for d in self.model_devices)
        self.can_parallel = len(unique_devices) > 1 and n_gpus > 1

        # Legacy compat
        self.device = self.model_devices[0] if self.model_devices else torch.device("cpu")

        Path(self.output_dir).mkdir(exist_ok=True, parents=True)

        # Resolve MBR weights from preset
        preset = MBR_PRESETS.get(self.mbr_preset, MBR_PRESETS["balanced"])
        self.mbr_w_chrf    = preset["chrf"]
        self.mbr_w_bleu    = preset["bleu"]
        self.mbr_w_jaccard = preset["jaccard"]
        self.mbr_w_length  = preset["length"]

        assert self.num_beams >= self.num_beam_cands
        if self.use_diverse_beam:
            assert self.num_diverse_beams % self.num_beam_groups == 0
            assert self.num_diverse_beams >= self.num_diverse_cands

# ---------------------------------------------------------------------------
# 4. Logging
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
# 5. Preprocessing
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
# 6. Postprocessing (merged from all codebases)
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
_FORBIDDEN_TRANS = str.maketrans("", "", '——<>⌈⌋⌊[]+ʾ;')  # FIX: () preserved (appear in test set)
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
        # v3.2: REMOVED PN→<gap> — proper nouns should stay for BLEU/chrF
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
# 7. Badness scoring + aggressive cleanup (from byt5_inference.py)
# ---------------------------------------------------------------------------

_AGG_REMOVE_NOTES = re.compile(
    r"\((?:plur\.?|sing\.?|fem\.?|masc\.?|uncertain|\?|\!|damaged|broken)\)",
    flags=re.I
)
_AGG_REMOVE_WEIRD = re.compile(r"[<>⌈⌋⌊⌉+ʾ/;]")
_AGG_GAP_RUNS     = re.compile(r"(?:<gap>\s*){2,}")
_AGG_REPEAT_WORD  = re.compile(r"\b(\w+)(\s+\1){1,}\b", flags=re.I)

def badness_score(text: str) -> float:
    """Lower is better. Penalise short, long, gappy, repetitive outputs."""
    t = (text or "").strip()
    if not t:
        return 10.0
    words = t.split()
    n = len(words)
    score = 0.0
    if n < 5:
        score += 3.0
    if n < 3:
        score += 3.0
    if n > 500:
        score += 2.0
    if n > 650:
        score += 3.0
    gaps = t.count("<gap>")
    if gaps > 6:
        score += (gaps - 6) * 0.35
    # 3+ consecutive same word
    for i in range(2, n):
        if words[i].lower() == words[i-1].lower() == words[i-2].lower():
            score += 0.75
    # high bigram repetitiveness
    if n >= 20:
        bigrams = list(zip(words, words[1:]))
        uniq = len(set(bigrams))
        if uniq > 0:
            rep = 1.0 - (uniq / max(1, len(bigrams)))
            if rep > 0.35:
                score += (rep - 0.35) * 6.0
    return score


def _aggressive_cleanup(text: str) -> str:
    """Extra cleanup for outputs with high badness."""
    t = str(text or "")
    t = _AGG_REMOVE_NOTES.sub("", t)
    t = _AGG_GAP_RUNS.sub("<gap> ", t)
    t = _AGG_REMOVE_WEIRD.sub(" ", t)
    t = _AGG_REPEAT_WORD.sub(r"\1", t)
    # n-gram dedup
    tokens = t.split()
    if len(tokens) >= 12:
        for ng in range(2, 5):
            i, out = 0, []
            while i < len(tokens):
                if i + 2*ng <= len(tokens) and tokens[i:i+ng] == tokens[i+ng:i+2*ng]:
                    out.extend(tokens[i:i+ng])
                    i += 2 * ng
                else:
                    out.append(tokens[i])
                    i += 1
            tokens = out
        t = " ".join(tokens)
    t = re.sub(r"\s+([,.;:!?])", r"\1", t)
    t = re.sub(r"([!?.,])\1{2,}", r"\1", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _final_polish(text: str, min_words: int = 3) -> str:
    """Capitalise, punctuate, and provide fallback for empty/short outputs."""
    t = (text or "").strip()
    if not t:
        return "The tablet is too damaged to translate."
    words = t.split()
    if len(words) < min_words:
        return "The tablet contains an incomplete inscription."
    # capitalise first letter
    if t[0].isalpha() and t[0].islower():
        t = t[0].upper() + t[1:]
    # add ending punctuation if missing
    if not t.endswith((".", "!", "?")):
        t += "."
    t = re.sub(r"\s+([,.;:!?])", r"\1", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t
# ---------------------------------------------------------------------------
# 7b. Lexicon-based proper noun repair (v3.2)
# ---------------------------------------------------------------------------

class ProperNounRepairer:
    """Use OA_Lexicon_eBL.csv to fix misspelled proper nouns in translations.
    
    Strategy:
    - Extract PN (person names) and GN (geographic names) from lexicon
    - Build a lookup from normalized form → canonical spelling
    - After MBR selects a candidate, scan for capitalized words
    - If a word is close to a known proper noun, replace with canonical form
    """
    
    def __init__(self, lexicon_path: str, logger: logging.Logger):
        self.logger = logger
        self.pn_lookup: Dict[str, str] = {}  # lowercase → canonical
        self._loaded = False
        
        if not os.path.exists(lexicon_path):
            logger.warning(f"Lexicon not found: {lexicon_path} — PN repair disabled")
            return
        
        try:
            lex_df = pd.read_csv(lexicon_path, encoding="utf-8")
            logger.info(f"Loaded lexicon: {len(lex_df)} entries from {lexicon_path}")
            
            # Filter to proper nouns (PN = person name, GN = geographic name)
            pn_types = {"PN", "GN", "DN", "TN", "RN"}  # person, geo, divine, temple, royal
            pn_df = lex_df[lex_df["type"].isin(pn_types)] if "type" in lex_df.columns else pd.DataFrame()
            
            if len(pn_df) == 0:
                logger.warning("No proper nouns found in lexicon")
                return
            
            # Build lookup: use 'norm' or 'lemma' as canonical, 'form' as key
            for _, row in pn_df.iterrows():
                form = str(row.get("form", "")).strip().replace("-", "")
                norm = str(row.get("norm", "")).strip().replace("-", "")
                lexeme = str(row.get("lexeme", "")).strip()
                
                if not form or form == "nan":
                    continue
                
                # Canonical name: prefer norm, fallback to lexeme, then form
                canonical = norm if norm and norm != "nan" else (
                    lexeme if lexeme and lexeme != "nan" else form
                )
                
                # Capitalize first letter for proper noun
                if canonical and canonical[0].islower():
                    canonical = canonical[0].upper() + canonical[1:]
                
                self.pn_lookup[form.lower()] = canonical
                if norm and norm != "nan":
                    self.pn_lookup[norm.lower()] = canonical
            
            self._loaded = True
            logger.info(f"PN repairer: {len(self.pn_lookup)} proper noun variants loaded")
            
        except Exception as e:
            logger.warning(f"Failed to load lexicon: {e}")
    
    def repair(self, text: str, source_translit: str = "") -> str:
        """Fix proper nouns in the translation using lexicon lookup."""
        if not self._loaded or not text:
            return text
        
        words = text.split()
        repaired = []
        changed = False
        
        for word in words:
            # Strip punctuation for matching
            stripped = word.strip(".,;:!?\"'()[]")
            lower = stripped.lower()
            
            # Only repair capitalized words (likely proper nouns)
            if stripped and stripped[0].isupper() and len(stripped) >= 3:
                if lower in self.pn_lookup:
                    canonical = self.pn_lookup[lower]
                    # Preserve trailing punctuation
                    suffix = word[len(stripped):] if len(word) > len(stripped) else ""
                    prefix = word[:word.index(stripped)] if word.index(stripped) > 0 else ""
                    repaired.append(prefix + canonical + suffix)
                    changed = True
                    continue
            
            repaired.append(word)
        
        return " ".join(repaired) if changed else text


# ---------------------------------------------------------------------------
# 8. Dataset + bucket batching
# ---------------------------------------------------------------------------

class AkkadianDataset(Dataset):
    def __init__(self, df: pd.DataFrame, preprocessor: OptimizedPreprocessor, logger: logging.Logger):
        self.sample_ids = df["id"].tolist()
        raw_texts = df["transliteration"].tolist()
        self.raw_transliterations = {str(sid): str(t) for sid, t in zip(self.sample_ids, raw_texts)}  # v3.2
        proc = preprocessor.preprocess_batch(raw_texts)
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
# 9. Model wrapper
# ---------------------------------------------------------------------------

class ModelWrapper:
    def __init__(
        self,
        model_path: str,
        cfg: EnsembleMBRConfig,
        logger: logging.Logger,
        label: str,
        device: torch.device,
    ):
        self.cfg     = cfg
        self.logger  = logger
        self.label   = label
        self.device  = device
        self.use_amp = cfg.use_mixed_precision and device.type == "cuda"

        logger.info(f"[{label}] Loading → {device}  (path: {model_path})")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

        # FP16 weight cast for T4
        if device.type == "cuda" and _fp16_supported(device) and not _bf16_supported(device):
            model = model.half()
            logger.info(f"[{label}] Weights cast to FP16 for T4")

        self.model = model.to(device).eval()

        if device.type == "cuda":
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

        if cfg.use_torch_compile and device.type == "cuda":
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead", fullgraph=False)
                logger.info(f"[{label}] torch.compile applied")
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
            logger.info(f"[{label}] GPU mem used: {used:.2f} GB")

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

    def generate_candidates(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        beam_size: int,
        length_penalty_override: Optional[float] = None,
    ) -> List[List[str]]:
        """Generate beam + diverse + sampling candidates for one pass."""
        cfg = self.cfg
        B   = input_ids.shape[0]
        ctx = _amp_ctx(self.device, self.use_amp)

        lp = length_penalty_override if length_penalty_override is not None else cfg.length_penalty

        Rb = cfg.num_beam_cands
        Rs = cfg.num_sample_per_temp

        with ctx:
            # --- Standard beam search ---
            nb = max(beam_size, Rb)
            beam_out = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                num_beams=nb,
                num_return_sequences=Rb,
                max_new_tokens=cfg.max_new_tokens,
                length_penalty=lp,
                early_stopping=cfg.early_stopping,
                repetition_penalty=cfg.repetition_penalty,
                use_cache=True,
            )
            beam_texts = self.tokenizer.batch_decode(beam_out, skip_special_tokens=True)

            # --- Diverse beam search ---
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
                        length_penalty=lp,
                        early_stopping=cfg.early_stopping,
                        repetition_penalty=cfg.repetition_penalty,
                        use_cache=True,
                    )
                    diverse_texts = self.tokenizer.batch_decode(div_out, skip_special_tokens=True)
                    actual_Rd = cfg.num_diverse_cands
                except Exception as e:
                    self.logger.warning(f"[{self.label}] Diverse beam failed: {e}")

            # --- Multi-temperature sampling ---
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
# 10. MBR selector
# ---------------------------------------------------------------------------

class MBRSelector:
    """v3.2: MBR with geometric-mean scoring + pre-filtering."""

    def __init__(self, pool_cap=48, w_chrf=0.45, w_bleu=0.40, w_jaccard=0.10, w_length=0.05):
        self._chrf_metric = sacrebleu.metrics.CHRF(word_order=2)
        self._bleu_metric = sacrebleu.metrics.BLEU(effective_order=True)
        self.pool_cap  = pool_cap
        self.w_chrf    = w_chrf
        self.w_bleu    = w_bleu
        self.w_jaccard = w_jaccard
        self.w_length  = w_length

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
        """v3.3: chrF++ only for speed (word_order=2 already captures word-level quality)."""
        chrf_s = self._chrfpp(a, b)
        jacc_s = self._jaccard(a, b)
        # chrF++ with word_order=2 is the best single proxy for quality
        return chrf_s + self.w_jaccard * jacc_s

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

    @staticmethod
    def _prefilter(cands):
        """v3.2: Remove obviously bad candidates before MBR."""
        if len(cands) <= 3:
            return cands  # don't filter if too few

        lengths = [len(c.split()) for c in cands]
        median_len = float(np.median(lengths)) if lengths else 10.0

        filtered = []
        for c, wlen in zip(cands, lengths):
            # Skip very short (< 2 words)
            if wlen < 2:
                continue
            # Skip absurdly long (> 4× median)
            if median_len > 0 and wlen > median_len * 4:
                continue
            # Skip if > 50% bigram repetition
            if wlen >= 8:
                words = c.split()
                bigrams = list(zip(words, words[1:]))
                if bigrams:
                    uniq_ratio = len(set(bigrams)) / len(bigrams)
                    if uniq_ratio < 0.4:
                        continue
            filtered.append(c)

        return filtered if len(filtered) >= 2 else cands

    def pick(self, candidates):
        cands = self._dedup(candidates)
        cands = self._prefilter(cands)              # v3.2: pre-filter
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
# 11. Engine — N-model, dual-pass
# ---------------------------------------------------------------------------

class EnsembleMBREngine:
    def __init__(self, cfg: EnsembleMBRConfig, logger: logging.Logger):
        self.cfg           = cfg
        self.logger        = logger
        self.preprocessor  = OptimizedPreprocessor()
        self.postprocessor = VectorizedPostprocessor()
        self.pn_repairer   = ProperNounRepairer(cfg.lexicon_path, logger)  # v3.2
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

    def _run_one_model(
        self,
        wrapper: ModelWrapper,
        dataset: AkkadianDataset,
        length_penalties: List[float],
    ) -> Dict[str, List[str]]:
        """Run inference for one model with one or more length_penalty passes."""
        dl = self._build_dataloader(dataset, wrapper)
        pools_by_id: Dict[str, List[str]] = {}

        for pass_idx, lp in enumerate(length_penalties):
            pass_label = f"{wrapper.label}/pass{pass_idx}(lp={lp})"
            self.logger.info(f"  [{pass_label}] Starting...")

            with torch.inference_mode():
                for batch_ids, enc in tqdm(dl, desc=f"  [{pass_label}]"):
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
                        batch_pools = wrapper.generate_candidates(
                            input_ids, attn, beam_size, length_penalty_override=lp
                        )
                        for sid, pool in zip(batch_ids, batch_pools):
                            sid_str = str(sid)
                            if sid_str not in pools_by_id:
                                pools_by_id[sid_str] = []
                            pools_by_id[sid_str].extend(pool)
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            self.logger.error(f"OOM [{pass_label}] — skipping batch")
                            torch.cuda.empty_cache()
                            for sid in batch_ids:
                                pools_by_id.setdefault(str(sid), [])
                        else:
                            raise
                    except Exception as e:
                        self.logger.error(f"[{pass_label}] batch error: {e}")
                        for sid in batch_ids:
                            pools_by_id.setdefault(str(sid), [])

                    if wrapper.device.type == "cuda":
                        torch.cuda.empty_cache()

        return pools_by_id

    def run(self, test_df: pd.DataFrame) -> pd.DataFrame:
        cfg, logger = self.cfg, self.logger
        n_models = len(cfg.model_paths)

        # Determine length_penalty passes
        if cfg.use_dual_pass:
            lp_passes = [cfg.length_penalty, cfg.dual_pass_length_penalty]
        else:
            lp_passes = [cfg.length_penalty]

        cands_per_pass = (
            cfg.num_beam_cands
            + (cfg.num_diverse_cands if cfg.use_diverse_beam else 0)
            + cfg.num_sample_cands
        )
        total_cands = cands_per_pass * len(lp_passes) * n_models

        logger.info("=" * 65)
        logger.info("Ensemble × MBR  |  v3 SCORE-OPTIMISED")
        logger.info(f"  GPUs available    : {_num_gpus()}")
        logger.info(f"  Models            : {n_models}")
        for i, (path, dev) in enumerate(zip(cfg.model_paths, cfg.model_devices)):
            logger.info(f"    [{i}] {path}  → {dev} ({_best_amp_dtype(dev)})")
        logger.info(f"  Parallel capable  : {cfg.can_parallel}")
        logger.info(f"  LP passes         : {lp_passes}")
        logger.info(f"  MBR preset        : {cfg.mbr_preset} (chrf={cfg.mbr_w_chrf}, bleu={cfg.mbr_w_bleu}, "
                     f"jaccard={cfg.mbr_w_jaccard}, length={cfg.mbr_w_length})")
        logger.info(f"  Cands/sample/pass : {cands_per_pass}")
        logger.info(f"  Total candidates  : ~{total_cands} per sample (before dedup)")
        logger.info(f"  MBR pool cap      : {cfg.mbr_pool_cap}")
        logger.info(f"  batch_size        : {cfg.batch_size}")
        logger.info(f"  Final polish      : {cfg.use_final_polish}")
        logger.info("=" * 65)

        dataset    = AkkadianDataset(test_df, self.preprocessor, logger)
        sample_ids = [str(s) for s in dataset.sample_ids]

        # ----------------------------------------------------------------
        # Run inference for each model
        # ----------------------------------------------------------------
        all_pools: List[Dict[str, List[str]]] = []

        # Group models by device for potential parallelism
        device_groups: Dict[str, List[int]] = {}
        for i, dev in enumerate(cfg.model_devices):
            key = str(dev)
            device_groups.setdefault(key, []).append(i)

        # Check if we can run models in parallel (different GPUs)
        if cfg.can_parallel:
            logger.info("Running models in PARALLEL where possible")

            # Group models that are on different GPUs
            parallel_pairs = []
            remaining = list(range(n_models))

            # Pair up models on different GPUs
            while len(remaining) >= 2:
                m0 = remaining[0]
                partner = None
                for j in range(1, len(remaining)):
                    m1 = remaining[j]
                    if str(cfg.model_devices[m0]) != str(cfg.model_devices[m1]):
                        partner = j
                        break
                if partner is not None:
                    parallel_pairs.append((remaining[0], remaining[partner]))
                    remaining.pop(partner)
                    remaining.pop(0)
                else:
                    break  # no more cross-device pairs

            # Run parallel pairs
            for m_a, m_b in parallel_pairs:
                logger.info(f"  Parallel: Model-{m_a} + Model-{m_b}")
                w_a = ModelWrapper(cfg.model_paths[m_a], cfg, logger,
                                   f"Model-{m_a}", cfg.model_devices[m_a])
                w_b = ModelWrapper(cfg.model_paths[m_b], cfg, logger,
                                   f"Model-{m_b}", cfg.model_devices[m_b])

                result_a: Dict[str, List[str]] = {}
                result_b: Dict[str, List[str]] = {}

                def _run_a():
                    result_a.update(self._run_one_model(w_a, dataset, lp_passes))
                def _run_b():
                    result_b.update(self._run_one_model(w_b, dataset, lp_passes))

                t_a = threading.Thread(target=_run_a, name=f"model-{m_a}")
                t_b = threading.Thread(target=_run_b, name=f"model-{m_b}")
                t_a.start(); t_b.start()
                t_a.join();  t_b.join()

                w_a.unload(); w_b.unload()
                del w_a, w_b
                all_pools.append(result_a)
                all_pools.append(result_b)
                logger.info(f"  Models {m_a}+{m_b} finished.")

            # Run remaining models sequentially
            for m_i in remaining:
                logger.info(f"  Sequential: Model-{m_i}")
                w = ModelWrapper(cfg.model_paths[m_i], cfg, logger,
                                 f"Model-{m_i}", cfg.model_devices[m_i])
                result = self._run_one_model(w, dataset, lp_passes)
                w.unload()
                del w
                all_pools.append(result)
        else:
            # Sequential: load → run → unload each model
            logger.info("Running models SEQUENTIALLY")
            for m_i in range(n_models):
                logger.info(f"  Model-{m_i}: {cfg.model_paths[m_i]}")
                w = ModelWrapper(cfg.model_paths[m_i], cfg, logger,
                                 f"Model-{m_i}", cfg.model_devices[m_i])
                result = self._run_one_model(w, dataset, lp_passes)
                w.unload()
                del w
                all_pools.append(result)

        # ----------------------------------------------------------------
        # Pool merge + MBR selection + post-processing
        # ----------------------------------------------------------------
        logger.info("Pool merge + MBR selection + post-processing")
        results = []

        for sid in tqdm(sample_ids, desc="  MBR"):
            # Merge candidates from all models
            combined = []
            for pool_dict in all_pools:
                combined.extend(pool_dict.get(sid, []))

            # Postprocess all candidates
            pp = self.postprocessor.postprocess_batch(combined) if combined else []

            # MBR selection
            chosen = self.mbr.pick(pp)

            # v3.2: Proper noun repair using lexicon
            src_translit = dataset.raw_transliterations.get(sid, "")
            chosen = self.pn_repairer.repair(chosen, src_translit)

            # Final polish (from byt5_inference.py)
            if cfg.use_final_polish:
                if not chosen or not chosen.strip():
                    chosen = "The tablet is too damaged to translate."
                else:
                    # Conditional aggressive cleanup for bad outputs
                    if badness_score(chosen) >= cfg.badness_threshold:
                        chosen = _aggressive_cleanup(chosen)
                    chosen = _final_polish(chosen, cfg.min_words_fallback)
            else:
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
# 12. Environment summary
# ---------------------------------------------------------------------------

def print_env(cfg: EnsembleMBRConfig):
    print(f"PyTorch  : {torch.__version__}")
    print(f"GPUs     : {_num_gpus()}")
    for i in range(_num_gpus()):
        dev   = torch.device(f"cuda:{i}")
        mem   = torch.cuda.get_device_properties(i).total_memory / 1e9
        dtype = _best_amp_dtype(dev)
        print(f"  GPU {i}  : {torch.cuda.get_device_name(i)}  "
              f"{mem:.1f} GB  best_amp={dtype}")
    print(f"Parallel : {cfg.can_parallel}")
    print(f"Models   : {len(cfg.model_paths)}")
    print(f"MBR preset: {cfg.mbr_preset}")
    print()

    lp_passes = 2 if cfg.use_dual_pass else 1
    cands = (cfg.num_beam_cands
             + (cfg.num_diverse_cands if cfg.use_diverse_beam else 0)
             + cfg.num_sample_cands)
    n_models = len(cfg.model_paths)
    print(f"Candidates/sample/pass/model : {cands}")
    print(f"LP passes per model          : {lp_passes}")
    print(f"Total pool ({n_models} models)          : ~{cands * lp_passes * n_models} (before dedup)")
    print()

# ---------------------------------------------------------------------------
# 13. Main
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
        "model_paths": cfg.model_paths,
        "num_models": len(cfg.model_paths),
        "mbr_preset": cfg.mbr_preset,
        "mbr_w_chrf": cfg.mbr_w_chrf,
        "mbr_w_bleu": cfg.mbr_w_bleu,
        "mbr_w_jaccard": cfg.mbr_w_jaccard,
        "mbr_w_length": cfg.mbr_w_length,
        "use_dual_pass": cfg.use_dual_pass,
        "length_penalty": cfg.length_penalty,
        "dual_pass_length_penalty": cfg.dual_pass_length_penalty,
        "num_beam_cands": cfg.num_beam_cands,
        "num_beams": cfg.num_beams,
        "repetition_penalty": cfg.repetition_penalty,
        "use_diverse_beam": cfg.use_diverse_beam,
        "num_diverse_cands": cfg.num_diverse_cands,
        "use_sampling": cfg.use_sampling,
        "sample_temperatures": cfg.sample_temperatures,
        "num_sample_per_temp": cfg.num_sample_per_temp,
        "mbr_pool_cap": cfg.mbr_pool_cap,
        "max_new_tokens": cfg.max_new_tokens,
        "batch_size": cfg.batch_size,
        "use_final_polish": cfg.use_final_polish,
        "badness_threshold": cfg.badness_threshold,
    }
    with open(Path(cfg.output_dir) / "ensemble_mbr_config.json", "w") as f:
        json.dump(cfg_snap, f, indent=2, default=str)

    print(f"Submission : {out_path}")
    print(f"Config     : {Path(cfg.output_dir) / 'ensemble_mbr_config.json'}")
