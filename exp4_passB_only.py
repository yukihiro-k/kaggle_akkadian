# ============================================================
# Robust ByT5 Inference + Two-pass Self-Ensemble (Merged)
# - Safe → Conditional Aggressive postprocess
# - Two runs (cfg A/B) + fuse (badness-first, quality tie-break)
# - Bucket batching, adaptive beams, optional BetterTransformer, optional FP16
# - Checkpointing + basic diagnostics
# ============================================================

import os
import re
import gc
import json
import math
import time
import logging
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ---------------------------
# 0) Environment knobs
# ---------------------------
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "0")
os.environ.setdefault("TORCH_CUDNN_V8_API_ENABLED", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("byt-robust")

def print_environment_info():
    logger.info(f"torch={torch.__version__}")
    logger.info(f"cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"gpu={torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        logger.info(f"gpu_mem_total_gb={props.total_memory/1024**3:.2f}")
    # BetterTransformer availability (optional)
    try:
        from optimum.bettertransformer import BetterTransformer  # noqa: F401
        logger.info("optimum.bettertransformer is available (will try to use if enabled).")
    except Exception:
        logger.info("optimum.bettertransformer not available (will skip).")

print_environment_info()

# ---------------------------
# 1) Config
# ---------------------------
@dataclass
class UltraConfig:
    # paths
    test_data_path: str = "/kaggle/input/deep-past-initiative-machine-translation/test.csv"   # <-- adjust to your dataset
    model_path: str = "/kaggle/input/final-byt5/byt5-akkadian-optimized-34x"
    output_dir: str = "/kaggle/working"

    # dataloader/tokenization
    max_length: int = 512
    batch_size: int = 8
    num_workers: int = 2
    pin_memory: bool = True

    # generation
    num_beams: int = 16
    max_new_tokens: int = 512
    length_penalty: float = 1.30
    early_stopping: bool = True
    no_repeat_ngram_size: int = 0
    repetition_penalty: Optional[float] = None

    # performance
    use_mixed_precision: bool = True
    use_better_transformer: bool = True
    use_bucket_batching: bool = True
    use_adaptive_beams: bool = False
    use_vectorized_postproc: bool = True

    # robustness/ops
    checkpoint_freq: int = 2000  # save partial predictions periodically
    empty_cache_every: int = 10  # batches
    validate_samples: int = 6

    # postprocess behavior
    postprocess_mode: str = "safe_then_conditional_aggressive"  # or "safe_only" / "aggressive_only"
    aggressive_trigger_badness: float = 2.5  # if badness >= threshold -> run aggressive refinement
    min_words_fallback: int = 3

    # device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        if self.device == "cpu":
            self.use_mixed_precision = False
            self.use_better_transformer = False
            self.pin_memory = False

# ---------------------------
# 2) Preprocess (input)
# ---------------------------
class OptimizedPreprocessor:
    def __init__(self):
        # big gaps: ... … …… etc
        self.re_big_gap = re.compile(r"(\.\.\.+|…+)")
        # small gaps: occurrences of 'x' / 'xx' in isolation-ish patterns
        # This is intentionally conservative; you can expand if needed.
        self.re_small_gap = re.compile(r"(?:(?<=\s)|^)(x{1,3})(?=(\s|$))", flags=re.IGNORECASE)

    def preprocess_batch(self, texts: List[str]) -> List[str]:
        s = pd.Series(texts, dtype="object").fillna("")
        s = s.str.replace(self.re_big_gap, " <big_gap> ", regex=True)
        s = s.str.replace(self.re_small_gap, " <gap> ", regex=True)
        s = s.str.replace(r"\s+", " ", regex=True).str.strip()
        return s.tolist()

# ---------------------------
# 3) Postprocess (output) – Safe + Aggressive
# ---------------------------
class SafePostprocessor:
    """
    Conservative cleanup.
    - Preserve common punctuation and formatting.
    - Protect <gap>/<big_gap>.
    - Normalize a few known characters.
    """
    SUBSCRIPT_TO_NORMAL = str.maketrans({
        "₀":"0","₁":"1","₂":"2","₃":"3","₄":"4","₅":"5","₆":"6","₇":"7","₈":"8","₉":"9"
    })

    def __init__(self, use_unicode_fractions: bool = False, strip_dash_old: bool = False):
        self.use_unicode_fractions = use_unicode_fractions
        self.strip_dash_old = strip_dash_old

        self.forbidden_chars = re.compile(r"[⌈⌋⌊⌉]")  # very conservative
        self.multi_space = re.compile(r"\s+")
        self.space_before_punct = re.compile(r"\s+([,.;:!?])")
        self.multi_punct = re.compile(r"([!?.,])\1{2,}")
        self.dashes = re.compile(r"[–—]")

        # Protect tokens
        self.prot_gap = "\uFFF0"
        self.prot_big = "\uFFF1"

        # Gaps normalization (mild)
        self.bracket_x = re.compile(r"\[\s*x\s*\]|\(\s*x\s*\)", re.IGNORECASE)
        self.bare_x = re.compile(r"(?:(?<=\s)|^)x(?=(\s|$))", re.IGNORECASE)
        self.ellipsis = re.compile(r"(\.\.\.+|…+)")

        # optional fractions
        self.frac_map = {
            "0.5": "½", "0.25": "¼", "0.75": "¾",
            "1/2": "½", "1/4": "¼", "3/4": "¾",
        }
        self.frac_re = re.compile(r"\b(0\.5|0\.25|0\.75|1/2|1/4|3/4)\b")

    def postprocess_one(self, text: str) -> str:
        if text is None:
            text = ""
        t = str(text)

        # Normalize known chars
        t = t.replace("ḫ", "h").replace("Ḫ", "H")
        t = t.translate(self.SUBSCRIPT_TO_NORMAL)
        t = self.dashes.sub("-", t)

        # Normalize gaps lightly
        t = self.bracket_x.sub("<gap>", t)
        t = self.bare_x.sub("<gap>", t)
        t = self.ellipsis.sub("<big_gap>", t)

        # Protect tokens before stripping
        t = t.replace("<gap>", self.prot_gap).replace("<big_gap>", self.prot_big)

        # Remove only very rare/unsafe glyphs
        t = self.forbidden_chars.sub("", t)

        # Restore protected tokens
        t = t.replace(self.prot_gap, "<gap>").replace(self.prot_big, "<big_gap>")

        # Optional: old dash stripping behavior (risky; kept for ablation)
        if self.strip_dash_old:
            t = t.strip(" -")

        # Optional unicode fractions
        if self.use_unicode_fractions:
            t = self.frac_re.sub(lambda m: self.frac_map.get(m.group(1), m.group(1)), t)

        # Whitespace & punctuation tidying (safe)
        t = self.space_before_punct.sub(r"\1", t)
        t = self.multi_punct.sub(r"\1", t)
        t = self.multi_space.sub(" ", t).strip()

        return t

    def postprocess_batch(self, texts: List[str]) -> List[str]:
        return [self.postprocess_one(x) for x in texts]


class AggressivePostprocessor:
    """
    More aggressive cleanup used only when output looks "bad".
    - Remove common bracketed grammatical notes
    - Dedupe repeated words and repeated ngrams
    - Stronger gap consolidation
    """
    def __init__(self, ngram_dedupe_max_n: int = 4):
        self.ngram_dedupe_max_n = ngram_dedupe_max_n

        self.multi_space = re.compile(r"\s+")
        self.space_before_punct = re.compile(r"\s+([,.;:!?])")
        self.multi_punct = re.compile(r"([!?.,])\1{2,}")

        self.remove_notes = re.compile(
            r"\((?:plur\.?|sing\.?|fem\.?|masc\.?|uncertain|\?|\!|damaged|broken)\)",
            flags=re.IGNORECASE
        )
        self.remove_weird = re.compile(r"[<>⌈⌋⌊⌉⌊⌋\+ʾ/;]")  # somewhat aggressive (still not too crazy)

        self.gap_runs = re.compile(r"(?:<gap>\s*){2,}")
        self.biggap_runs = re.compile(r"(?:<big_gap>\s*){2,}")

        self.repeat_word = re.compile(r"\b(\w+)(\s+\1){1,}\b", flags=re.IGNORECASE)

    def _dedupe_ngrams(self, text: str) -> str:
        # simple greedy n-gram dedupe for 2..N
        tokens = text.split()
        if len(tokens) < 12:
            return text
        for n in range(2, self.ngram_dedupe_max_n + 1):
            i = 0
            out = []
            while i < len(tokens):
                if i + 2*n <= len(tokens) and tokens[i:i+n] == tokens[i+n:i+2*n]:
                    # skip one repetition
                    out.extend(tokens[i:i+n])
                    i += 2*n
                else:
                    out.append(tokens[i])
                    i += 1
            tokens = out
        return " ".join(tokens)

    def postprocess_one(self, text: str) -> str:
        t = str(text or "")

        # remove notes
        t = self.remove_notes.sub("", t)

        # stronger gap consolidation
        t = self.gap_runs.sub("<big_gap> ", t)   # 2+ gaps -> big gap
        t = self.biggap_runs.sub("<big_gap> ", t)

        # remove weird chars
        t = self.remove_weird.sub(" ", t)

        # word repeat dedupe
        t = self.repeat_word.sub(r"\1", t)

        # ngram dedupe
        t = self._dedupe_ngrams(t)

        # normalize spaces/punct
        t = self.space_before_punct.sub(r"\1", t)
        t = self.multi_punct.sub(r"\1", t)
        t = self.multi_space.sub(" ", t).strip()

        return t

    def postprocess_batch(self, texts: List[str]) -> List[str]:
        return [self.postprocess_one(x) for x in texts]


# ---------------------------
# 4) Scoring (badness + quality)
# ---------------------------
def badness_score(text: str) -> float:
    """
    Lower is better.
    Penalize:
      - very short
      - very long
      - excessive gaps
      - heavy repetition
    """
    if text is None:
        text = ""
    t = str(text).strip()
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

    gaps = t.count("<gap>") + t.count("<big_gap>")
    if gaps > 6:
        score += (gaps - 6) * 0.35

    # 3+ consecutive repeats (same word)
    rep = 0
    for i in range(2, n):
        if words[i].lower() == words[i-1].lower() == words[i-2].lower():
            rep += 1
    score += rep * 0.75

    # long repeated bigrams (very rough)
    if n >= 20:
        bigrams = list(zip(words, words[1:]))
        uniq = len(set(bigrams))
        if uniq > 0:
            repetitiveness = 1.0 - (uniq / max(1, len(bigrams)))
            if repetitiveness > 0.35:
                score += (repetitiveness - 0.35) * 6.0

    return score


KEYWORDS = {
    "tablet", "king", "city", "god", "silver", "gold", "temple", "house", "palace",
    "year", "son", "daughter", "brother", "mother", "father", "gave", "took", "sent",
    "received", "grain", "sheep", "oil", "wine"
}

def quality_score(text: str) -> float:
    """
    Higher is better (simple heuristic).
    """
    if text is None:
        text = ""
    t = str(text).strip()
    if not t:
        return 0.0

    score = 0.0
    words = t.split()

    # length preference
    n = len(words)
    if 8 <= n <= 120:
        score += 2.0
    elif 5 <= n <= 200:
        score += 1.0

    # formatting
    if t[0].isupper():
        score += 0.5
    if t.endswith((".", "!", "?")):
        score += 0.5

    # keyword hints
    lw = {w.strip(".,;:!?\"'()[]").lower() for w in words}
    hit = len(lw.intersection(KEYWORDS))
    score += min(2.0, 0.25 * hit)

    # penalize too many gaps
    gaps = t.count("<gap>") + t.count("<big_gap>")
    score -= min(2.0, 0.15 * gaps)

    # penalize heavy repetition by badness
    score -= 0.5 * max(0.0, badness_score(t) - 1.0)

    return score


def fuse_texts(a: str, b: str, prefer: str = "a", tie_badness_thresh: float = 0.5,
               w_a: float = 0.60, w_b: float = 0.40) -> str:
    """
    Robust fuse:
      1) pick lower badness if clearly better
      2) if close, pick higher weighted quality
    """
    ba = badness_score(a)
    bb = badness_score(b)

    if ba + tie_badness_thresh < bb:
        return a
    if bb + tie_badness_thresh < ba:
        return b

    qa = quality_score(a) * w_a
    qb = quality_score(b) * w_b
    if qa > qb:
        return a
    if qb > qa:
        return b
    return a if prefer == "a" else b


# ---------------------------
# 5) Dataset + Bucket sampler
# ---------------------------
class AkkadianDataset(Dataset):
    def __init__(self, df: pd.DataFrame, preprocessor: OptimizedPreprocessor):
        self.ids = df["id"].astype(str).tolist()
        raw = df["transliteration"].astype("object").fillna("").tolist()
        proc = preprocessor.preprocess_batch(raw)
        self.inputs = [f"translate Akkadian to English: {x}" for x in proc]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return self.ids[i], self.inputs[i]


class BucketBatchSampler(Sampler[List[int]]):
    def __init__(self, texts: List[str], batch_size: int, num_buckets: int = 32, shuffle: bool = False):
        self.batch_size = batch_size
        self.shuffle = shuffle

        lengths = np.array([max(1, len(t.split())) for t in texts], dtype=np.int32)
        order = np.argsort(lengths)
        self.indices = order.tolist()

        # split into buckets
        self.num_buckets = max(1, num_buckets)
        self.buckets = np.array_split(self.indices, self.num_buckets)

        self.batches = []
        rng = np.random.default_rng(12345)
        for b in self.buckets:
            b = list(b)
            if shuffle:
                rng.shuffle(b)
            for i in range(0, len(b), batch_size):
                chunk = b[i:i+batch_size]
                if len(chunk) > 0:
                    self.batches.append(chunk)

        if shuffle:
            rng.shuffle(self.batches)

    def __iter__(self):
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)


# ---------------------------
# 6) Inference Engine
# ---------------------------
class UltraInferenceEngine:
    def __init__(self, cfg: UltraConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.preprocessor = OptimizedPreprocessor()

        # postprocessors
        self.safe_pp = SafePostprocessor(
            use_unicode_fractions=False,
            strip_dash_old=False,
        )
        self.safe_pp_stripdash = SafePostprocessor(
            use_unicode_fractions=False,
            strip_dash_old=True,
        )
        self.agg_pp = AggressivePostprocessor()

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_path).to(self.device)
        self.model.eval()

        # optional BetterTransformer
        if cfg.use_better_transformer and cfg.device == "cuda":
            try:
                from optimum.bettertransformer import BetterTransformer
                self.model = BetterTransformer.transform(self.model)
                logger.info("BetterTransformer enabled.")
            except Exception as e:
                logger.warning(f"BetterTransformer failed, continue without it: {e}")

    def _collate_fn(self, batch):
        ids, texts = zip(*batch)
        enc = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.cfg.max_length,
            return_tensors="pt",
        )
        return list(ids), enc

    def _adaptive_beams(self, attention_mask: torch.Tensor) -> int:
        if not self.cfg.use_adaptive_beams:
            return self.cfg.num_beams
        # use median token count in batch for robustness
        lens = attention_mask.sum(dim=1).detach().cpu().numpy()
        med = float(np.median(lens)) if len(lens) else 0.0
        if med < 100:
            return max(4, self.cfg.num_beams // 2)
        return self.cfg.num_beams

    def _postprocess(self, texts: List[str]) -> List[str]:
        mode = self.cfg.postprocess_mode

        if mode == "safe_only":
            out = self.safe_pp.postprocess_batch(texts)
            return out

        if mode == "aggressive_only":
            out = self.safe_pp.postprocess_batch(texts)
            out = self.agg_pp.postprocess_batch(out)
            return out

        # default: safe then conditional aggressive for bad samples
        safe = self.safe_pp.postprocess_batch(texts)
        refined = []
        thr = self.cfg.aggressive_trigger_badness
        for t in safe:
            if badness_score(t) >= thr:
                refined.append(self.agg_pp.postprocess_one(t))
            else:
                refined.append(t)
        return refined

    @staticmethod
    def _final_fix_one(t: str, min_words: int = 3) -> str:
        tt = (t or "").strip()
        if not tt:
            return "The tablet contains fragmentary text."
        words = tt.split()
        if len(words) < min_words:
            return "The tablet contains an incomplete inscription."
        # capitalize first letter if it starts with alpha
        if tt and tt[0].isalpha() and tt[0].islower():
            tt = tt[0].upper() + tt[1:]
        # add ending punctuation if missing
        if not tt.endswith((".", "!", "?")):
            tt = tt + "."
        # clean spaces before punctuation just in case
        tt = re.sub(r"\s+([,.;:!?])", r"\1", tt)
        tt = re.sub(r"\s+", " ", tt).strip()
        return tt

    def run_inference(self, test_df: pd.DataFrame, run_tag: str = "run") -> pd.DataFrame:
        cfg = self.cfg
        ds = AkkadianDataset(test_df, self.preprocessor)

        if cfg.use_bucket_batching:
            sampler = BucketBatchSampler(ds.inputs, batch_size=cfg.batch_size, num_buckets=32, shuffle=False)
            dl = DataLoader(
                ds,
                batch_sampler=sampler,
                num_workers=cfg.num_workers,
                pin_memory=cfg.pin_memory,
                collate_fn=self._collate_fn,
            )
        else:
            dl = DataLoader(
                ds,
                batch_size=cfg.batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
                pin_memory=cfg.pin_memory,
                collate_fn=self._collate_fn,
            )

        results: List[Tuple[str, str]] = []
        t0 = time.time()

        # autocast context
        if cfg.use_mixed_precision and cfg.device == "cuda":
            autocast_ctx = torch.cuda.amp.autocast
        else:
            # no-op context
            class _NullCtx:
                def __enter__(self): return None
                def __exit__(self, exc_type, exc, tb): return False
            autocast_ctx = lambda: _NullCtx()

        for step, (ids, enc) in enumerate(dl):
            input_ids = enc["input_ids"].to(self.device, non_blocking=True)
            attn = enc["attention_mask"].to(self.device, non_blocking=True)

            beams = self._adaptive_beams(attn)

            gen_kwargs = dict(
                num_beams=beams,
                max_new_tokens=cfg.max_new_tokens,
                length_penalty=cfg.length_penalty,
                early_stopping=cfg.early_stopping,
                no_repeat_ngram_size=cfg.no_repeat_ngram_size,
            )
            if cfg.repetition_penalty is not None:
                gen_kwargs["repetition_penalty"] = cfg.repetition_penalty

            with torch.inference_mode():
                with autocast_ctx():
                    out_ids = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attn,
                        **gen_kwargs
                    )

            decoded = self.tokenizer.batch_decode(out_ids, skip_special_tokens=True)
            processed = self._postprocess(decoded)
            processed = [self._final_fix_one(x, cfg.min_words_fallback) for x in processed]

            results.extend(list(zip(ids, processed)))

            # periodic checkpoint
            if cfg.checkpoint_freq and (len(results) % cfg.checkpoint_freq == 0):
                ck = pd.DataFrame(results, columns=["id", "translation"])
                ck_path = os.path.join(cfg.output_dir, f"checkpoint_{run_tag}_{len(results)}.csv")
                ck.to_csv(ck_path, index=False)
                logger.info(f"Saved checkpoint: {ck_path}")

            if cfg.device == "cuda" and cfg.empty_cache_every and (step + 1) % cfg.empty_cache_every == 0:
                torch.cuda.empty_cache()

            if (step + 1) % 50 == 0:
                elapsed = time.time() - t0
                logger.info(f"[{run_tag}] step={step+1}/{len(dl)} | done={len(results)} | elapsed={elapsed:.1f}s")

        df_out = pd.DataFrame(results, columns=["id", "translation"])

        # quick validation
        self._validate(df_out, run_tag=run_tag)

        return df_out

    def _validate(self, df: pd.DataFrame, run_tag: str):
        if df.empty:
            logger.warning(f"[{run_tag}] Empty output dataframe.")
            return
        lens = df["translation"].fillna("").map(lambda x: len(str(x).split()))
        empties = (df["translation"].fillna("").str.strip() == "").mean() * 100
        short = (lens < 5).sum()
        logger.info(f"[{run_tag}] rows={len(df)} | empty%={empties:.2f} | len(mean/med/min/max)={lens.mean():.1f}/{lens.median():.1f}/{lens.min()}/{lens.max()} | short(<5)={short}")

        k = min(self.cfg.validate_samples, len(df))
        if k > 0:
            sample = df.sample(k, random_state=123)
            for _, r in sample.iterrows():
                logger.info(f"[{run_tag}] sample id={r['id']} | {str(r['translation'])[:160]}")

# ---------------------------
# 7) Helpers to build configs (two-pass)
# ---------------------------
def make_cfg(base: UltraConfig,
             preset: str,
             batch_size: Optional[int] = None,
             num_beams: Optional[int] = None,
             length_penalty: Optional[float] = None,
             repetition_penalty: Optional[float] = None,
             no_repeat_ngram_size: Optional[int] = None,
             strip_dash_old: Optional[bool] = None) -> UltraConfig:
    cfg = UltraConfig(**asdict(base))

    # presets
    if preset == "baseline":
        cfg.length_penalty = 1.30
        cfg.repetition_penalty = None
    elif preset == "len115":
        cfg.length_penalty = 1.15
        cfg.repetition_penalty = None
    elif preset == "rep_pen":
        cfg.length_penalty = 1.30
        cfg.repetition_penalty = 1.08
    elif preset == "len115_rep":
        cfg.length_penalty = 1.15
        cfg.repetition_penalty = 1.08
    else:
        raise ValueError(f"Unknown preset: {preset}")

    # overrides
    if batch_size is not None: cfg.batch_size = int(batch_size)
    if num_beams is not None: cfg.num_beams = int(num_beams)
    if length_penalty is not None: cfg.length_penalty = float(length_penalty)
    if repetition_penalty is not None: cfg.repetition_penalty = float(repetition_penalty)
    if no_repeat_ngram_size is not None: cfg.no_repeat_ngram_size = int(no_repeat_ngram_size)

    # strip_dash_old is implemented inside SafePostprocessor; we control it via engine if desired
    # here we store it in cfg as metadata only
    cfg._strip_dash_old = bool(strip_dash_old) if strip_dash_old is not None else False  # type: ignore

    return cfg

def fuse_submissions(df_a: pd.DataFrame, df_b: pd.DataFrame,
                     prefer: str = "a", tie_badness_thresh: float = 0.5,
                     w_a: float = 0.60, w_b: float = 0.40) -> pd.DataFrame:
    a = df_a.set_index("id")["translation"]
    b = df_b.set_index("id")["translation"]
    ids = a.index

    out = []
    for _id in ids:
        ta = a.loc[_id]
        tb = b.loc[_id]
        fused = fuse_texts(ta, tb, prefer=prefer, tie_badness_thresh=tie_badness_thresh, w_a=w_a, w_b=w_b)
        out.append((_id, fused))
    return pd.DataFrame(out, columns=["id", "translation"])


# ---------------------------
# 8) Main: EXP4 – Pass B ONLY (no fusion)
# ---------------------------
base_cfg = UltraConfig()

# Ensure output dir exists
os.makedirs(base_cfg.output_dir, exist_ok=True)

# Load test
test_df = pd.read_csv(base_cfg.test_data_path)
assert "id" in test_df.columns and "transliteration" in test_df.columns, "test.csv must have columns: id, transliteration"

# Save config snapshot
with open(os.path.join(base_cfg.output_dir, "run_config_base.json"), "w", encoding="utf-8") as f:
    json.dump(asdict(base_cfg), f, ensure_ascii=False, indent=2)

# Pass B only
cfg_b = make_cfg(base_cfg, preset="rep_pen")

logger.info(f"PassB ONLY: len_pen={cfg_b.length_penalty}, rep_pen={cfg_b.repetition_penalty}, beams={cfg_b.num_beams}, bs={cfg_b.batch_size}")

# Run inference B
engine_b = UltraInferenceEngine(cfg_b)
sub_b = engine_b.run_inference(test_df, run_tag="B_rep_pen")

# Final sanity (no empties)
sub_b["translation"] = sub_b["translation"].fillna("").map(lambda x: UltraInferenceEngine._final_fix_one(str(x), min_words=base_cfg.min_words_fallback))

out_path = os.path.join(base_cfg.output_dir, "submission.csv")
sub_b.to_csv(out_path, index=False)
logger.info(f"Saved FINAL submission (Pass B only): {out_path}")

# Quick summary
lens = sub_b["translation"].map(lambda x: len(str(x).split()))
logger.info(f"FINAL | rows={len(sub_b)} | len(mean/med/min/max)={lens.mean():.1f}/{lens.median():.1f}/{lens.min()}/{lens.max()}")
print(sub_b.head())
print(sub_b.tail())

