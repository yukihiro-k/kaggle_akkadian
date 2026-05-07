#!/usr/bin/env python3
"""
ByT5 Continued Fine-Tuning for Akkadian → English Translation
==============================================================
Fine-tunes an EXISTING Akkadian ByT5 checkpoint on competition train.csv
to create a 3rd ensemble member for 355by_v5.py.

Key design choices for continued fine-tuning:
  - Low learning rate (5e-5) to preserve base model knowledge
  - Fewer epochs (10) with early stopping (patience=3)
  - chrF evaluation metric (matches competition scoring)
  - Sentence-level splitting for more training pairs
  - Preprocessing aligned with inference pipeline

Kaggle usage:
  1. Add competition dataset as input
  2. Add base model (assiaben/final-byt5) as input
  3. Enable GPU accelerator (P100 or T4)
  4. Run this script
  5. Save /kaggle/working/byt5-akkadian-v2/final_model as a dataset
  6. Add saved dataset to inference notebook inputs
"""

import os
import re
import gc
import json
import math
import logging
import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    set_seed,
)

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import sacrebleu
    _HAS_SACREBLEU = True
except ImportError:
    _HAS_SACREBLEU = False

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # Force single GPU — avoids DataParallel OOM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("byt5-train-v2")


# ===========================================================================
# 1. Configuration — optimised for CONTINUED fine-tuning
# ===========================================================================

@dataclass
class TrainConfig:
    # --- Paths ---
    train_csv:    str = "/kaggle/input/competitions/deep-past-initiative-machine-translation/train.csv"
    sentence_csv: str = "/kaggle/input/competitions/deep-past-initiative-machine-translation/Sentences_Oare_FirstWord_LinNum.csv"

    # Base model — mattiaangeli's MBR-v2 checkpoint
    model_name: str = "/kaggle/input/models/mattiaangeli/byt5-akkadian-mbr-v2/pytorch/default/1"

    # Output
    output_dir:  str = "/kaggle/working/byt5-akkadian-v2"
    logging_dir: str = "/kaggle/working/logs"

    # --- Tokenization ---
    max_source_length: int = 384
    max_target_length: int = 384
    input_prefix: str = "translate Akkadian to English: "

    # --- Training (tuned for continued fine-tuning) ---
    seed: int = 42
    num_epochs: int = 10                    # fewer — base is already trained
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size:  int = 1
    gradient_accumulation_steps: int = 16   # effective batch = 1 * 16 = 16
    learning_rate: float = 5e-5             # very conservative for stability
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    lr_scheduler_type: str = "cosine"
    fp16: bool = False                      # Disable half-precision to stop NaNs
    bf16: bool = False
    gradient_checkpointing: bool = True     # essential for T4 16 GB

    # --- Evaluation ---
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "loss"      # loss is always available
    greater_is_better: bool = False           # lower loss = better
    early_stopping_patience: int = 3        # faster stopping — converges quickly

    # --- Generation (during eval) ---
    predict_with_generate: bool = True
    generation_max_new_tokens: int = 256
    generation_num_beams: int = 4

    # --- Data split ---
    val_fraction: float = 0.10
    use_sentence_splitting: bool = True

    # --- Device ---
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        # Respect the fp16/bf16 settings from fields — don't overwrite them here.
        if self.device == "cpu":
            self.fp16 = False
            self.bf16 = False
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.logging_dir, exist_ok=True)


# ===========================================================================
# 2. Preprocessor — aligned with inference pipeline
# ===========================================================================

_V2 = re.compile(r"([aAeEiIuU])(?:2|₂)")
_V3 = re.compile(r"([aAeEiIuU])(?:3|₃)")
_ACUTE = str.maketrans({"a":"á","e":"é","i":"í","u":"ú","A":"Á","E":"É","I":"Í","U":"Ú"})
_GRAVE = str.maketrans({"a":"à","e":"è","i":"ì","u":"ù","A":"À","E":"È","I":"Ì","U":"Ù"})

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

_CHAR_TRANS = str.maketrans({
    "ḫ":"h", "Ḫ":"H", "ʾ":"",
    "₀":"0","₁":"1","₂":"2","₃":"3","₄":"4",
    "₅":"5","₆":"6","₇":"7","₈":"8","₉":"9",
    "—":"-","–":"-",
})

_UNICODE_UPPER = r"A-ZŠṬṢḪ\u00C0-\u00D6\u00D8-\u00DE\u0160\u1E00-\u1EFF"
_UNICODE_LOWER = r"a-zšṭṣḫ\u00E0-\u00F6\u00F8-\u00FF\u0161\u1E01-\u1EFF"
_DET_UPPER_RE = re.compile(r"\(([" + _UNICODE_UPPER + r"0-9]{1,6})\)")
_DET_LOWER_RE = re.compile(r"\(([" + _UNICODE_LOWER + r"]{1,4})\)")


def _ascii_to_diacritics(s: str) -> str:
    s = s.replace("sz","š").replace("SZ","Š")
    s = s.replace("s,","ṣ").replace("S,","Ṣ")
    s = s.replace("t,","ṭ").replace("T,","Ṭ")
    s = _V2.sub(lambda m: m.group(1).translate(_ACUTE), s)
    s = _V3.sub(lambda m: m.group(1).translate(_GRAVE), s)
    return s


class Preprocessor:
    """Same normalisation as inference pipeline."""

    def clean(self, text: str) -> str:
        if not text or pd.isna(text):
            return ""
        t = str(text)
        t = _ascii_to_diacritics(t)
        t = _DET_UPPER_RE.sub(r"\1", t)
        t = _DET_LOWER_RE.sub(r"{\1}", t)
        t = _GAP_UNIFIED_RE.sub("<gap>", t)
        t = t.translate(_CHAR_TRANS)
        t = t.replace("ₓ", "")
        t = re.sub(r"\s+", " ", t).strip()
        return t


# ===========================================================================
# 3. Sentence-level splitting
# ===========================================================================

def split_documents_to_sentences(
    train_df: pd.DataFrame,
    sentence_df: pd.DataFrame,
) -> pd.DataFrame:
    logger.info(f"Attempting sentence-level splitting...")
    logger.info(f"  train documents: {len(train_df)}")
    logger.info(f"  sentence hints: {len(sentence_df)}")
    logger.info(f"  sentence CSV columns: {list(sentence_df.columns)}")

    id_col = None
    for candidate in ["oare_id", "text_id", "id", "ID"]:
        if candidate in sentence_df.columns:
            id_col = candidate
            break

    if id_col is None:
        logger.warning("Could not find ID column in sentence CSV. Falling back.")
        return _fallback_sentence_split(train_df)

    line_col = None
    for candidate in ["line_num", "line", "LinNum", "line_number", "line_start"]:
        if candidate in sentence_df.columns:
            line_col = candidate
            break

    word_col = None
    for candidate in ["first_word", "FirstWord", "word", "first"]:
        if candidate in sentence_df.columns:
            word_col = candidate
            break

    logger.info(f"  Using columns: id={id_col}, line={line_col}, word={word_col}")

    sent_groups = sentence_df.groupby(id_col)
    results = []
    split_count = 0
    fallback_count = 0

    for _, row in train_df.iterrows():
        oare_id = row.get("oare_id", "")
        transliteration = str(row.get("transliteration", ""))
        translation = str(row.get("translation", ""))

        if not transliteration or not translation:
            continue

        if oare_id in sent_groups.groups and word_col is not None:
            group = sent_groups.get_group(oare_id)
            sentences = _split_by_boundaries(
                transliteration, translation, group, word_col, line_col
            )
            if sentences and len(sentences) > 1:
                for src, tgt in sentences:
                    if src.strip() and tgt.strip():
                        results.append({
                            "oare_id": oare_id,
                            "transliteration": src.strip(),
                            "translation": tgt.strip(),
                        })
                split_count += 1
                continue

        sentences = _heuristic_split(transliteration, translation)
        for src, tgt in sentences:
            if src.strip() and tgt.strip():
                results.append({
                    "oare_id": oare_id,
                    "transliteration": src.strip(),
                    "translation": tgt.strip(),
                })
        fallback_count += 1

    result_df = pd.DataFrame(results)
    logger.info(
        f"Sentence splitting done: {len(result_df)} sentence pairs "
        f"(split={split_count}, fallback={fallback_count})"
    )
    return result_df


def _split_by_boundaries(
    transliteration: str,
    translation: str,
    group: pd.DataFrame,
    word_col: str,
    line_col: Optional[str],
) -> List[Tuple[str, str]]:
    first_words = group[word_col].dropna().tolist()
    if len(first_words) < 2:
        return []

    trans_parts = []
    remaining = transliteration
    for fw in first_words[1:]:
        fw_str = str(fw).strip()
        if not fw_str:
            continue
        idx = remaining.find(fw_str)
        if idx > 0:
            trans_parts.append(remaining[:idx].strip())
            remaining = remaining[idx:]
    trans_parts.append(remaining.strip())

    if len(trans_parts) < 2:
        return []

    transl_sentences = re.split(r'(?<=[.!?])\s+', translation)
    transl_sentences = [s.strip() for s in transl_sentences if s.strip()]

    if len(trans_parts) == len(transl_sentences):
        return list(zip(trans_parts, transl_sentences))

    if len(transl_sentences) > 1 and len(trans_parts) > 1:
        return _proportional_align(trans_parts, transl_sentences)

    return []


def _proportional_align(
    sources: List[str], targets: List[str]
) -> List[Tuple[str, str]]:
    if len(sources) == len(targets):
        return list(zip(sources, targets))
    if len(targets) > len(sources):
        merged = _merge_to_count(targets, len(sources))
        return list(zip(sources, merged))
    else:
        merged = _merge_to_count(sources, len(targets))
        return list(zip(merged, targets))


def _merge_to_count(items: List[str], target_count: int) -> List[str]:
    if target_count >= len(items):
        return items
    total_len = sum(len(s) for s in items)
    chunk_len = total_len / target_count
    merged = []
    current = []
    current_len = 0
    for item in items:
        current.append(item)
        current_len += len(item)
        if current_len >= chunk_len and len(merged) < target_count - 1:
            merged.append(" ".join(current))
            current = []
            current_len = 0
    if current:
        merged.append(" ".join(current))
    return merged


def _heuristic_split(
    transliteration: str, translation: str
) -> List[Tuple[str, str]]:
    trans_lines = re.split(r"\n+", transliteration)
    trans_lines = [l.strip() for l in trans_lines if l.strip()]
    transl_sents = re.split(r'(?<=[.!?])\s+', translation)
    transl_sents = [s.strip() for s in transl_sents if s.strip()]
    if len(trans_lines) > 1 and len(transl_sents) > 1:
        aligned = _proportional_align(trans_lines, transl_sents)
        if aligned:
            return aligned
    return [(transliteration, translation)]


def _fallback_sentence_split(train_df: pd.DataFrame) -> pd.DataFrame:
    results = []
    for _, row in train_df.iterrows():
        transliteration = str(row.get("transliteration", ""))
        translation = str(row.get("translation", ""))
        if transliteration.strip() and translation.strip():
            sentences = _heuristic_split(transliteration, translation)
            for src, tgt in sentences:
                if src.strip() and tgt.strip():
                    results.append({
                        "oare_id": row.get("oare_id", ""),
                        "transliteration": src.strip(),
                        "translation": tgt.strip(),
                    })
    return pd.DataFrame(results)


# ===========================================================================
# 4. Dataset
# ===========================================================================

class AkkadianTranslationDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer,
        preprocessor: Preprocessor,
        max_source_length: int = 512,
        max_target_length: int = 512,
        input_prefix: str = "translate Akkadian to English: ",
    ):
        self.tokenizer = tokenizer
        self.preprocessor = preprocessor
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.input_prefix = input_prefix

        raw_inputs = data["transliteration"].fillna("").tolist()
        self.inputs = [
            f"{input_prefix}{preprocessor.clean(t)}" for t in raw_inputs
        ]
        self.targets = data["translation"].fillna("").tolist()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        source = self.inputs[idx]
        target = self.targets[idx]

        source_encoding = self.tokenizer(
            source,
            max_length=self.max_source_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        target_encoding = self.tokenizer(
            target,
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = source_encoding["input_ids"].squeeze()
        attention_mask = source_encoding["attention_mask"].squeeze()
        labels = target_encoding["input_ids"].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# ===========================================================================
# 5. chrF + BLEU metrics
# ===========================================================================

def _simple_chrf(predictions, references, n=6, beta=2.0):
    """Simple chrF implementation (no sacrebleu dependency)."""
    def _char_ngrams(text, n):
        ngrams = {}
        for i in range(len(text) - n + 1):
            ng = text[i:i+n]
            ngrams[ng] = ngrams.get(ng, 0) + 1
        return ngrams

    total_p, total_r = 0.0, 0.0
    count = 0
    for pred, ref in zip(predictions, references):
        for order in range(1, n + 1):
            p_ngrams = _char_ngrams(pred, order)
            r_ngrams = _char_ngrams(ref, order)
            common = sum(min(p_ngrams.get(ng, 0), r_ngrams.get(ng, 0)) for ng in r_ngrams)
            p_total = max(sum(p_ngrams.values()), 1)
            r_total = max(sum(r_ngrams.values()), 1)
            total_p += common / p_total
            total_r += common / r_total
            count += 1
    avg_p = total_p / max(count, 1)
    avg_r = total_r / max(count, 1)
    if avg_p + avg_r == 0:
        return 0.0
    return 100.0 * (1 + beta**2) * avg_p * avg_r / (beta**2 * avg_p + avg_r)


def build_compute_metrics(tokenizer):
    """Build compute_metrics. Uses sacrebleu if available, otherwise simple chrF."""
    try:
        import sacrebleu
        has_sacrebleu = True
        logger.info("Using sacrebleu for chrF + BLEU.")
    except ImportError:
        has_sacrebleu = False
        logger.warning("sacrebleu not available. Install: pip install sacrebleu")

    try:
        import evaluate
        bleu_metric = evaluate.load("sacrebleu")
        logger.info("Using `evaluate` for BLEU fallback.")
    except ImportError:
        bleu_metric = None

    def compute_metrics(eval_preds):
        preds, labels = eval_preds

        if isinstance(preds, tuple):
            preds = preds[0]

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        # ByT5 maps token IDs directly to chr() — clip to valid range
        vocab_size = tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else 384
        preds  = np.clip(preds, 0, vocab_size - 1)
        labels = np.clip(labels, 0, vocab_size - 1)

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = [p.strip() for p in decoded_preds]
        decoded_labels = [l.strip() for l in decoded_labels]

        metrics = {}

        # chrF (primary metric — matches competition)
        if has_sacrebleu:
            chrf = sacrebleu.corpus_chrf(decoded_preds, [decoded_labels], word_order=2)
            metrics["chrf"] = round(chrf.score, 4)
        else:
            metrics["chrf"] = round(_simple_chrf(decoded_preds, decoded_labels), 4)

        # BLEU (secondary)
        if has_sacrebleu:
            bleu = sacrebleu.corpus_bleu(decoded_preds, [decoded_labels])
            metrics["bleu"] = round(bleu.score, 4)
        elif bleu_metric is not None:
            result = bleu_metric.compute(
                predictions=decoded_preds,
                references=[[l] for l in decoded_labels],
            )
            metrics["bleu"] = round(result["score"], 4)

        # Average prediction length
        pred_lens = [len(p.split()) for p in decoded_preds]
        metrics["avg_pred_len"] = round(np.mean(pred_lens), 1)

        # Log samples
        n_show = min(3, len(decoded_preds))
        for i in range(n_show):
            logger.info(f"  [eval sample {i}]")
            logger.info(f"    pred: {decoded_preds[i][:150]}")
            logger.info(f"    gold: {decoded_labels[i][:150]}")

        return metrics

    return compute_metrics


# ===========================================================================
# 6. Main
# ===========================================================================

def main():
    cfg = TrainConfig()
    set_seed(cfg.seed)

    logger.info("=" * 60)
    logger.info("ByT5 Continued Fine-Tuning — 3rd Ensemble Member")
    logger.info("=" * 60)
    logger.info(f"Base model : {cfg.model_name}")
    logger.info(f"Device     : {cfg.device}")
    logger.info(f"FP16       : {cfg.fp16}")
    logger.info(f"LR         : {cfg.learning_rate}")
    logger.info(f"Epochs     : {cfg.num_epochs}")
    logger.info(f"Eval metric: {cfg.metric_for_best_model}")
    if torch.cuda.is_available():
        logger.info(f"GPU        : {torch.cuda.get_device_name(0)}")
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"GPU Memory : {mem_gb:.1f} GB")

    # =====================
    # 0. Cleanup old runs to save disk space
    # =====================
    if os.path.exists(cfg.output_dir):
        logger.info(f"Cleaning up old output directory: {cfg.output_dir}")
        import shutil
        for item in os.listdir(cfg.output_dir):
            item_path = os.path.join(cfg.output_dir, item)
            try:
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                else:
                    os.remove(item_path)
            except Exception as e:
                logger.warning(f"Failed to delete {item_path}: {e}")

    # =====================
    # Load data
    # =====================
    logger.info(f"Loading training data from: {cfg.train_csv}")
    train_df = pd.read_csv(cfg.train_csv)
    logger.info(f"  Loaded {len(train_df)} documents")

    assert "transliteration" in train_df.columns
    assert "translation" in train_df.columns

    train_df = train_df.dropna(subset=["transliteration", "translation"])
    train_df = train_df[
        (train_df["transliteration"].str.strip() != "")
        & (train_df["translation"].str.strip() != "")
    ].reset_index(drop=True)
    logger.info(f"  After filtering: {len(train_df)} documents")

    # =====================
    # Sentence-level splitting
    # =====================
    if cfg.use_sentence_splitting and os.path.exists(cfg.sentence_csv):
        logger.info(f"Loading sentence hints from: {cfg.sentence_csv}")
        sentence_df = pd.read_csv(cfg.sentence_csv)
        data_df = split_documents_to_sentences(train_df, sentence_df)
    else:
        logger.info("Sentence splitting disabled or CSV not found.")
        data_df = _fallback_sentence_split(train_df)

    logger.info(f"Total training pairs: {len(data_df)}")

    # =====================
    # Train/val split (by oare_id to avoid leakage)
    # =====================
    if "oare_id" in data_df.columns:
        unique_ids = data_df["oare_id"].unique()
        np.random.seed(cfg.seed)
        np.random.shuffle(unique_ids)
        val_size = max(1, int(len(unique_ids) * cfg.val_fraction))
        val_ids = set(unique_ids[:val_size])

        val_df = data_df[data_df["oare_id"].isin(val_ids)].reset_index(drop=True)
        trn_df = data_df[~data_df["oare_id"].isin(val_ids)].reset_index(drop=True)
    else:
        val_size = max(1, int(len(data_df) * cfg.val_fraction))
        indices = np.random.RandomState(cfg.seed).permutation(len(data_df))
        val_df = data_df.iloc[indices[:val_size]].reset_index(drop=True)
        trn_df = data_df.iloc[indices[val_size:]].reset_index(drop=True)

    logger.info(f"Train: {len(trn_df)} pairs | Val: {len(val_df)} pairs")

    # =====================
    # Load model & tokenizer from base checkpoint
    # =====================
    logger.info(f"Loading base model: {cfg.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_name)

    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters    : {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # =====================
    # Create datasets
    # =====================
    preprocessor = Preprocessor()

    train_dataset = AkkadianTranslationDataset(
        data=trn_df,
        tokenizer=tokenizer,
        preprocessor=preprocessor,
        max_source_length=cfg.max_source_length,
        max_target_length=cfg.max_target_length,
        input_prefix=cfg.input_prefix,
    )

    val_dataset = AkkadianTranslationDataset(
        data=val_df,
        tokenizer=tokenizer,
        preprocessor=preprocessor,
        max_source_length=cfg.max_source_length,
        max_target_length=cfg.max_target_length,
        input_prefix=cfg.input_prefix,
    )

    logger.info(f"Train dataset: {len(train_dataset)} samples")
    logger.info(f"Val dataset  : {len(val_dataset)} samples")

    # =====================
    # Data collator
    # =====================
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding="longest",
        max_length=cfg.max_source_length,
        label_pad_token_id=-100,
    )

    # =====================
    # Training arguments
    # =====================
    training_args = Seq2SeqTrainingArguments(
        output_dir=cfg.output_dir,
        logging_dir=cfg.logging_dir,

        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler_type,
        fp16=cfg.fp16,
        bf16=cfg.bf16,
        fp16_full_eval=False,  # safer
        bf16_full_eval=cfg.bf16,

        eval_strategy=cfg.eval_strategy,
        save_strategy=cfg.save_strategy,
        load_best_model_at_end=cfg.load_best_model_at_end,
        metric_for_best_model=cfg.metric_for_best_model,
        greater_is_better=cfg.greater_is_better,
        save_total_limit=2,                 # reduced from 3 to save disk space (FP32 is large)

        predict_with_generate=cfg.predict_with_generate,
        generation_max_length=cfg.generation_max_new_tokens,
        generation_num_beams=cfg.generation_num_beams,

        logging_steps=10,
        report_to="none",

        seed=cfg.seed,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
    )

    # =====================
    # Metrics
    # =====================
    compute_metrics = build_compute_metrics(tokenizer)

    # =====================
    # Train
    # =====================
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=cfg.early_stopping_patience
            ),
        ],
    )

    logger.info("=" * 60)
    logger.info("Starting continued fine-tuning...")
    logger.info(f"  Epochs           : {cfg.num_epochs}")
    logger.info(f"  Batch (per GPU)  : {cfg.per_device_train_batch_size}")
    logger.info(f"  Grad accum       : {cfg.gradient_accumulation_steps}")
    logger.info(f"  Effective batch  : {cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps}")
    logger.info(f"  Learning rate    : {cfg.learning_rate}")
    logger.info(f"  Early stopping   : patience={cfg.early_stopping_patience}")
    logger.info("=" * 60)

    train_result = trainer.train()

    # =====================
    # Save final model
    # =====================
    final_model_dir = os.path.join(cfg.output_dir, "final_model")
    trainer.save_model(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    logger.info(f"Final model saved to: {final_model_dir}")

    # Save training metrics
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    # =====================
    # Final evaluation
    # =====================
    logger.info("Running final evaluation...")
    eval_metrics = trainer.evaluate()
    eval_metrics["eval_samples"] = len(val_dataset)
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    # =====================
    # 🎉 Completion banner (stdout)
    # =====================
    chrf_val = eval_metrics.get("eval_chrf", "N/A")
    bleu_val = eval_metrics.get("eval_bleu", "N/A")

    print()
    print("=" * 60)
    print("=" * 60)
    print("  ✅  TRAINING COMPLETE!  ✅")
    print("=" * 60)
    print()
    print(f"  📁 Model saved to : {final_model_dir}")
    print(f"  📊 Final chrF     : {chrf_val}")
    print(f"  📊 Final BLEU     : {bleu_val}")
    print()
    print("  📋 NEXT STEPS:")
    print("  1. この出力を Kaggle Dataset として保存")
    print("     (New Dataset → byt5-akkadian-v2)")
    print("  2. 推論ノートブックの model_paths に追加:")
    print(f'     "{final_model_dir}"')
    print()
    print("=" * 60)
    print("=" * 60)
    print()

    del model, trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return eval_metrics


if __name__ == "__main__":
    main()
