# ============================================================
# ByT5 Fine-Tuning for Akkadian → English Translation
# - Kaggle Notebook optimized (T4/P100 16GB)
# - Sentence-level splitting from Sentences_Oare_FirstWord_LinNum.csv
# - HuggingFace Seq2SeqTrainer with BLEU evaluation
# - Mixed precision (FP16), gradient accumulation, early stopping
# ============================================================

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

# ---------------------------
# 0) Environment & logging
# ---------------------------
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("byt5-train")


# ---------------------------
# 1) Configuration
# ---------------------------
@dataclass
class TrainConfig:
    """Training configuration for ByT5 fine-tuning on Kaggle."""

    # ===== Paths =====
    # Kaggle competition data
    train_csv: str = "/kaggle/input/deep-past-initiative-machine-translation/train.csv"
    sentence_csv: str = "/kaggle/input/deep-past-initiative-machine-translation/Sentences_Oare_FirstWord_LinNum.csv"

    # Model - change this to your existing model path for continued fine-tuning
    # e.g., "/kaggle/input/final-byt5/byt5-akkadian-optimized-34x"
    model_name: str = "google/byt5-small"

    # Output
    output_dir: str = "/kaggle/working/byt5-akkadian-finetuned"
    logging_dir: str = "/kaggle/working/logs"

    # ===== Tokenization =====
    max_source_length: int = 512   # max input tokens (transliteration)
    max_target_length: int = 512   # max output tokens (translation)
    input_prefix: str = "translate Akkadian to English: "

    # ===== Training =====
    seed: int = 42
    num_epochs: int = 20
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4   # effective batch size = 4 * 4 = 16
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.10
    lr_scheduler_type: str = "cosine"
    fp16: bool = True
    gradient_checkpointing: bool = True    # saves GPU memory at cost of speed

    # ===== Evaluation =====
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "bleu"
    greater_is_better: bool = True
    early_stopping_patience: int = 5

    # ===== Generation (during eval) =====
    predict_with_generate: bool = True
    generation_max_new_tokens: int = 256   # shorter for eval speed
    generation_num_beams: int = 4          # fewer beams for eval speed

    # ===== Data split =====
    val_fraction: float = 0.10
    use_sentence_splitting: bool = True    # use Sentences CSV for sentence-level data

    # ===== Device =====
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        if self.device == "cpu":
            self.fp16 = False
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.logging_dir, exist_ok=True)


# ---------------------------
# 2) Preprocessor (same as inference)
# ---------------------------
class Preprocessor:
    """Normalize Akkadian transliterations for model input."""

    def __init__(self):
        self.re_big_gap = re.compile(r"(\.\.\.+|…+)")
        self.re_small_gap = re.compile(
            r"(?:(?<=\s)|^)(x{1,3})(?=(\s|$))", flags=re.IGNORECASE
        )

    def clean(self, text: str) -> str:
        if not text or pd.isna(text):
            return ""
        t = str(text)
        t = self.re_big_gap.sub(" <big_gap> ", t)
        t = self.re_small_gap.sub(" <gap> ", t)
        t = re.sub(r"\s+", " ", t).strip()
        return t


# ---------------------------
# 3) Sentence-Level Splitting
# ---------------------------
def split_documents_to_sentences(
    train_df: pd.DataFrame,
    sentence_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Split document-level train.csv into sentence-level pairs
    using the Sentences_Oare_FirstWord_LinNum.csv helper file.

    The sentence CSV indicates where each sentence begins within the
    transliteration. We use this info to split both the transliteration
    and the translation into aligned sentence pairs.

    If alignment fails for a document, we fall back to keeping the
    full document as a single training example.
    """
    logger.info(f"Attempting sentence-level splitting...")
    logger.info(f"  train documents: {len(train_df)}")
    logger.info(f"  sentence hints: {len(sentence_df)}")

    # Examine the sentence CSV columns to determine the format
    logger.info(f"  sentence CSV columns: {list(sentence_df.columns)}")

    # Try to identify the oare_id / text_id column in sentences CSV
    id_col = None
    for candidate in ["oare_id", "text_id", "id", "ID"]:
        if candidate in sentence_df.columns:
            id_col = candidate
            break

    if id_col is None:
        logger.warning("Could not find ID column in sentence CSV. Falling back to document-level.")
        return _fallback_sentence_split(train_df)

    # Identify line number column
    line_col = None
    for candidate in ["line_num", "line", "LinNum", "line_number", "line_start"]:
        if candidate in sentence_df.columns:
            line_col = candidate
            break

    # Identify first word column
    word_col = None
    for candidate in ["first_word", "FirstWord", "word", "first"]:
        if candidate in sentence_df.columns:
            word_col = candidate
            break

    logger.info(f"  Using columns: id={id_col}, line={line_col}, word={word_col}")

    # Group sentence boundaries by document
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

        # Try to split this document using sentence boundaries
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

        # Fallback: use simple heuristic splitting
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
    """
    Attempt to split a document into sentence-level pairs using
    the first-word boundary information.
    """
    first_words = group[word_col].dropna().tolist()
    if len(first_words) < 2:
        return []

    # Split transliteration at boundary words
    trans_parts = []
    remaining = transliteration
    for fw in first_words[1:]:  # skip first (it's the start)
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

    # Split translation by sentence boundaries (period/end punctuation)
    # This is heuristic - split on ". " or "." at likely sentence ends
    transl_sentences = re.split(r'(?<=[.!?])\s+', translation)
    transl_sentences = [s.strip() for s in transl_sentences if s.strip()]

    # If counts don't match, try to align by distributing evenly
    if len(trans_parts) == len(transl_sentences):
        return list(zip(trans_parts, transl_sentences))

    # Attempt proportional alignment by character length
    if len(transl_sentences) > 1 and len(trans_parts) > 1:
        return _proportional_align(trans_parts, transl_sentences)

    return []


def _proportional_align(
    sources: List[str], targets: List[str]
) -> List[Tuple[str, str]]:
    """
    Align source and target sequences proportionally by length.
    If counts differ, merge shorter list's segments to match the longer.
    """
    if len(sources) == len(targets):
        return list(zip(sources, targets))

    # Make targets match sources count by merging
    if len(targets) > len(sources):
        # Merge targets to match source count
        merged = _merge_to_count(targets, len(sources))
        return list(zip(sources, merged))
    else:
        # Merge sources to match target count
        merged = _merge_to_count(sources, len(targets))
        return list(zip(merged, targets))


def _merge_to_count(items: List[str], target_count: int) -> List[str]:
    """Merge a list of strings into target_count groups, proportional by length."""
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
    """
    Simple heuristic: split by line breaks or numbered lines in the
    transliteration, and by sentences in the translation.
    Falls back to returning the whole document as one pair.
    """
    # Try splitting transliteration on line numbers (e.g., "1. ", "2. ")
    trans_lines = re.split(r"\n+", transliteration)
    trans_lines = [l.strip() for l in trans_lines if l.strip()]

    transl_sents = re.split(r'(?<=[.!?])\s+', translation)
    transl_sents = [s.strip() for s in transl_sents if s.strip()]

    if len(trans_lines) > 1 and len(transl_sents) > 1:
        aligned = _proportional_align(trans_lines, transl_sents)
        if aligned:
            return aligned

    # Final fallback: single pair
    return [(transliteration, translation)]


def _fallback_sentence_split(train_df: pd.DataFrame) -> pd.DataFrame:
    """When sentence CSV is unavailable, use heuristic splitting."""
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


# ---------------------------
# 4) Dataset
# ---------------------------
class AkkadianTranslationDataset(Dataset):
    """
    PyTorch Dataset for Akkadian → English translation.
    Tokenizes on-the-fly for memory efficiency.
    """

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

        # Preprocess all transliterations
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

        # Tokenize source
        source_encoding = self.tokenizer(
            source,
            max_length=self.max_source_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Tokenize target
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

        # Replace padding token id with -100 so it's ignored in loss
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# ---------------------------
# 5) BLEU Metric
# ---------------------------
def build_compute_metrics(tokenizer):
    """
    Build a compute_metrics function for Seq2SeqTrainer.
    Uses sacrebleu for BLEU score computation.
    """
    try:
        import evaluate
        bleu_metric = evaluate.load("sacrebleu")
        logger.info("Using `evaluate` library for BLEU.")
    except ImportError:
        bleu_metric = None
        logger.warning(
            "The `evaluate` library is not installed. "
            "Install via: pip install evaluate sacrebleu. "
            "BLEU will be computed manually."
        )

    def compute_metrics(eval_preds):
        preds, labels = eval_preds

        # Decode predictions
        if isinstance(preds, tuple):
            preds = preds[0]

        # Replace -100 in labels (can't decode them)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Strip whitespace
        decoded_preds = [p.strip() for p in decoded_preds]
        decoded_labels = [l.strip() for l in decoded_labels]

        if bleu_metric is not None:
            # sacrebleu expects references as list of lists
            result = bleu_metric.compute(
                predictions=decoded_preds,
                references=[[l] for l in decoded_labels],
            )
            bleu = result["score"]
        else:
            # Simple manual BLEU approximation (unigram precision)
            bleu = _simple_bleu(decoded_preds, decoded_labels)

        # Also compute average prediction length
        pred_lens = [len(p.split()) for p in decoded_preds]

        metrics = {
            "bleu": round(bleu, 4),
            "avg_pred_len": round(np.mean(pred_lens), 1),
        }

        # Log a few examples
        n_show = min(3, len(decoded_preds))
        for i in range(n_show):
            logger.info(f"  [eval sample {i}]")
            logger.info(f"    pred: {decoded_preds[i][:150]}")
            logger.info(f"    gold: {decoded_labels[i][:150]}")

        return metrics

    return compute_metrics


def _simple_bleu(preds: List[str], refs: List[str]) -> float:
    """Fallback: simple unigram BLEU approximation."""
    if not preds:
        return 0.0
    scores = []
    for p, r in zip(preds, refs):
        p_tokens = p.lower().split()
        r_tokens = r.lower().split()
        if not p_tokens or not r_tokens:
            scores.append(0.0)
            continue
        r_set = set(r_tokens)
        matches = sum(1 for t in p_tokens if t in r_set)
        precision = matches / len(p_tokens)
        # brevity penalty
        bp = min(1.0, math.exp(1 - len(r_tokens) / max(1, len(p_tokens))))
        scores.append(precision * bp * 100)
    return float(np.mean(scores))


# ---------------------------
# 6) Main Training Function
# ---------------------------
def main():
    cfg = TrainConfig()
    set_seed(cfg.seed)

    logger.info("=" * 60)
    logger.info("ByT5 Fine-Tuning for Akkadian → English")
    logger.info("=" * 60)
    logger.info(f"Model: {cfg.model_name}")
    logger.info(f"Device: {cfg.device}")
    logger.info(f"FP16: {cfg.fp16}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"GPU Memory: {mem_gb:.1f} GB")

    # =====================
    # Load data
    # =====================
    logger.info(f"Loading training data from: {cfg.train_csv}")
    train_df = pd.read_csv(cfg.train_csv)
    logger.info(f"  Loaded {len(train_df)} documents")

    assert "transliteration" in train_df.columns, "train.csv must have 'transliteration' column"
    assert "translation" in train_df.columns, "train.csv must have 'translation' column"

    # Drop rows with empty translations
    train_df = train_df.dropna(subset=["transliteration", "translation"])
    train_df = train_df[
        (train_df["transliteration"].str.strip() != "")
        & (train_df["translation"].str.strip() != "")
    ].reset_index(drop=True)
    logger.info(f"  After filtering: {len(train_df)} documents with valid pairs")

    # =====================
    # Sentence-level splitting
    # =====================
    if cfg.use_sentence_splitting and os.path.exists(cfg.sentence_csv):
        logger.info(f"Loading sentence hints from: {cfg.sentence_csv}")
        sentence_df = pd.read_csv(cfg.sentence_csv)
        data_df = split_documents_to_sentences(train_df, sentence_df)
    else:
        logger.info("Sentence splitting disabled or CSV not found. Using heuristic splitting.")
        data_df = _fallback_sentence_split(train_df)

    logger.info(f"Total training pairs (after splitting): {len(data_df)}")

    # =====================
    # Train/Val split
    # =====================
    # Split by oare_id to avoid data leakage (sentences from same document
    # should not appear in both train and val)
    if "oare_id" in data_df.columns:
        unique_ids = data_df["oare_id"].unique()
        np.random.seed(cfg.seed)
        np.random.shuffle(unique_ids)
        val_size = max(1, int(len(unique_ids) * cfg.val_fraction))
        val_ids = set(unique_ids[:val_size])

        val_df = data_df[data_df["oare_id"].isin(val_ids)].reset_index(drop=True)
        trn_df = data_df[~data_df["oare_id"].isin(val_ids)].reset_index(drop=True)
    else:
        # Random split
        val_size = max(1, int(len(data_df) * cfg.val_fraction))
        indices = np.random.RandomState(cfg.seed).permutation(len(data_df))
        val_df = data_df.iloc[indices[:val_size]].reset_index(drop=True)
        trn_df = data_df.iloc[indices[val_size:]].reset_index(drop=True)

    logger.info(f"Train: {len(trn_df)} pairs | Val: {len(val_df)} pairs")

    # Print some stats
    trn_src_lens = trn_df["transliteration"].str.split().str.len()
    trn_tgt_lens = trn_df["translation"].str.split().str.len()
    logger.info(
        f"Train source length: mean={trn_src_lens.mean():.1f}, "
        f"median={trn_src_lens.median():.1f}, max={trn_src_lens.max()}"
    )
    logger.info(
        f"Train target length: mean={trn_tgt_lens.mean():.1f}, "
        f"median={trn_tgt_lens.median():.1f}, max={trn_tgt_lens.max()}"
    )

    # =====================
    # Load model & tokenizer
    # =====================
    logger.info(f"Loading tokenizer and model: {cfg.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_name)

    # Enable gradient checkpointing for memory efficiency
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
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
    logger.info(f"Val dataset: {len(val_dataset)} samples")

    # =====================
    # Data collator
    # =====================
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding="longest",       # dynamic padding per-batch (more efficient)
        max_length=cfg.max_source_length,
        label_pad_token_id=-100,
    )

    # =====================
    # Training arguments
    # =====================
    training_args = Seq2SeqTrainingArguments(
        output_dir=cfg.output_dir,
        logging_dir=cfg.logging_dir,

        # Training
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler_type,
        fp16=cfg.fp16,

        # Evaluation
        eval_strategy=cfg.eval_strategy,
        save_strategy=cfg.save_strategy,
        load_best_model_at_end=cfg.load_best_model_at_end,
        metric_for_best_model=cfg.metric_for_best_model,
        greater_is_better=cfg.greater_is_better,
        save_total_limit=3,

        # Generation during eval
        predict_with_generate=cfg.predict_with_generate,
        generation_max_length=cfg.generation_max_new_tokens,
        generation_num_beams=cfg.generation_num_beams,

        # Logging
        logging_steps=10,
        report_to="none",       # change to "wandb" if you have wandb set up

        # Misc
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
    # Trainer
    # =====================
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=cfg.early_stopping_patience
            ),
        ],
    )

    # =====================
    # Train!
    # =====================
    logger.info("=" * 60)
    logger.info("Starting training...")
    logger.info(f"  Epochs: {cfg.num_epochs}")
    logger.info(f"  Batch size (per device): {cfg.per_device_train_batch_size}")
    logger.info(f"  Gradient accumulation: {cfg.gradient_accumulation_steps}")
    logger.info(f"  Effective batch size: {cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps}")
    logger.info(f"  Learning rate: {cfg.learning_rate}")
    logger.info(f"  Warmup ratio: {cfg.warmup_ratio}")
    logger.info(f"  Early stopping patience: {cfg.early_stopping_patience}")
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
    logger.info("Running final evaluation on validation set...")
    eval_metrics = trainer.evaluate()
    eval_metrics["eval_samples"] = len(val_dataset)
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info(f"Best model saved at: {final_model_dir}")
    logger.info(f"Final eval BLEU: {eval_metrics.get('eval_bleu', 'N/A')}")
    logger.info("=" * 60)
    logger.info(
        "To use this model for inference, set model_path in your "
        "inference script to the saved model directory."
    )

    # Cleanup
    del model, trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return eval_metrics


# ---------------------------
# 7) Entry point
# ---------------------------
if __name__ == "__main__":
    main()
