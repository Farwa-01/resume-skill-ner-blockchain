# NOTE ON WARMUP: the code requests warmup_ratio, but with the transformers version used for the
# reported results this argument was not supported and was skipped (the log prints
# "[make_training_args] warmup_ratio not supported"). The reported results therefore use a linear
# learning-rate decay with NO warmup. To reproduce them exactly, do not add warmup_steps.
# ============================================================================
# Secure & Scalable Skill Extraction — RoBERTa NER Pipeline (CLEANED)
# ----------------------------------------------------------------------------
# Changes vs. the original notebook:
#   - De-duplicated cells (data loading / BIO preprocessing were repeated 3x)
#   - Global random seeds for reproducibility (ablation sampling was unseeded)
#   - Confusion matrix (Table 2a / Figure 8) added, plus a per-class accuracy
#     bar chart (Figure 9) highlighting the boundary-detection weakness
#   - Multi-size dataset-scale ablation (50 / 500 / 1000 / 2000 / 3000 / full)
#   - Per-class seqeval classification report added alongside the aggregate one
#     (strict-mode IOB2 report was dropped: seqeval.scheme isn't importable on
#     every installed version, e.g. some Kaggle images, and the default report
#     already backs the Table 2 numbers reported in the paper)
#   - Learning-curve plot (Figure 6) now combines ALL THREE training stages
#     into one continuous curve (epochs 1-14), fixing a bug where it only
#     plotted Stage 3's 3 epochs because each stage resets its own Trainer's
#     internal epoch counter back to 1
#   - CELL 13 (new, run last): one consolidated Results Summary that gathers
#     Table 2 / 2a / 3 / 4 into a single printed block and a results_summary.json
#     file — paste straight into the manuscript instead of hunting through
#     scattered cell outputs
#   - Post-processing no longer re-merges B-/I- spans with a hardcoded keyword
#     list; the pipeline's own aggregation_strategy already does this from the
#     model's BIO predictions, so the manual merge step was redundant and
#     fragile (silently failing on any multi-word skill not in the list).
# Run as a script (python train_ner.py) or paste cell by cell into Colab/Kaggle.
# Each section is a Colab/Kaggle cell — copy them in order, or run as a script.
# ============================================================================

# ──────────────────────────────────────────────────────────────────────────
# CELL 0 — Setup & reproducibility
# ──────────────────────────────────────────────────────────────────────────
# If not already installed in this environment, run first:
#   !pip install -q evaluate seqeval
# (If seqeval's build fails with an egg_info/setup.py error, try:
#   !pip install -q -U pip setuptools wheel
#   !pip install -q --no-build-isolation evaluate seqeval
#  seaborn is also used in Cell 7 — usually preinstalled on Kaggle/Colab.)

import gc
import os
import re
import json
import random
import logging

import numpy as np
import torch
from tqdm import tqdm

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

gc.collect()
torch.cuda.empty_cache()

# Folder containing the annotated resume JSON files (Kaggle "NER Annotated CVs", folder ResumesJsonAnnotated).
# Set with:  export JSON_DIR=/path/to/ResumesJsonAnnotated   (or edit the default below)
JSON_DIR = os.environ.get("JSON_DIR", "data/ResumesJsonAnnotated")
# Where checkpoints, final models and figures are written.
WORK_DIR = os.environ.get("WORK_DIR", "outputs")
os.makedirs(WORK_DIR, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────
# CELL 1 — Load raw resumes + annotations, normalize to single "SKILL" label
# ──────────────────────────────────────────────────────────────────────────
def trim_entity_spans(data: list) -> list:
    """Strip leading/trailing whitespace from annotated spans so they align
    exactly with text tokens; drop any span that becomes invalid afterward."""
    invalid_span_tokens = re.compile(r"\s")
    cleaned_data = []
    for text, annotations in data:
        entities = annotations["entities"]
        valid_entities = []
        for start, end, label in entities:
            valid_start, valid_end = start, end
            while valid_start < len(text) and invalid_span_tokens.match(text[valid_start]):
                valid_start += 1
            while valid_end > 1 and invalid_span_tokens.match(text[valid_end - 1]):
                valid_end -= 1
            if valid_start < valid_end:
                valid_entities.append((valid_start, valid_end, label))
        cleaned_data.append((text, {"entities": valid_entities}))
    return cleaned_data


def load_training_data(json_dir: str) -> list:
    training_data = []
    skipped = 0
    for filename in tqdm(os.listdir(json_dir)):
        if not filename.endswith(".json"):
            continue
        file_path = os.path.join(json_dir, filename)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                cv_data = json.load(f)

            text = cv_data.get("content", cv_data.get("text", cv_data.get("resume_text", "")))
            text = text.replace("\n", " ").replace("\r", " ").strip()

            annotations = cv_data.get("annotation", cv_data.get("annotations", []))
            if not annotations:
                skipped += 1
                continue

            entities = []
            for ann in annotations:
                if isinstance(ann, list) and len(ann) >= 3:
                    start, end = ann[0], ann[1]
                elif isinstance(ann, dict):
                    point = ann.get("points", [{}])[0]
                    start = point.get("start", 0)
                    end = point.get("end", 0)
                else:
                    continue
                if start < end and start >= 0 and end <= len(text):
                    entities.append((start, end, "SKILL"))

            if text and entities:
                training_data.append((text, {"entities": entities}))
            else:
                skipped += 1

        except Exception as e:
            logging.warning(f"Failed on {filename}: {e}")
            skipped += 1
            continue

    print(f"Loaded {len(training_data)} CVs with annotations ({skipped} skipped/invalid)")
    return training_data


training_data = load_training_data(JSON_DIR)
training_data = trim_entity_spans(training_data)

entity_counts = [len(ents["entities"]) for _, ents in training_data]
print(f"Average skills per CV: {sum(entity_counts) / len(entity_counts):.1f}")
print(f"Min / Max skills: {min(entity_counts)} / {max(entity_counts)}")
print(f"Total CVs after cleaning: {len(training_data)}")


# ──────────────────────────────────────────────────────────────────────────
# CELL 2 — Build BIO-tagged word-level examples, clean Unicode, tokenize
# ──────────────────────────────────────────────────────────────────────────
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

tokenizer = AutoTokenizer.from_pretrained("roberta-base", add_prefix_space=True)

label_list = ["O", "B-SKILL", "I-SKILL"]
label_map = {"O": 0, "B-SKILL": 1, "I-SKILL": 2}
id2label = {v: k for k, v in label_map.items()}


def build_bio_examples(training_data):
    examples = []
    for text, ann in training_data:
        words = text.split()
        bio_tags = [0] * len(words)  # O = 0
        for start, end, _ in ann["entities"]:
            pos = 0
            for i, word in enumerate(words):
                w_start = text.find(word, pos)
                if w_start == -1:
                    continue
                w_end = w_start + len(word)
                if w_end <= start or w_start >= end:
                    pos = w_end
                    continue
                if w_start >= start and w_end <= end:
                    bio_tags[i] = 2 if (i > 0 and bio_tags[i - 1] in (1, 2)) else 1
                elif w_start < end and w_end > start:
                    bio_tags[i] = 1
                pos = w_end
        if words:
            examples.append({"tokens": words, "ner_tags": bio_tags})
    return examples


def clean_invalid_unicode(text: str) -> str:
    cleaned = "".join(c for c in text if not (0xD800 <= ord(c) <= 0xDFFF))
    return " ".join(cleaned.split())


examples = build_bio_examples(training_data)
print(f"Created {len(examples)} examples with BIO tags")

clean_examples = []
for ex in examples:
    clean_tokens = [clean_invalid_unicode(t) for t in ex["tokens"]]
    if any(len(t.strip()) > 0 for t in clean_tokens):
        clean_examples.append({"tokens": clean_tokens, "ner_tags": ex["ner_tags"]})
print(f"After surrogate cleaning: {len(clean_examples)} kept (dropped {len(examples) - len(clean_examples)})")

# NOTE: fixed random_state=42 (same seed as CELL 0) -> this 90:10 split is reproducible
train_ex, val_ex = train_test_split(clean_examples, test_size=0.1, random_state=SEED)
print(f"Train: {len(train_ex)} | Val: {len(val_ex)}")

train_dataset = Dataset.from_list(train_ex)
val_dataset = Dataset.from_list(val_ex)


def tokenize_and_align_labels(batch):
    tokenized = tokenizer(
        batch["tokens"],
        truncation=True,
        is_split_into_words=True,
        max_length=512,
        padding="max_length",
        return_offsets_mapping=True,
        return_tensors=None,
    )
    all_labels = []
    for i in range(len(batch["tokens"])):
        word_ids = tokenized.word_ids(batch_index=i)
        label_ids = [-100] * len(tokenized["input_ids"][i])
        prev_word_idx = None
        for token_idx, word_idx in enumerate(word_ids):
            if word_idx is None:
                continue
            if word_idx != prev_word_idx:
                label_ids[token_idx] = batch["ner_tags"][i][word_idx]
            else:
                if label_ids[token_idx - 1] in (1, 2):
                    label_ids[token_idx] = 2
            prev_word_idx = word_idx
        all_labels.append(label_ids)
    tokenized["labels"] = all_labels
    return tokenized


tokenized_train = train_dataset.map(
    tokenize_and_align_labels, batched=True, remove_columns=train_dataset.column_names, desc="Tokenizing train"
)
tokenized_val = val_dataset.map(
    tokenize_and_align_labels, batched=True, remove_columns=val_dataset.column_names, desc="Tokenizing val"
)
dataset = DatasetDict({"train": tokenized_train, "validation": tokenized_val})
print("\nDataset ready:", dataset)


# ──────────────────────────────────────────────────────────────────────────
# CELL 2b — Version-proof TrainingArguments builder
# ──────────────────────────────────────────────────────────────────────────
# Different transformers versions accept different kwarg names (e.g. some
# builds want `warmup_ratio`, older/renamed ones want `warmup_steps`; some
# want `eval_strategy`, older ones want `evaluation_strategy`). Instead of
# hardcoding one set of names, this inspects the installed TrainingArguments
# signature and only passes kwargs it actually accepts, substituting a known
# equivalent where one exists, and dropping the rest with a printed warning
# rather than crashing.
import inspect
from transformers import TrainingArguments

_TA_PARAMS = set(inspect.signature(TrainingArguments.__init__).parameters.keys())

_EQUIVALENTS = {
    "eval_strategy": ["eval_strategy", "evaluation_strategy"],
    "warmup_ratio": ["warmup_ratio"],  # no direct equivalent -> handled specially below
}


def make_training_args(warmup_ratio=None, eval_strategy_value="epoch", **kwargs):
    """Build TrainingArguments using only kwargs the installed transformers
    version supports."""
    final_kwargs = {}

    # eval_strategy / evaluation_strategy naming differs across versions
    if "eval_strategy" in _TA_PARAMS:
        final_kwargs["eval_strategy"] = eval_strategy_value
    elif "evaluation_strategy" in _TA_PARAMS:
        final_kwargs["evaluation_strategy"] = eval_strategy_value

    # warmup_ratio vs warmup_steps
    if warmup_ratio is not None:
        if "warmup_ratio" in _TA_PARAMS:
            final_kwargs["warmup_ratio"] = warmup_ratio
        elif "warmup_steps" in _TA_PARAMS:
            # approximate: caller passes total_train_steps via kwargs if precise
            # conversion is needed; otherwise this is skipped with a warning.
            print(f"[make_training_args] warmup_ratio not supported; "
                  f"pass warmup_steps explicitly if precise warmup is needed.")
        else:
            print("[make_training_args] WARNING: no warmup kwarg supported in this "
                  "transformers version; warmup is being skipped.")

    dropped = []
    for k, v in kwargs.items():
        if k in _TA_PARAMS:
            final_kwargs[k] = v
        else:
            dropped.append(k)

    if dropped:
        print(f"[make_training_args] NOTE: dropped unsupported kwargs for this "
              f"transformers version: {dropped}")

    return TrainingArguments(**final_kwargs)


# ──────────────────────────────────────────────────────────────────────────
# CELL 3 — Metrics (seqeval) shared by all training runs
# ──────────────────────────────────────────────────────────────────────────
# NOTE: we deliberately do NOT use evaluate.load("seqeval"). That downloads a
# community metric script from the HF Hub whose cached version calls
# seqeval's classification_report(..., output_dict=True); if the locally
# installed `seqeval` package predates that parameter (as happened here),
# training crashes mid-run. Calling seqeval's own precision_score/recall_score/
# f1_score/accuracy_score directly avoids that mismatch entirely and gives
# identical numbers (same default scheme), with no dependency on evaluate
# or on what metric-script version is cached on the Hub.
from seqeval.metrics import (
    precision_score as seqeval_precision,
    recall_score as seqeval_recall,
    f1_score as seqeval_f1,
    accuracy_score as seqeval_accuracy,
    classification_report as seqeval_report,
)


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    true_labels = [[label_list[l] for l in row if l != -100] for row in labels]
    true_predictions = [
        [label_list[p] for p, l in zip(prow, lrow) if l != -100] for prow, lrow in zip(predictions, labels)
    ]

    return {
        "precision": seqeval_precision(true_labels, true_predictions),
        "recall": seqeval_recall(true_labels, true_predictions),
        "f1": seqeval_f1(true_labels, true_predictions),
        "accuracy": seqeval_accuracy(true_labels, true_predictions),
    }


# ──────────────────────────────────────────────────────────────────────────
# CELL 4 — Multi-stage training (Stage 1: 5 epochs, lr=2e-5)
# ──────────────────────────────────────────────────────────────────────────
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer

gc.collect()
torch.cuda.empty_cache()

model = AutoModelForTokenClassification.from_pretrained(
    "roberta-base", num_labels=len(label_list), id2label=id2label, label2id=label_map
).to("cuda" if torch.cuda.is_available() else "cpu")

stage1_args = make_training_args(
    output_dir=f"{WORK_DIR}/roberta-ner-skills",
    num_train_epochs=5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=8,  # effective batch size 32
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    eval_strategy_value="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=False,
    fp16=True,
    optim="adamw_torch",
    logging_steps=20,
    seed=SEED,
    report_to="none",
)

trainer = Trainer(
    model=model, args=stage1_args,
    train_dataset=dataset["train"], eval_dataset=dataset["validation"],
    compute_metrics=compute_metrics,
)

print("Stage 1/3: coarse fitting (5 epochs, lr=2e-5)...")
trainer.train()
stage1_eval = trainer.evaluate()
stage1_log = trainer.state.log_history  # saved for the combined learning curve (Cell 9)
print("Stage 1 eval:", stage1_eval)

trainer.save_model(f"{WORK_DIR}/best_roberta_ner")
tokenizer.save_pretrained(f"{WORK_DIR}/best_roberta_ner")


# ──────────────────────────────────────────────────────────────────────────
# CELL 5 — Stage 2: refinement (6 epochs, lr=5e-6)
# ──────────────────────────────────────────────────────────────────────────
gc.collect()
torch.cuda.empty_cache()

model = AutoModelForTokenClassification.from_pretrained(
    f"{WORK_DIR}/best_roberta_ner", num_labels=len(label_list), id2label=id2label, label2id=label_map
).to("cuda" if torch.cuda.is_available() else "cpu")

stage2_args = make_training_args(
    output_dir=f"{WORK_DIR}/roberta-ner-more-epochs",
    num_train_epochs=6,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=8,
    learning_rate=5e-6,
    weight_decay=0.01,
    warmup_ratio=0.05,
    eval_strategy_value="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    fp16=True,
    optim="adamw_torch",
    logging_steps=20,
    seed=SEED,
    report_to="none",
)

trainer = Trainer(
    model=model, args=stage2_args,
    train_dataset=dataset["train"], eval_dataset=dataset["validation"],
    compute_metrics=compute_metrics,
)

print("Stage 2/3: refinement (6 epochs, lr=5e-6)...")
trainer.train()
stage2_eval = trainer.evaluate()
stage2_log = trainer.state.log_history
print("Stage 2 eval:", stage2_eval)

trainer.save_model(f"{WORK_DIR}/best_roberta_ner_v2")
tokenizer.save_pretrained(f"{WORK_DIR}/best_roberta_ner_v2")


# ──────────────────────────────────────────────────────────────────────────
# CELL 6 — Stage 3: final boundary fine-tuning (3 epochs, lr=1e-6)
# ──────────────────────────────────────────────────────────────────────────
gc.collect()
torch.cuda.empty_cache()

model = AutoModelForTokenClassification.from_pretrained(
    f"{WORK_DIR}/best_roberta_ner_v2", num_labels=len(label_list), id2label=id2label, label2id=label_map
).to("cuda" if torch.cuda.is_available() else "cpu")

stage3_args = make_training_args(
    output_dir=f"{WORK_DIR}/roberta-ner-final-3epochs",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=8,
    learning_rate=1e-6,
    weight_decay=0.01,
    warmup_ratio=0.1,
    eval_strategy_value="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    fp16=True,
    optim="adamw_torch",
    logging_steps=20,
    seed=SEED,
    report_to="none",
)

trainer = Trainer(
    model=model, args=stage3_args,
    train_dataset=dataset["train"], eval_dataset=dataset["validation"],
    compute_metrics=compute_metrics,
)

print("Stage 3/3: final fine-tuning (3 epochs, lr=1e-6)...")
trainer.train()

final_eval = trainer.evaluate()
stage3_log = trainer.state.log_history
print("\nFinal evaluation (Table 2 numbers):")
for k, v in final_eval.items():
    print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

trainer.save_model(f"{WORK_DIR}/best_roberta_ner_final")
tokenizer.save_pretrained(f"{WORK_DIR}/best_roberta_ner_final")
print(f"\nFinal model saved to {WORK_DIR}/best_roberta_ner_final")


# ──────────────────────────────────────────────────────────────────────────
# CELL 7 — Confusion matrix (Table 2a) + strict-mode seqeval report
# ──────────────────────────────────────────────────────────────────────────
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

predictions_output = trainer.predict(dataset["validation"])
logits, labels = predictions_output.predictions, predictions_output.label_ids
preds = np.argmax(logits, axis=-1)

flat_true, flat_pred = [], []
for pred_row, label_row in zip(preds, labels):
    for p, l in zip(pred_row, label_row):
        if l != -100:
            flat_true.append(l)
            flat_pred.append(p)

cm = confusion_matrix(flat_true, flat_pred, labels=[0, 1, 2])  # O, B-SKILL, I-SKILL
print("Confusion matrix (rows=actual, cols=predicted), labels =", label_list)
print(cm)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_list, yticklabels=label_list)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Token-Level Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.show()

true_labels = [[label_list[l] for l in row if l != -100] for row in labels]
true_predictions = [
    [label_list[p] for p, l in zip(prow, lrow) if l != -100] for prow, lrow in zip(preds, labels)
]

print("\nSeqeval classification report (per-class precision/recall/F1 — paste into Table 2a discussion):")
print(seqeval_report(true_labels, true_predictions, digits=4))

# Per-class accuracy bar chart (O vs B-SKILL vs I-SKILL) — highlights the
# boundary-detection weakness (B-SKILL has the lowest per-class accuracy)
per_class_acc = (cm.diagonal() / cm.sum(axis=1)) * 100
plt.figure(figsize=(6, 5))
bars = plt.bar(label_list, per_class_acc, color=["#4C72B0", "#DD8452", "#55A868"])
plt.ylabel("Per-Class Accuracy (%)")
plt.ylim(0, 100)
plt.title("Per-Class Token Accuracy (O / B-SKILL / I-SKILL)")
for bar, val in zip(bars, per_class_acc):
    plt.text(bar.get_x() + bar.get_width() / 2, val + 1.5, f"{val:.1f}%", ha="center")
plt.tight_layout()
plt.savefig("per_class_accuracy.png", dpi=300)
plt.show()


# ──────────────────────────────────────────────────────────────────────────
# CELL 8 — Multi-size dataset-scale ablation (Table 4 / Figure 7)
# ──────────────────────────────────────────────────────────────────────────
ablation_sizes = [50, 500, 1000, 2000, 3000, len(dataset["train"])]
ablation_results = {}
full_train_list = dataset["train"].to_list()

for size in ablation_sizes:
    print(f"\n=== Ablation: training on {size} examples ===")
    random.seed(SEED)  # re-seed before each sample draw for independent, reproducible subsets
    subset = random.sample(full_train_list, size) if size < len(full_train_list) else full_train_list
    subset_dataset = Dataset.from_list(subset)

    gc.collect()
    torch.cuda.empty_cache()

    model_i = AutoModelForTokenClassification.from_pretrained(
        "roberta-base", num_labels=len(label_list), id2label=id2label, label2id=label_map
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    args_i = make_training_args(
        output_dir=f"{WORK_DIR}/ablation_{size}",
        num_train_epochs=8 if size < 1000 else 5,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy_value="epoch",
        save_strategy="no",
        fp16=True,
        optim="adamw_torch",
        logging_steps=20,
        seed=SEED,
        report_to="none",
    )

    trainer_i = Trainer(
        model=model_i, args=args_i,
        train_dataset=subset_dataset, eval_dataset=dataset["validation"],
        compute_metrics=compute_metrics,
    )
    trainer_i.train()
    res = trainer_i.evaluate()
    ablation_results[size] = res
    print(f"{size} CVs -> P={res['eval_precision']:.4f}  R={res['eval_recall']:.4f}  F1={res['eval_f1']:.4f}")

    del model_i, trainer_i
    gc.collect()
    torch.cuda.empty_cache()

print("\nFinal ablation table (paste into Table 4):")
print(f"{'Dataset Size':<15}{'Precision':<12}{'Recall':<12}{'F1-score':<12}")
for size, res in ablation_results.items():
    print(f"{size:<15}{res['eval_precision']:<12.4f}{res['eval_recall']:<12.4f}{res['eval_f1']:<12.4f}")

sizes_sorted = sorted(ablation_results.keys())
f1_values = [ablation_results[s]["eval_f1"] for s in sizes_sorted]

plt.figure(figsize=(7, 5))
plt.plot(sizes_sorted, f1_values, marker="o", linewidth=2)
plt.xscale("log")
plt.xlabel("Training Set Size (log scale)")
plt.ylabel("Entity F1 Score")
plt.title("Impact of Dataset Scale on Performance")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("scale_ablation_full.png", dpi=300)
plt.show()


# ──────────────────────────────────────────────────────────────────────────
# CELL 9 — Full learning-curve plot (Figure 6), combining ALL THREE stages
# ──────────────────────────────────────────────────────────────────────────
# IMPORTANT: Stage 1/2/3 each reset the Trainer's internal epoch counter to
# start at 1, so their log_history entries are NOT globally numbered. We
# stitch them into one continuous x-axis (Stage 1: epochs 1-5, Stage 2:
# epochs 6-11, Stage 3: epochs 12-14) matching the "5+6+3" schedule described
# in Section 3.9/Table 1 of the manuscript. This replaces the earlier version
# of this cell, which only plotted Stage 3's 3 epochs.


def extract_epoch_f1(log_history, epoch_offset=0):
    pts = []
    for entry in log_history:
        if "eval_f1" in entry:
            pts.append((entry.get("epoch", 0) + epoch_offset, entry["eval_f1"]))
    return pts


stage1_pts = extract_epoch_f1(stage1_log, epoch_offset=0)
stage2_pts = extract_epoch_f1(stage2_log, epoch_offset=5)   # continues after Stage 1's 5 epochs
stage3_pts = extract_epoch_f1(stage3_log, epoch_offset=11)  # continues after Stage 1+2's 11 epochs

all_pts = stage1_pts + stage2_pts + stage3_pts
all_pts.sort(key=lambda p: p[0])
epochs = [p[0] for p in all_pts]
f1_scores = [p[1] for p in all_pts]

print("Combined learning curve (epoch, F1):")
for e, f in all_pts:
    print(f"  epoch {e:>4.1f}  F1 = {f:.4f}")

plt.figure(figsize=(8, 5))
plt.plot(epochs, f1_scores, marker="o", color="blue", linewidth=2, label="Validation F1")
plt.axvline(x=5, color="gray", linestyle=":", alpha=0.7, label="Stage 1 -> 2")
plt.axvline(x=11, color="gray", linestyle="--", alpha=0.7, label="Stage 2 -> 3")
plt.xlabel("Epoch (combined across all 3 training stages)")
plt.ylabel("Entity F1 Score")
plt.title("Learning Curve: Validation F1 over Epochs (Full 5+6+3 Schedule)")
plt.grid(True, linestyle="--", alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig("learning_curve_full.png", dpi=300)
plt.show()


# ──────────────────────────────────────────────────────────────────────────
# CELL 10 — Final metrics bar chart (Figure 5)
# ──────────────────────────────────────────────────────────────────────────
metrics = ["Precision", "Recall", "F1"]
values = [final_eval["eval_precision"], final_eval["eval_recall"], final_eval["eval_f1"]]

plt.figure(figsize=(6, 5))
bars = plt.bar(metrics, values, color=["teal", "goldenrod", "purple"])
plt.ylim(0, max(values) + 0.15)
plt.title("Final Model Performance on Validation Set")
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.01, f"{yval:.3f}", ha="center")
plt.tight_layout()
plt.savefig("final_metrics_bar.png", dpi=300)
plt.show()


# ──────────────────────────────────────────────────────────────────────────
# CELL 11 — Inference pipeline + cleaned post-processing (no fragile manual
# merge step; relies on aggregation_strategy to assemble B-/I- spans)
# ──────────────────────────────────────────────────────────────────────────
from transformers import pipeline

ner = pipeline(
    "ner",
    model=f"{WORK_DIR}/best_roberta_ner_final",
    tokenizer=f"{WORK_DIR}/best_roberta_ner_final",
    aggregation_strategy="average",
    device=0 if torch.cuda.is_available() else -1,
)

NON_SKILL_STOPWORDS = {
    "results", "oriented", "engineer", "proficient", "summary", "an", "analytical",
    "driven", "application", "development", "scripting", "coding", "automation",
    "design", "technical", "skills", "databases", "build", "deploy", "professional",
    "full-stack", "software", "web", "years", "experience", "familiar", "also",
}


def extract_raw(text: str):
    """Model output with no post-processing, for before/after comparison."""
    return [e["word"].strip(",. ") for e in ner(text) if e["score"] > 0.5]


def extract_cleaned(text: str):
    """Filters stopwords/low-value tokens from the model's own aggregated
    entities. Does NOT re-merge spans — aggregation_strategy already groups
    B-/I- continuations and subwords using the model's BIO predictions."""
    results = ner(text)
    cleaned, seen = [], set()
    for e in results:
        word = e["word"].strip(",. \n")
        if not word or len(word) < 3:
            continue
        word_lower = word.lower()
        if word_lower in NON_SKILL_STOPWORDS or word.isdigit():
            continue
        if word in seen:
            continue
        seen.add(word)
        cleaned.append(f"{word:<40} -> SKILL (score: {e['score']:.3f})")
    return cleaned


# ──────────────────────────────────────────────────────────────────────────
# CELL 12 — Qualitative test set (before/after post-processing)
# ──────────────────────────────────────────────────────────────────────────
qualitative_tests = [
    "Expert in Python, TensorFlow, Docker, Kubernetes, and Agile.",
    "Software Engineer with 5 years experience in full-stack development using React.js and Node.js.",
    "Results-oriented professional proficient in project management and team leadership.",
    "Technical Skills: JavaScript, React, Angular, Spring Boot, MongoDB, AWS, Git, CI/CD.",
    "Strong communication skills, problem-solving abilities, and experience in Scrum and Kanban.",
    "Proficient in HTML, CSS, JavaScript, TypeScript, Next.js, Tailwind CSS, PostgreSQL, Firebase.",
]

for i, text in enumerate(qualitative_tests, 1):
    print(f"\nTest {i}: {text.strip()}")
    print("  Raw model output:     ", extract_raw(text))
    print("  After post-processing:")
    for line in extract_cleaned(text):
        print("   ", line)


# ──────────────────────────────────────────────────────────────────────────
# CELL 13 — Consolidated Results Summary (everything the paper needs, in
# one place) — run this LAST, after Cells 4-8 have all completed.
# ──────────────────────────────────────────────────────────────────────────
import json
import datetime

summary = {
    "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
    "dataset": {
        "total_cvs_loaded": len(training_data),
        "train_examples": len(dataset["train"]),
        "val_examples": len(dataset["validation"]),
    },
    "table2_final_model": {
        "precision": round(final_eval["eval_precision"], 4),
        "recall": round(final_eval["eval_recall"], 4),
        "f1": round(final_eval["eval_f1"], 4),
        "accuracy": round(final_eval["eval_accuracy"], 4),
    },
    "table2a_confusion_matrix": {
        "labels": label_list,
        "matrix": cm.tolist(),  # rows = actual, cols = predicted
        "per_class_accuracy_pct": {
            label_list[i]: round(float(per_class_acc[i]), 2) for i in range(len(label_list))
        },
    },
    "table3_stage_eval": {
        "stage1_5ep": {k: round(v, 4) if isinstance(v, float) else v for k, v in stage1_eval.items()},
        "stage2_11ep": {k: round(v, 4) if isinstance(v, float) else v for k, v in stage2_eval.items()},
        "stage3_14ep_final": {k: round(v, 4) if isinstance(v, float) else v for k, v in final_eval.items()},
    },
    "table4_ablation": {
        str(size): {
            "precision": round(res["eval_precision"], 4),
            "recall": round(res["eval_recall"], 4),
            "f1": round(res["eval_f1"], 4),
        }
        for size, res in ablation_results.items()
    },
    "figures_saved": [
        "confusion_matrix.png",
        "per_class_accuracy.png",
        "learning_curve_full.png",
        "scale_ablation_full.png",
        "final_metrics_bar.png",
    ],
}

print("=" * 70)
print("RESULTS SUMMARY — paste directly into the manuscript")
print("=" * 70)

print(f"\nDataset: {summary['dataset']['total_cvs_loaded']} CVs loaded -> "
      f"{summary['dataset']['train_examples']} train / {summary['dataset']['val_examples']} val")

print("\nTable 2 — Overall Model Performance (final, 3-stage model):")
for k, v in summary["table2_final_model"].items():
    print(f"  {k:<10}: {v}")

print("\nTable 2a — Confusion Matrix (rows=actual, cols=predicted):")
print(f"  {'':>10}" + "".join(f"{l:>10}" for l in label_list))
for i, row in enumerate(cm):
    print(f"  {label_list[i]:>10}" + "".join(f"{v:>10}" for v in row))
print("  Per-class accuracy:", summary["table2a_confusion_matrix"]["per_class_accuracy_pct"])

print("\nTable 3 — F1 at end of each training stage:")
for stage, vals in summary["table3_stage_eval"].items():
    print(f"  {stage:<16}: F1 = {vals['eval_f1']}")

print("\nTable 4 — Dataset-Scale Ablation:")
print(f"  {'Size':<10}{'Precision':<12}{'Recall':<12}{'F1':<10}")
for size, vals in summary["table4_ablation"].items():
    print(f"  {size:<10}{vals['precision']:<12}{vals['recall']:<12}{vals['f1']:<10}")

with open("results_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("\nFull summary saved to results_summary.json")
print("Figures saved:", ", ".join(summary["figures_saved"]))