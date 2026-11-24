import pandas as pd
import numpy as np
from transformers import  Trainer, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments
from datasets import load_dataset
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
import joblib

# ---------- Paths ----------
# utils.py lives in src/, so the project root is one level up.
# All paths are built from it, so scripts work no matter where they are run from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_ROOT / "data" / "raw"          # original Financial PhraseBank files
SPLIT_DIR = PROJECT_ROOT / "data" / "splits"     # train.csv, val.csv, test.csv
OUTPUT_DIR = PROJECT_ROOT / "outputs"            # result tables and error-analysis CSVs
FIG_DIR = PROJECT_ROOT / "figures"               # charts from visualization.py
MODEL_DIR = PROJECT_ROOT / "models"              # saved baseline (not on GitHub)

MODEL_PATH = PROJECT_ROOT / "results" / "checkpoint-453"   # fine-tuned DistilBERT (not on GitHub)
BASELINE_PATH = MODEL_DIR / "pipeline.joblib"

LABEL_NAMES = ["negative", "neutral", "positive"]

MAX_LENGTH = 128


# -------------------------Functions-------------------------

# function loads one data split ('train', 'val' or 'test') as a Hugging Face Dataset.
def load_split(name):

    ds = load_dataset("csv", data_files={name: str(SPLIT_DIR / f"{name}.csv")})
    return ds[name]


def load_bert(path=MODEL_PATH):
    """Load the fine-tuned DistilBERT model and its matching tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    return tokenizer, model


def predict_bert(model, tokenizer, dataset):
    """Return predicted class IDs (0/1/2) for every sentence in a HF Dataset."""

    def tokenize(batch):      #tokenizer 
        return tokenizer(batch["sentence"], truncation=True, max_length=MAX_LENGTH)

    tokenized = dataset.map(tokenize, batched=True) # tokenizing the set 

      # Trainer module, includes: trainer's arguments
    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        args=TrainingArguments(
            output_dir=str(PROJECT_ROOT / "tmp_trainer"),
            per_device_eval_batch_size=16,                        #16 sentences per prediction
            report_to="none",             # don't send logs to tracking tools (TensorBoard, W&B); terminal output only
        ),
    )
    logits = trainer.predict(tokenized).predictions
    return np.argmax(logits, axis=-1)




def score(name, y_true, y_pred):
    """Print a classification report and return the key metrics as a dict."""
    print(f"\n===== {name} =====")
    print(classification_report(y_true, y_pred, target_names=LABEL_NAMES, digits=3, zero_division=0))

    report = classification_report( # makes a class_report a dictionary, so values can be pulled from more easily.
        y_true, y_pred, target_names=LABEL_NAMES, output_dict=True, zero_division=0
    )
    return {
        "model": name,
        "accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "f1_negative": report["negative"]["f1-score"],
        "f1_neutral": report["neutral"]["f1-score"],
        "f1_positive": report["positive"]["f1-score"],
    }
