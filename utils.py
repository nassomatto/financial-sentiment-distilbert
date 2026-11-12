import pandas as pd
import numpy as np
from transformers import  Trainer, AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
import joblib


DATA_DIR = Path(__file__).resolve().parent
MODEL_PATH = DATA_DIR / "results" / "checkpoint-453"
BASELINE_PATH = DATA_DIR / "pipeline.joblib"
LABEL_NAMES = ["negative", "neutral", "positive"]
MAX_LENGTH = 128


# -------------------------Functions-------------------------
def load_split(name):
    """Load one data split ('train', 'val' or 'test') as a Hugging Face Dataset."""

    ds = load_dataset("csv", data_files={name: str(DATA_DIR / f"{name}.csv")})
    return ds[name]

def load_bert(path=MODEL_PATH):
    """Load the fine-tuned DistilBERT model and its matching tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    return tokenizer, model


def predict_bert(model, tokenizer, dataset):
    """Return predicted class IDs (0/1/2) for every sentence in a HF Dataset."""

    def tokenize(batch):
        return tokenizer(batch["sentence"], truncation=True, max_length=MAX_LENGTH)

    tokenized = dataset.map(tokenize, batched=True)

    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        args=TrainingArguments(
            output_dir=str(DATA_DIR / "tmp_trainer"),
            per_device_eval_batch_size=16,
            report_to="none",
        ),
    )
    logits = trainer.predict(tokenized).predictions
    return np.argmax(logits, axis=-1)




def score(name, y_true, y_pred):
    """Print a classification report and return the key metrics as a dict."""
    print(f"\n===== {name} =====")
    print(classification_report(y_true, y_pred, target_names=LABEL_NAMES, digits=3))

    report = classification_report(
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
