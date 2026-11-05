import pandas as pd
import numpy as np
from transformers import  Trainer, AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix

# folder this file lives in, so the CSVs are found no matter where the script is run from
DATA_DIR = Path(__file__).resolve().parent

#load datasets
ds = load_dataset("csv", data_files={
      "val": str(DATA_DIR / "val.csv"),
      #"test": str(DATA_DIR / "test.csv"),
})

#load model content into this script
model_path = DATA_DIR/"results"/"checkpoint-453"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

"""
TOKENIZER
"""
def tokenize(batch):
      return tokenizer(batch["sentence"], truncation=True, max_length=128)


# tokenizing val and test sets
valid_token = ds["val"].map(tokenize, batched=True)
#test_token = ds["test"].map(tokenize, batched=True)

"""
TRAINER AND RUNS
"""
trainer = Trainer(
      model,
      processing_class=tokenizer,
)

validation_results = trainer.predict(valid_token)

"""
ERROR ANALYSIS
"""
preds = np.argmax(validation_results.predictions, axis=-1)
labels = validation_results.label_ids
accuracy = accuracy_score(labels, preds)
macro_f1 = f1_score(labels, preds, average="macro")

print(accuracy, "\n", macro_f1)


print(classification_report(labels, preds))
print(confusion_matrix(labels, preds))

