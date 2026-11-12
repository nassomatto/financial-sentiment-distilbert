import pandas as pd
import numpy as np
from transformers import  Trainer, AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
import joblib


# folder this file lives in, so the CSVs are found no matter where the script is run from
DATA_DIR = Path(__file__).resolve().parent

# Load dataset
ds = load_dataset("csv", data_files = {
      "test": str(DATA_DIR/"test.csv")
      })


model_path = DATA_DIR/"results"/"checkpoint-453"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

"""
TOKENIZER
"""
def tokenize(batch):
      return tokenizer(batch["sentence"], truncation=True, max_length=128) 


# tokenizing test set
test_token = ds["test"].map(tokenize, batched=True) 
test = ds["test"]

"""
LOADING BACK THE BASELINE MODEL (TF-IDF Vectorizer + Logistic Regression(balanced))
"""
pipe = joblib.load(DATA_DIR/"pipeline.joblib")

print(pipe.predict(test["sentence"]))