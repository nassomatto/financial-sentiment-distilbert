import numpy as np
import pandas as pd
import torch

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    DataCollatorWithPadding,
    Trainer
)
from sklearn.metrics import f1_score, accuracy_score
from pathlib import Path




# folder this file lives in, so the CSVs are found no matter where the script is run from
DATA_DIR = Path(__file__).resolve().parent

ds = load_dataset("csv", data_files={
    "train": str(DATA_DIR / "train.csv"),
    "val":   str(DATA_DIR / "val.csv"),
    "test":  str(DATA_DIR / "test.csv"),
})

"""
TOKENIZER
"""
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
        return tokenizer(batch["sentence"], truncation=True, max_length=128)
      

train_token = ds["train"].map(tokenize, batched=True)
valid_token = ds["val"].map(tokenize, batched=True)


print(train_token)

"""
MODEL
"""
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)
# gives us the pretrained DistilBERT body + fresh classification head.

"""
TRAINER-SETUP
"""
training_arguments = TrainingArguments(output_dir = "./results", #creates a new folder if doesn't exist
                  num_train_epochs = 3, 
                  learning_rate=2e-5, 
                  per_device_train_batch_size=16,
                  per_device_eval_batch_size=16,
                  eval_strategy='epoch')

print(valid_token)

# EVALUATION
# We're gonna compute metrics first, which mean after each epoch we test the weights on the valid set, 
# without nudging any weight, specifically we'll compute F1-scores. That will give us better understanding
# of what each epoch does for us.


def compute_metrics(eval_preds):    
         logits, labels = eval_preds 
         preds = np.argmax(logits, axis=-1)
         return {
                 "accuracy": accuracy_score(labels,preds),
                 "macro_f1": f1_score(labels, preds, average="macro")
         }

"""
TRAINER
"""
trainer=Trainer(
        model,
        training_arguments,
        train_dataset=train_token,
        eval_dataset=valid_token,
        processing_class=tokenizer,
        compute_metrics=compute_metrics
)

trainer.train()
trainer.evaluate()

trainer.save_model("./results")




    
