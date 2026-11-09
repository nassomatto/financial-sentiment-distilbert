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


print(classification_report(labels, preds, target_names = ["negative", "neutral", "positive"]))
print(confusion_matrix(labels, preds))


# Check the missclassified sentences 

df = valid_token.to_pandas()
df["truth"] = labels
df["predicted"] = preds
df_clean = df[["sentence","truth", "predicted"]]


print(df_clean.head())

#filter down the missclassified 
df_missed = df_clean[df_clean["truth"] != df_clean["predicted"]].copy()
#renaming back the variables
df_missed["truth"] = df_missed["truth"].map({0: "negative", 1: "neutral", 2: "positive"}) 
df_missed["predicted"] = df_missed["predicted"].map({0: "negative", 1: "neutral", 2: "positive"})

#saving to csv file
df_missed.to_csv(DATA_DIR / "misclassified_val.csv", index=False)

"""
MANUALLY ANALYSING THE CONTEXT OF ERRORS  


I've added a column named 'category' to manually label what type of mismatch happened.
'category' values are: 

- 'numbers': meaning depends on comparing figures(loss grew, stock fell vs index rose)
- 'missing context(missing_ctxt)': sentence starts with 'However','since then','this'
- 'plan': about intentions, plans, or announcements, not results (plans to invest, is developing)
- 'upbeat/downbeat (fluct_word) words': neutral fact, but contains words that sound good or bad (awards, strengthen, ban, rejected)
- 'debatable label': truly debatable for human labels.
- 'other': model is plainly wrong and nothing above explains it 

"""


df_tagged = pd.read_csv(DATA_DIR/"misclassified_val_tagged.csv")

print("\n\n", df_tagged.value_counts("category"), "\n")

# adding crosstab
# A crosstab is a table that counts combinations of two columns.
# One column’s values become the rows, the other’s become the columns,
# and each cell counts how many rows have that pair.
pd.set_option("display.max_columns", None)
print(pd.crosstab(df_tagged["category"], df_tagged["truth"] + "->" + df_tagged["predicted"]))

"""
Debate: 2 errors. label noise. 

Sentiment-laden words (3 errors): all neutral sentences predicted positive or negative. The model reacts to emotionally charged words (e.g. “awards”, “ban”) even when the sentence states a neutral fact.

Plans: 5 errors, either positive -> neutral or neutral -> positive. Could be random error.

missing_ctxt (5): mixed directions. Sentence is a fragment (“However…”, “since then”), so sentiment depends on the previous sentence. A limit of the data, not the model.

numbers (3): all complete flips. Sentiment depends on comparing figures; the model doesn’t read whether numbers went up or down.

other (6): no clear pattern; plain model mistakes.

"""
