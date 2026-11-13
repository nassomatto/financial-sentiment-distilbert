import pandas as pd
import numpy as np
from utils import load_split, load_bert, predict_bert, score, LABEL_NAMES, DATA_DIR



#load datasets
val = load_split('val')


"""
TOKENIZER + TRAINER AND RUNS
"""

# retreive the tokenizer and the model from checkpoint folder
tokenizer, model = load_bert()

# return predicted class IDs for each sentence in Dataset
# output: np.array of integers [0,2] for each sentence. Length of the list = n (number of sentences)
preds = predict_bert(model, tokenizer, val) 


"""
ERROR ANALYSIS
"""
score("DistilBERT validation set", val['label'], preds)


# Check the missclassified sentences 

df = val.to_pandas()
df = df.rename(columns={'label':'truth'})
df["predicted"] = preds

df_clean = df[["sentence","truth", "predicted"]]


print("\n\nHead of the dataset with 3 neede columns: \n",df_clean.head())

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


df_tagged = pd.read_csv(DATA_DIR/"misclassified_val_tagged.csv") #dataset of missed sentences with labels on them

print("\n\nCounts of missclassifications", df_tagged.value_counts("category"), "\n")

# adding crosstab
# A crosstab is a table that counts combinations of two columns.
# One column’s values become the rows, the other’s become the columns,
# and each cell counts how many rows have that pair.
pd.set_option("display.max_columns", None)
print("\n-------------------CROSSTAB--------------------\n",pd.crosstab(df_tagged["category"], df_tagged["truth"] + "->" + df_tagged["predicted"]))

"""
Debate: 2 errors. label noise. 

Sentiment-laden words (3 errors): all neutral sentences predicted positive or negative. The model reacts to emotionally charged words (e.g. “awards”, “ban”) even when the sentence states a neutral fact.

Plans: 5 errors, either positive -> neutral or neutral -> positive. Could be random error.

missing_ctxt (5): mixed directions. Sentence is a fragment (“However…”, “since then”), so sentiment depends on the previous sentence. A limit of the data, not the model.

numbers (3): all complete flips. Sentiment depends on comparing figures; the model doesn’t read whether numbers went up or down.

other (6): no clear pattern; plain model mistakes.

"""
