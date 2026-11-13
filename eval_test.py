import pandas as pd
import numpy as np
from utils import load_split, load_bert, predict_bert, score, LABEL_NAMES, DATA_DIR
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
import joblib

test = load_split('test') # load test set


"""
LOADING BACK THE BASELINE MODELS (1. Majority baseline 2.TF-IDF Vectorizer + Logistic Regression(balanced))
"""

#1 Baseline 1 - all 3448 rows with "negative" label
baseline1_prediction = len(test) * [1] 

#2 Baseline 2
pipe = joblib.load(DATA_DIR/"pipeline.joblib")

preds_baseline = pipe.predict(test['sentence'])

"""
TOKENIZER + TRAINER AND RUNS
"""
tokenizer, model = load_bert()
preds = predict_bert(model, tokenizer, test)



"""
FINAL ASSESSMENT
"""
results = []

predictions_by_model = {'Majority Class Report' : len(test) * [1],
               'TF-IDF LogReg(balanced) Report' : preds_baseline,
               'DistilBERT': preds
               }

for key, value in predictions_by_model.items():
      results.append(score(key, test['label'], value))

df = pd.DataFrame(results)
print(df.round(3))      
df.to_csv(DATA_DIR / "test_results.csv", index=False)


