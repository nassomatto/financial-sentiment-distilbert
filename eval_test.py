import pandas as pd
from utils import load_split, load_bert, predict_bert, score, DATA_DIR, BASELINE_PATH
import joblib

test = load_split('test') # load test set


"""
LOADING BACK THE BASELINE MODELS (1. Majority baseline 2.TF-IDF Vectorizer + Logistic Regression(balanced))
"""

#1 Baseline 1 - all 3448 rows with "negative" label
baseline1_prediction = len(test) * [1] 

#2 Baseline 2
pipe = joblib.load(BASELINE_PATH)

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

predictions_by_model = {'Majority Class Report' : baseline1_prediction,
               'TF-IDF LogReg(balanced) Report' : preds_baseline,
               'DistilBERT': preds
               }

for key, value in predictions_by_model.items():
      results.append(score(key, test['label'], value))

df = pd.DataFrame(results)
print(df.round(3))      
df.to_csv(DATA_DIR / "test_results.csv", index=False)



# Saving individual prediction columns
test_preds_df = pd.DataFrame({
      'true':test['label'],
      'predicted':preds
})

test_preds_df.to_csv(DATA_DIR / "test-predictions.csv", index=False)


