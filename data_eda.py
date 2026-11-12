import pandas as pd
from pathlib import Path

# folder this file lives in, so the data files are found no matter where the script is run from
DATA_DIR = Path(__file__).resolve().parent

from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
import joblib # for transferring pipeline


"""
EDA of THE HuggingFace Dataset
"Financial Sentiment Analysis"

Metric of scoring the model: macro F1 scores.


by takala: https://huggingface.co/takala
link: https://huggingface.co/datasets/takala/financial_phrasebank
"""

# Check how many rows each agreement-sentence file contains 
for tier in ["50Agree", "66Agree", "75Agree", "AllAgree"]:
    df = pd.read_csv(DATA_DIR / f"Sentences_{tier}.txt", sep="@",
                     names=["sentence", "label"], encoding="latin-1")
    print(tier,"  Number of rows ", df.shape[0])

# EDA of Sentence_75. Decided work on 75% fin. experts agree, as the highest agreement rate and number of rows greater than the "AllAgree" text.
df = pd.read_csv(DATA_DIR / "Sentences_75Agree.txt", sep="@", names=["sentence", "label"], encoding="latin-1")
print("\nAbout Sentences_75Agree.txt - dimensions: ", df.shape,"\n", df.describe(include="object"), "\n")
df.info()

###
print("\nExtra:\n",df.head())
print(df['label'].value_counts(normalize=True)*100) # shows proportion of specific label/total 
print(df['sentence'].duplicated().sum(),"\n\n")
#output: 5
#check them 
print(df[df[['sentence', 'label']].duplicated(keep=False)])
df = df.drop_duplicates()   #drop duplicates


print("\n\nInfo of words of a sentence:\n", df['sentence'].str.split().str.len().describe())

# The imbalance states this: 
# 62.1% of the sentences - neutral
# 25.7% - positive
# 12.2% - negative.

# This means, when we split the dataset into 3 piles(train, valid, test), 
# we should split the sentences from each category keeping the same proportions of the main dataset.


"""SPLITTING THE DATA

# 1. Create a sample dataset with 3 target categories (neutral, positive, neagative)
# Target distribution: % Class 0(neg) - 12%, Class 1(neutral) - 62%, Class 2(pos) - 26% 

    total rows: 3448
    stratifying them: 70/15/15 — train ~2,414, val ~517, test ~517
    from train_valid: valid/train == 517/2931=0.176
"""

#proportion = df['label'].value_counts(normalize=True)

# prep
df['label'] = df['label'].map({'negative':0, 'neutral':1, 'positive':2}) #renames the values 

# splitting
from sklearn.model_selection import train_test_split

#first splitting to train + test
train_valid, test = train_test_split(
    df, test_size=0.15, stratify=df["label"], random_state=42
)

#second split the train into train and validation (0.176 because 15% of original is 0.176% of train_val)
train, val = train_test_split(
    train_valid, test_size=0.176, stratify=train_valid['label'], random_state=42
)

for name, d in [("train", train), ("val", val), ("test", test)]:
    print(name, len(d))
    print(d['label'].value_counts(normalize=True))



"""
We're gonna run TWO models to check how the dataset works 
and later on compare our main NLP model to these basic statistic ones.

BASELINES:

1. No Model
Construct a prediction array of all 1s (neutral), same length as the test set.
Score with accuracy_score and f1_score(average='macro'). No training, no model object.

2. Logistic Regression
Fit TfidfVectorizer on train['sentence'], transform val and test with it. Fit LogisticRegression on the resulting train matrix and train['label']. 
Predict on the test matrix. Score with classification_report for per-class and macro F1.
"""


#1 Baseline 1 - all 3448 rows with "negative label"
base1_prediction = len(val) * [1] 
truth1 = val['label']
print("\n\n\nOutput for Baseline 1:\n")
print("Accuracy score:", accuracy_score(truth1, base1_prediction),"  F1 score:", f1_score(truth1, base1_prediction, average='macro'))
""" As we see, the accuracy score corresponds to the proportion of neutral labels primarily in the sentence. """


#2 Baseline 2 - model that vectorizes the sentences and uses logistic regression to classify words into 3 categories.

from sklearn.feature_extraction.text import TfidfVectorizer

#init. vectorizer
vectorizer = TfidfVectorizer()
#defining documents
documents = train["sentence"]

# For training set
# vectorizing -> getting a tf-idf matrix 
X_train = vectorizer.fit_transform(documents)

#For validation set
X_val = vectorizer.transform(val["sentence"])
print("\nOutput for baseline 2:")



# Classification
#Initialize and fit Logistic Regression
model = LogisticRegression()
model.fit(X_train, train["label"])

#Predict
y_pred = model.predict(X_val)
y_pred_proba = model.predict_proba(X_val)

#Score
print(classification_report(val["label"], y_pred))

"""
Explanation of the output: The baseline 2 model has a macro of 0.74, which is not bad. The problem is, the recall of both negative and positive is basically low. That means, the model overclassifies(relies) a sentence more for Neutral, missing its real classes.
"""

# Counteract the model's tendency to default to majority class
model_balanced = LogisticRegression(class_weight='balanced')
model_balanced.fit(X_train, train["label"])
y_pred_balanced = model_balanced.predict(X_val)
print("\Balanced Logistic Regression:\n",classification_report(val["label"], y_pred_balanced))

"""
We used *class_weight='balanced'* to adjust the contribution to errors more balanced. For class neutral, as it has more rows, thus contributing more to the error while training, meaning one error of class Neutral costs much more then the error of either class. 
So we neutrilize this imbalance by adjusting weights to the contirubtions to the loss:
weight = n_samples / (n_classes × n_samples_in_class)
"""


"""
CREATING A PIPELINE FOR FUTURE USE

sklearn's Pipeline will consist of TF-IDF Vectorizer -> LogisticRegression(balanced) 
"""

# Split the data into (X) and (y)
X1_train, y1_train = train["sentence"], train["label"]
X1_val, y1_val = val["sentence"], val["label"]


#Defining pipeline 
pipe = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('log_regression_balanced', LogisticRegression(class_weight="balanced"))
])

pipe.fit(X1_train, y1_train) 
predictions = pipe.predict(X1_val)


joblib.dump(pipe, DATA_DIR/'pipeline.joblib')











"Putting the data splits into seperate .csv files"
train.to_csv(DATA_DIR / "train.csv", index=False)
val.to_csv(DATA_DIR / "val.csv", index=False)
test.to_csv(DATA_DIR / "test.csv", index=False)