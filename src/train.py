# =====================================
# IMPORTS
# =====================================

import os
import joblib
import pandas as pd

from datasets import load_dataset

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

from src.preprocessing import clean_text

from src.config import (

    MODEL_PATH,

    VECTORIZER_PATH

)


# =====================================
# LOAD DATASET
# =====================================

dataset = load_dataset(
    "mteb/banking77"
)

train_df = pd.DataFrame(
    dataset["train"]
)

test_df = pd.DataFrame(
    dataset["test"]
)


# =====================================
# PREPROCESSING
# =====================================

train_df["clean_text"] = train_df[
    "text"
].apply(clean_text)

test_df["clean_text"] = test_df[
    "text"
].apply(clean_text)


# =====================================
# TF-IDF
# =====================================

vectorizer = TfidfVectorizer(

    max_features=5000,

    ngram_range=(1, 2),

    min_df=2

)

X_train = vectorizer.fit_transform(
    train_df["clean_text"]
)

X_test = vectorizer.transform(
    test_df["clean_text"]
)


# =====================================
# TARGETS
# =====================================

y_train = train_df["label"]

y_test = test_df["label"]


# =====================================
# MODEL TRAINING
# =====================================

model = LogisticRegression(

    max_iter=1000,

    random_state=42

)

model.fit(
    X_train,
    y_train
)


# =====================================
# EVALUATION
# =====================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print(f"\nAccuracy: {accuracy:.4f}")


# =====================================
# SAVE MODEL + VECTORIZER
# =====================================

os.makedirs(
    "models",
    exist_ok=True
)

joblib.dump(
    model,
    MODEL_PATH
)

joblib.dump(
    vectorizer,
    VECTORIZER_PATH
)

print(
    "\nModel + Vectorizer saved successfully."
)