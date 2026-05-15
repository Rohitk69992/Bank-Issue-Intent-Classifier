# =====================================
# IMPORTS
# =====================================

import joblib
import numpy as np
import pandas as pd

from datasets import load_dataset

from src.preprocessing import clean_text

from src.config import (

    MODEL_PATH,

    VECTORIZER_PATH,

    TOP_K_PREDICTIONS

)


# =====================================
# LOAD MODEL + VECTORIZER
# =====================================

model = joblib.load(
    MODEL_PATH
)

vectorizer = joblib.load(
    VECTORIZER_PATH
)


# =====================================
# LOAD LABEL MAPPING
# =====================================

dataset = load_dataset(
    "mteb/banking77"
)

train_df = pd.DataFrame(
    dataset["train"]
)

label_mapping = dict(

    zip(

        train_df["label"],

        train_df["label_text"]

    )
)


# =====================================
# PREDICTION FUNCTION
# =====================================

def predict_top_k_intents(

    text,

    k=TOP_K_PREDICTIONS

):

    cleaned_text = clean_text(text)

    vectorized_text = vectorizer.transform(
        [cleaned_text]
    )

    probabilities = model.predict_proba(
        vectorized_text
    )[0]

    top_k_indices = np.argsort(
        probabilities
    )[::-1][:k]

    predictions = []

    for index in top_k_indices:

        predictions.append({

            "label": int(index),

            "intent": label_mapping[index],

            "confidence_score": round(
                float(probabilities[index]),
                4
            )

        })

    return {

        "query": text,

        "cleaned_query": cleaned_text,

        "top_predictions": predictions

    }