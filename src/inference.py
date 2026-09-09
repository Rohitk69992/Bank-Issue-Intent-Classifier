# =====================================
# IMPORTS
# =====================================

import json
import joblib
import numpy as np

from src.preprocessing import clean_text

from src.config import (
    MODEL_PATH,
    VECTORIZER_PATH,
    LABEL_MAPPING_PATH,
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

with open(
    LABEL_MAPPING_PATH,
    "r",
    encoding="utf-8"
) as f:

    label_mapping = json.load(f)


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

        index = int(index)

        predictions.append({

            "label": index,

            "intent": label_mapping[str(index)],

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