# =====================================
# IMPORTS
# =====================================

from flask import Flask
from flask import render_template
from flask import request

from src.inference import predict_top_k_intents

from src.database import (

    create_prediction_table,

    save_prediction,

    fetch_all_predictions,

    save_prediction_to_csv

)


# =====================================
# CREATE FLASK APP
# =====================================

app = Flask(__name__)


# =====================================
# CREATE DATABASE TABLE
# =====================================

create_prediction_table()


# =====================================
# HOME ROUTE
# =====================================

@app.route("/", methods=["GET", "POST"])

def home():

    prediction_result = None

    if request.method == "POST":

        query = request.form.get("query")

        if query:

            prediction_result = predict_top_k_intents(query)

            # ---------------------------------
            # TOP PREDICTION
            # ---------------------------------

            top_prediction = prediction_result[
                "top_predictions"
            ][0]

            top_intent = top_prediction[
                "intent"
            ]

            confidence_score = top_prediction[
                "confidence_score"
            ]

            # ---------------------------------
            # USER DISPLAY INTENT
            # ---------------------------------

            display_intent = (

                top_intent
                .replace("_", " ")
                .title()

            )

            prediction_result[
                "display_intent"
            ] = display_intent

            # ---------------------------------
            # SAVE TO SQLITE DATABASE
            # ---------------------------------

            save_prediction(

                query=query,

                cleaned_query=prediction_result[
                    "cleaned_query"
                ],

                predicted_intent=top_intent,

                confidence_score=confidence_score

            )

            # ---------------------------------
            # SAVE TO CSV
            # ---------------------------------

            save_prediction_to_csv(

                query=query,

                cleaned_query=prediction_result[
                    "cleaned_query"
                ],

                predicted_intent=top_intent,

                confidence_score=confidence_score

            )

            # ---------------------------------
            # TERMINAL LOGGING
            # ---------------------------------

            print("\n=====================")
            print(prediction_result)
            print("=====================\n")

    return render_template(

        "index.html",

        result=prediction_result

    )


# =====================================
# HISTORY ROUTE
# =====================================

@app.route("/history")

def history():

    records = fetch_all_predictions()

    return render_template(

        "history.html",

        records=records

    )


# =====================================
# RUN FLASK APP
# =====================================

if __name__ == "__main__":

    app.run(
        debug=True
    )