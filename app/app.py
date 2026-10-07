# =====================================
# IMPORTS
# =====================================

from flask import Flask
from flask import render_template
from flask import request

from src.inference import predict_top_k_intents


# =====================================
# CREATE FLASK APP
# =====================================

app = Flask(__name__)


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
# RUN FLASK APP
# =====================================

if __name__ == "__main__":

    app.run(
        debug=True
    )