# =====================================
# IMPORTS
# =====================================

import csv
import os
import sqlite3

from datetime import datetime


# =====================================
# DATABASE PATH
# =====================================

DATABASE_PATH = "database/predictions.db"


# =====================================
# CREATE TABLE
# =====================================

def create_prediction_table():

    os.makedirs(
        "database",
        exist_ok=True
    )

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    cursor = connection.cursor()

    cursor.execute("""

        CREATE TABLE IF NOT EXISTS predictions (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            query TEXT,

            cleaned_query TEXT,

            predicted_intent TEXT,

            confidence_score REAL,

            timestamp TEXT

        )

    """)

    connection.commit()

    connection.close()


# =====================================
# SAVE TO SQLITE
# =====================================

def save_prediction(

    query,

    cleaned_query,

    predicted_intent,

    confidence_score

):

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    cursor = connection.cursor()

    current_timestamp = str(
        datetime.now()
    )

    cursor.execute("""

        INSERT INTO predictions (

            query,

            cleaned_query,

            predicted_intent,

            confidence_score,

            timestamp

        )

        VALUES (?, ?, ?, ?, ?)

    """, (

        query,

        cleaned_query,

        predicted_intent,

        confidence_score,

        current_timestamp

    ))

    connection.commit()

    connection.close()


# =====================================
# FETCH ALL PREDICTIONS
# =====================================

def fetch_all_predictions():

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    cursor = connection.cursor()

    cursor.execute("""

        SELECT * FROM predictions

    """)

    records = cursor.fetchall()

    connection.close()

    return records


# =====================================
# SAVE TO CSV
# =====================================

def save_prediction_to_csv(

    query,

    cleaned_query,

    predicted_intent,

    confidence_score

):

    os.makedirs(
        "artifacts",
        exist_ok=True
    )

    csv_path = "artifacts/predictions.csv"

    file_exists = os.path.isfile(
        csv_path
    )

    with open(

        csv_path,

        mode="a",

        newline="",

        encoding="utf-8"

    ) as file:

        writer = csv.writer(file)

        # ----------------------------
        # HEADER
        # ----------------------------

        if not file_exists:

            writer.writerow([

                "query",

                "cleaned_query",

                "predicted_intent",

                "confidence_score",

                "timestamp"

            ])

        # ----------------------------
        # DATA ROW
        # ----------------------------

        writer.writerow([

            query,

            cleaned_query,

            predicted_intent,

            confidence_score,

            str(datetime.now())

        ])