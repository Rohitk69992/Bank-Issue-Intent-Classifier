# =====================================
# IMPORTS
# =====================================

import csv
import os
import psycopg2

from datetime import datetime


# =====================================
# DATABASE CONNECTION
# =====================================

DATABASE_URL = os.getenv("DATABASE_URL")


def get_connection():

    return psycopg2.connect(
        DATABASE_URL,
        sslmode="require"
    )


# =====================================
# CREATE TABLE
# =====================================

def create_prediction_table():

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute("""

        CREATE TABLE IF NOT EXISTS predictions (

            id SERIAL PRIMARY KEY,

            query TEXT,

            cleaned_query TEXT,

            predicted_intent TEXT,

            confidence_score REAL,

            timestamp TIMESTAMP

        )

    """)

    connection.commit()

    cursor.close()
    connection.close()


# =====================================
# SAVE TO DATABASE
# =====================================

def save_prediction(

    query,

    cleaned_query,

    predicted_intent,

    confidence_score

):

    connection = get_connection()

    cursor = connection.cursor()

    current_timestamp = datetime.now()

    cursor.execute("""

        INSERT INTO predictions (

            query,

            cleaned_query,

            predicted_intent,

            confidence_score,

            timestamp

        )

        VALUES (%s, %s, %s, %s, %s)

    """, (

        query,

        cleaned_query,

        predicted_intent,

        confidence_score,

        current_timestamp

    ))

    connection.commit()

    cursor.close()
    connection.close()


# =====================================
# FETCH ALL PREDICTIONS
# =====================================

def fetch_all_predictions():

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute("""

        SELECT * FROM predictions
        ORDER BY id DESC

    """)

    records = cursor.fetchall()

    cursor.close()
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

        # HEADER

        if not file_exists:

            writer.writerow([

                "query",

                "cleaned_query",

                "predicted_intent",

                "confidence_score",

                "timestamp"

            ])

        # DATA ROW

        writer.writerow([

            query,

            cleaned_query,

            predicted_intent,

            confidence_score,

            str(datetime.now())

        ])