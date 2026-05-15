# Bank Issue Intent Classifier

An end-to-end NLP-powered banking issue detection system built using TF-IDF and Logistic Regression. The application classifies customer banking queries into predefined intent categories through a Flask web application and stores predictions for monitoring and analysis.

---

# Project Overview

This project focuses on practical Natural Language Processing (NLP) workflow implementation for banking customer support automation.

The system:

* Accepts user banking queries through a web interface
* Cleans and preprocesses text data
* Converts text into numerical vectors using TF-IDF
* Predicts the most likely banking intent using Logistic Regression
* Displays predictions through a Flask frontend
* Stores prediction history in SQLite database and CSV logs
* Can be deployed publicly using Render

---

# Features

* Real-time banking issue classification
* TF-IDF feature engineering
* Logistic Regression intent prediction
* Flask-based frontend and backend
* SQLite database logging
* CSV prediction history export
* Modular project structure
* Deployment-ready architecture
* Top-K prediction confidence scores

---

# Example Queries

| User Query                          | Predicted Intent                   |
| ----------------------------------- | ---------------------------------- |
| my card payment failed online       | declined_card_payment              |
| cash withdrawal is still pending    | pending_cash_withdrawal            |
| i did not receive the bank transfer | transfer_not_received_by_recipient |
| my account verification failed      | unable_to_verify_identity          |

---

# Tech Stack

## Machine Learning

* Scikit-learn
* TF-IDF Vectorizer
* Logistic Regression

## Backend

* Flask
* SQLite

## Frontend

* HTML
* CSS

## Data Handling

* Pandas
* NumPy

## Deployment

* Git
* GitHub
* Render
* Gunicorn

---

# Dataset

Dataset used:

* Banking77 Dataset
* Source: Hugging Face Datasets

The dataset contains customer banking support queries mapped to multiple banking intent categories.

---

# Project Structure

```bash
Bank-Issue-Intent-Classifier/
│
├── app/
│   ├── static/
│   │   └── style.css
│   │
│   ├── templates/
│   │   ├── index.html
│   │   └── history.html
│   │
│   └── app.py
│
├── artifacts/
│   └── predictions.csv
│
├── database/
│   └── predictions.db
│
├── models/
│   ├── logistic_regression.pkl
│   ├── tfidf_vectorizer.pkl
│   └── label_mapping.pkl
│
├── src/
│   ├── config.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── inference.py
│   └── database.py
│
├── requirements.txt
├── runtime.txt
├── Procfile
├── .gitignore
└── README.md
```

---

# Machine Learning Pipeline

## 1. Text Preprocessing

The preprocessing pipeline performs:

* Lowercasing
* Whitespace normalization
* Text cleaning
* Regex-based normalization

---

## 2. Feature Engineering

TF-IDF Vectorizer converts cleaned text into sparse numerical vectors.

### Parameters Used

```python
TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2
)
```

---

## 3. Model Training

The classification model uses:

```python
LogisticRegression(
    max_iter=1000,
    random_state=42
)
```

---

## 4. Inference Pipeline

The deployed inference system:

* Cleans incoming user query
* Vectorizes text using saved TF-IDF vectorizer
* Predicts probabilities
* Returns Top-K intent predictions
* Stores predictions in SQLite + CSV

---

# Model Performance

## Accuracy

```text
Accuracy: ~85.9%
```

The model performs strongly on intent classification using classical NLP techniques.

---

# Installation

## Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/Bank-Issue-Intent-Classifier.git
```

---

## Move Into Project

```bash
cd Bank-Issue-Intent-Classifier
```

---

## Create Virtual Environment

### Windows

```bash
python -m venv venv
```

---

## Activate Virtual Environment

### Windows PowerShell

```bash
venv\Scripts\activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Train Model

```bash
python -m src.train
```

This generates:

* trained model
* TF-IDF vectorizer
* label mapping

inside the `models/` folder.

---

# Run Flask Application

```bash
python -m app.app
```

Application runs on:

```text
http://127.0.0.1:5000
```

---

# Deployment

The project can be deployed publicly using Render.

## Deployment Stack

```text
GitHub → Render → Public URL
```

### Required Deployment Files

* requirements.txt
* runtime.txt
* Procfile

---

# Database Logging

Each prediction stores:

* user query
* cleaned query
* predicted intent
* confidence score
* timestamp

Stored in:

* SQLite database
* CSV prediction logs

---

# Future Improvements

Potential upgrades:

* Transformer-based models (BERT, DistilBERT)
* FastAPI backend
* Docker containerization
* User authentication
* Admin dashboard
* Confidence thresholding
* Real-time analytics
* REST API endpoints
* Kubernetes deployment

---

# Learning Outcomes

This project demonstrates:

* End-to-end NLP workflow
* Classical machine learning for text classification
* Modular Python project structuring
* Flask backend development
* ML inference engineering
* Model serialization
* Database integration
* Cloud deployment workflow
* Git and GitHub integration

---


