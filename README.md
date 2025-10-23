# Customer Churn Prediction 📉📊

A simple Streamlit web application that predicts whether a customer will churn (leave a service) based on customer data and a pre-trained machine learning model.

Live demo: https://customer-churn-prediction-001.streamlit.app/

---

## Table of Contents

- [Project Overview](#project-overview)
- [Demo](#demo)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Running the App (Local)](#running-the-app-local)
- [Using the App](#using-the-app)
- [Input CSV format & Example](#input-csv-format--example)
- [Model](#model)
- [Retraining the Model (recommended workflow)](#retraining-the-model-recommended-workflow)
- [Evaluation & Metrics](#evaluation--metrics)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License & Contact](#license--contact)

---

## Project Overview

This repository contains a Streamlit application that loads a saved (pre-trained) classification model and uses it to estimate churn probability for customers from a CSV input. It also provides basic visualization of churn-related features so users can explore patterns in the data.

The repository includes:
- A sample dataset (`customer.csv`) to test the app.
- A saved model file (`churn_model.pkl`) used for inference.
- A Streamlit app entrypoint (`main.py`) which handles upload, preprocessing, visualization and prediction.
- `requirements.txt` listing Python dependencies.

---

## Demo

- Live App: https://customer-churn-prediction-001.streamlit.app/

---

## Features

- Upload CSV files with customer records.
- View simple exploratory visualizations (churn counts, distribution of key features).
- Get churn probability and binary churn prediction per customer using a pre-trained model.
- Download results (predictions) from the app.

---

## Tech Stack

- Python 3.10
- Streamlit — UI and app hosting
- pandas — data loading and preprocessing
- scikit-learn — model serialization and (optionally) training
- matplotlib / seaborn — visualizations
- joblib / pickle — model persistence

---

## Project Structure

Customer-Churn-Prediction/
- churn_model.pkl        # Pre-trained ML model used for inference
- customer.csv           # Sample input dataset (example)
- main.py                # Streamlit application code (app entrypoint)
- requirements.txt       # Python dependencies
- README.md              # Project description (this file)

---

## Installation

1. Clone the repository
   ```
   git clone https://github.com/surya01t/Customer-Churn-Prediction.git
   cd Customer-Churn-Prediction
   ```

2. (Recommended) Create a virtual environment and activate it
   ```
   python3 -m venv venv
   source venv/bin/activate   # macOS / Linux
   venv\Scripts\activate      # Windows
   ```

3. Install dependencies
   ```
   pip install -r requirements.txt
   ```

Note: If you run into platform-specific issues, ensure you are using Python 3.8+ (project is tested with Python 3.10).

---

## Running the App (Local)

Start the Streamlit app with:
```
streamlit run main.py
```

This will open a browser tab (or show a local URL) where you can upload `customer.csv` or your own dataset and see predictions and visualizations.

---

## Using the App

1. Open the app URL shown after running the `streamlit run` command.
2. Upload a CSV file containing customer records (see the [Input CSV format](#input-csv-format--example) below).
3. The app will:
   - Display a preview of the uploaded data.
   - Run any preprocessing steps required (e.g., encoding, missing-value handling).
   - Display visualizations (e.g., churn counts, distributions).
   - Display churn probability and predicted churn label for each row.
4. Download results (if the app includes a download button).

---

## Input CSV format & Example

The repository contains a sample `customer.csv`. The exact required columns depend on how `main.py` preprocesses data before feeding to `churn_model.pkl`. If you plan to use your own dataset, make sure to include the same columns used during model training.

Common columns in typical customer churn datasets:
- customerID
- gender
- SeniorCitizen
- Partner
- Dependents
- tenure
- PhoneService
- MultipleLines
- InternetService
- OnlineSecurity
- OnlineBackup
- DeviceProtection
- TechSupport
- StreamingTV
- StreamingMovies
- Contract
- PaperlessBilling
- PaymentMethod
- MonthlyCharges
- TotalCharges
- Churn (optional, for evaluation)

Example (CSV header + one row):
```csv
customerID,gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,Contract,MonthlyCharges,TotalCharges
0001,Male,0,Yes,No,12,Yes,Month-to-month,70.35,845.30
```

If your CSV lacks certain columns the app expects, it may error or produce incorrect predictions — ensure the schema matches.

---

## Model

- churn_model.pkl contains a serialized, pre-trained classifier used for inference in the Streamlit app.
- The original README suggested possible model types such as Random Forest or Logistic Regression. The exact algorithm, hyperparameters and training pipeline are not included in this repository (unless you have additional scripts).
- The app loads `churn_model.pkl` (likely via joblib or pickle) and applies the same preprocessing pipeline expected by that model.

Tip: Inspect `main.py` to see the exact preprocessing steps and the expected columns — adjust your input dataset accordingly.

---

## Retraining the Model (recommended workflow)

If you want to retrain the model yourself (replacing `churn_model.pkl`), follow this general workflow:

1. Prepare a training dataset with the same features as used by the app.
2. Preprocess:
   - Fill or drop missing values.
   - Encode categorical variables (LabelEncoder / OneHotEncoder / OrdinalEncoder or a pipeline).
   - Scale numeric features if model requires it (StandardScaler / MinMaxScaler).
3. Split into train/test (e.g., train_test_split).
4. Train a classifier, e.g.:
   ```python
   from sklearn.ensemble import RandomForestClassifier
   model = RandomForestClassifier(n_estimators=100, random_state=42)
   model.fit(X_train, y_train)
   ```
5. Evaluate on the test set and save the model:
   ```python
   import joblib
   joblib.dump(model, "churn_model.pkl")
   ```
6. Replace the `churn_model.pkl` in the repo (or update app code to point to your new model).

If you'd like, I can provide a training script (train.py) that implements a reproducible pipeline (preprocessing, training, evaluation and model saving). Ask me to create one and I will scaffold it using typical churn dataset steps.

---

## Evaluation & Metrics

For churn prediction tasks, common evaluation metrics include:
- Accuracy
- Precision, Recall, F1-score
- ROC AUC
- Confusion matrix

Because churn datasets are often imbalanced, prefer metrics like ROC AUC and F1-score, and consider using stratified sampling, class weighting, or resampling methods (SMOTE, undersampling) while training.

---

## Troubleshooting

- "Module not found" on running the app: ensure dependencies in `requirements.txt` are installed in the active environment.
- Model load errors: confirm `churn_model.pkl` is present and was serialized with a compatible version of scikit-learn / joblib / pickle.
- Schema mismatch errors: check `main.py` to see which columns and preprocessing are expected and ensure your CSV matches.

---

## Contributing

Contributions are welcome. Some ideas:
- Add a reproducible training script (train.py) with hyperparameter options.
- Add model versioning and model card with details (algorithm, training data, metrics).
- Improve UI and add more visualizations (feature importances, partial dependence plots).
- Add unit tests for data validation and prediction functions.

Please open an issue or a pull request describing your change.

---

## License & Contact

This repository does not include an explicit license file. If you plan to reuse or distribute the code, add a LICENSE file to clarify terms.

Author / Maintainer: repository owner (surya01t)

If you'd like help:
- I can create a training script (train.py) that mirrors the app's preprocessing and saves a model compatible with `main.py`.
- I can inspect `main.py` and `customer.csv` to produce an exact input schema and a validation function for uploads — tell me to proceed and I will generate the files.

---

What I did: I expanded the original short README into a detailed, practical README describing project purpose, installation, usage, model notes, retraining guidance, and troubleshooting. 

What's next: if you want, I can (a) generate a reproducible training script that saves a compatible `churn_model.pkl`, (b) create a data-validation helper to ensure uploaded CSVs match expected columns, or (c) open a short CONTRIBUTING.md and LICENSE file — tell me which and I will create the files.
