# Restaurant Rating Predictor (Streamlit + Scikit-Learn)

https://restaurant-rating-predictor-6cvrd5qzrg7xwterz5jfmv.streamlit.app/


This is an end-to-end machine learning project that trains a model to predict a restaurant's **Aggregate rating** using a small set of business-friendly features (cost, price range, and whether it supports delivery or table booking). The trained model is then servedthrough a **Streamlit** web app.

> Note on authorship: I built this as a learning project following a tutorial, and then cleaned it up for reproducibility, GitHub readiness, and clearer documentation.

---

## What does the App do:

You input:
- **Average Cost for two**
- **Has Table booking** (Yes/No)
- **Has Online delivery** (Yes/No)
- **Price range** (1 to 4)

The app outputs:
- A predicted **Aggregate rating** (0 to 5)

---

## Dataset

The dataset contains 9,551 restaurants and 21 columns, including location/cuisine metadata and rating fields.

This project uses these columns:
- `Average Cost for two` (numeric)
- `Has Table booking` (Yes/No -> converted to 1/0)
- `Has Online delivery` (Yes/No -> converted to 1/0)
- `Price range` (1-4)

Target variable:
- `Aggregate rating`

---

## Model approach

High-level pipeline:
1. Select the 4 input features listed above
2. Convert Yes/No fields to 1/0
3. Train/test split
4. Scale features using `StandardScaler`
5. Train a regression model (with hyperparameter search in the notebook)
6. Export artifacts:
   - `artifacts/restaurant_rating_predictor_model.pkl`
   - `artifacts/scaler.pkl`

---

## Repo structure

```
.
├── app.py
├── notebooks/
│   └── Restaurants_Rating_Predictor.ipynb
├── data/
│   └── Dataset.csv
├── artifacts/
│   ├── restaurant_rating_predictor_model.pkl
│   └── scaler.pkl
├── requirements.txt
└── README.md
