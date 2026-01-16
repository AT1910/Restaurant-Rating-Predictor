# Restaurant Rating Predictor (Streamlit + Scikit-Learn)

https://restaurant-rating-predictor-6cvrd5qzrg7xwterz5jfmv.streamlit.app/


This is an end-to-end machine learning project that trains a model to predict a restaurant's **Aggregate rating** using a small set of business-friendly features (cost, price range, and whether it supports delivery or table booking). The trained model is then servedthrough a **Streamlit** web app.

> Note on authorship: I built this as a learning project following a tutorial, and then cleaned it up for reproducibility, GitHub readiness, and clearer documentation.

---

## What the app does

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

## Repo structure (recommended)

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
```

Why this structure:
- GitHub users can run the notebook without editing local file paths.
- Streamlit can reliably find model artifacts in `artifacts/`.

---

## How to run

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Train and export artifacts

Open the notebook and run all cells. Make sure it writes:
- `artifacts/restaurant_rating_predictor_model.pkl`
- `artifacts/scaler.pkl`

### 3) Run the Streamlit app

```bash
streamlit run app.py
```

---

## Common GitHub notebook issue (and the fix)

If your notebook uses a local path like:

```python
df = pd.read_csv('/Users/yourname/Desktop/.../Dataset.csv')
```

It will break for anyone else (and often when rendered on GitHub). Replace it with a repo-relative path:

```python
from pathlib import Path
DATA_PATH = Path('data') / 'Dataset.csv'
df = pd.read_csv(DATA_PATH)
```

---

## Next improvements

If you want this to look more "real" than a tutorial project, these are high-signal upgrades:
- Add proper evaluation (RMSE/MAE) and cross-validation reporting
- Add feature importance (if using a tree model)
- Add input validation + realistic bounds by city/country
- Package training into a script (`train.py`) so artifacts can be rebuilt without a notebook

