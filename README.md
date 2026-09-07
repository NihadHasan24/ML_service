# Loan Approval Model

This project trains a binary classifier for the dataset's `loan_status` field
(`Approved` or `Rejected`). The saved artifact includes preprocessing and the
random-forest classifier, so it can predict directly from raw feature rows.

## Dataset review

The source file contains 4,269 records, 11 predictive features, one unique
identifier, and one target:

- Approved: 2,656 (62.22%)
- Rejected: 1,613 (37.78%)
- Missing values: none
- Duplicate rows and duplicate loan IDs: none
- Categorical fields: `education` and `self_employed`
- `loan_id` is excluded because it is an identifier, not an applicant feature
- 28 `residential_assets_value` entries are `-100000`; training reports this
  anomaly and retains the values rather than silently changing source data

`cibil_score` is the strongest individual signal in this dataset. `loan_term`
also contributes, while education and self-employment have almost identical
approval rates when examined individually.

## Train

Create the local environment, activate it, and install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python train.py
```

If `.venv` already exists, skip the first command. In VS Code, choose
`.venv\Scripts\python.exe` through **Python: Select Interpreter**.

The training program:

1. Cleans whitespace and validates the input schema and labels.
2. Excludes `loan_id` from training.
3. Makes a stratified 80/20 train/test split.
4. Runs five-fold stratified cross-validation on the training partition.
5. Fits a class-balanced random forest and evaluates the untouched holdout set.
6. Writes the complete pipeline to `models/model.pkl` and its
   detailed results to `models/metrics.json`.

`train.py` is divided into numbered `# %%` cells. Run the cells from top to
bottom, or execute the complete training workflow with `python train.py`.

If imports have yellow underlines, open the Command Palette with
`Ctrl+Shift+P`, choose **Python: Select Interpreter**, and select the same
environment in which you ran `python -m pip install -r requirements.txt`.

## Run the API

Train the model first, then start the FastAPI service:

```powershell
python train.py
uvicorn main:app --reload
```

Open `http://127.0.0.1:8000/docs` for interactive API documentation. The
service provides `GET /health` and `POST /predict` endpoints.

Example prediction request:

```json
{
  "no_of_dependents": 2,
  "education": "Graduate",
  "self_employed": "No",
  "income_annum": 5000000,
  "loan_amount": 12000000,
  "loan_term": 10,
  "cibil_score": 750,
  "residential_assets_value": 5000000,
  "commercial_assets_value": 2000000,
  "luxury_assets_value": 8000000,
  "bank_asset_value": 3000000
}
```

## Test

```powershell
pytest
```

## Use the trained model

Pass a pandas DataFrame with the same 11 feature columns used for training:

```python
import joblib
import pandas as pd

model = joblib.load("models/model.pkl")

applicant = pd.DataFrame(
    [
        {
            "no_of_dependents": 2,
            "education": "Graduate",
            "self_employed": "No",
            "income_annum": 5_000_000,
            "loan_amount": 12_000_000,
            "loan_term": 10,
            "cibil_score": 750,
            "residential_assets_value": 5_000_000,
            "commercial_assets_value": 2_000_000,
            "luxury_assets_value": 8_000_000,
            "bank_asset_value": 3_000_000,
        }
    ]
)

label = model.predict(applicant)[0]
approved_index = list(model.classes_).index("Approved")
approval_probability = model.predict_proba(applicant)[0, approved_index]

print(label, approval_probability)
```

The output probability is a model estimate, not a guarantee. Before using the
model in a real lending workflow, validate it on representative, time-separated
data and perform fairness, calibration, explainability, and regulatory reviews.
