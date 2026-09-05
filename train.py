"""Train a model that predicts whether a loan is Approved or Rejected.

VS Code recognizes each ``# %%`` section as a runnable Python cell. Run the
cells from top to bottom, or run the entire file with ``python train.py``.
"""

# %% 1. Import libraries
import json
import warnings
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# %% 2. Project settings and dataset schema
script_path = globals().get("__file__")
PROJECT_ROOT = Path(script_path).resolve().parent if script_path else Path.cwd()
DATA_PATH = PROJECT_ROOT / "dataset" / "loan_approval_dataset.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "loan_approval_model.joblib"
METRICS_PATH = PROJECT_ROOT / "models" / "metrics.json"
PIPELINE_CACHE = PROJECT_ROOT / ".pipeline_cache"

TEST_SIZE = 0.20
CV_FOLDS = 5
RANDOM_STATE = 42

TARGET_COLUMN = "loan_status"
ID_COLUMN = "loan_id"
LABELS = ["Rejected", "Approved"]

NUMERIC_FEATURES = [
    "no_of_dependents",
    "income_annum",
    "loan_amount",
    "loan_term",
    "cibil_score",
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value",
]
CATEGORICAL_FEATURES = ["education", "self_employed"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
ASSET_FEATURES = [
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value",
]


# %% 3. Load and clean the dataset
if not DATA_PATH.is_file():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

# skipinitialspace removes the spaces that appear after commas in the CSV.
data = pd.read_csv(DATA_PATH, skipinitialspace=True)
data.columns = data.columns.str.strip()

required_columns = set(FEATURES + [ID_COLUMN, TARGET_COLUMN])
missing_columns = sorted(required_columns.difference(data.columns))
if missing_columns:
    raise ValueError(f"Dataset is missing required columns: {missing_columns}")

# Keep only the columns used by this project and normalize their values.
data = data[FEATURES + [ID_COLUMN, TARGET_COLUMN]].copy()

for column in CATEGORICAL_FEATURES + [TARGET_COLUMN]:
    data[column] = data[column].astype("string").str.strip()

for column in NUMERIC_FEATURES + [ID_COLUMN]:
    data[column] = pd.to_numeric(data[column], errors="raise")

print(f"Dataset loaded: {data.shape[0]} rows and {data.shape[1]} columns")
print(data.head())


# %% 4. Validate and understand the data
if data.empty:
    raise ValueError("Dataset contains no rows.")
if data[ID_COLUMN].isna().any():
    raise ValueError(f"{ID_COLUMN} contains missing values.")
if data[ID_COLUMN].duplicated().any():
    raise ValueError(f"{ID_COLUMN} must be unique.")
if data[FEATURES + [TARGET_COLUMN]].isna().any().any():
    missing_values = data[FEATURES + [TARGET_COLUMN]].isna().sum()
    missing_values = missing_values[missing_values.gt(0)].to_dict()
    raise ValueError(f"Dataset contains missing values: {missing_values}")

observed_labels = set(data[TARGET_COLUMN].unique())
if observed_labels != set(LABELS):
    raise ValueError(
        f"Expected target labels {LABELS}; found {sorted(observed_labels)}"
    )

allowed_categories = {
    "education": {"Graduate", "Not Graduate"},
    "self_employed": {"Yes", "No"},
}
for column, allowed_values in allowed_categories.items():
    unexpected_values = sorted(set(data[column].unique()).difference(allowed_values))
    if unexpected_values:
        raise ValueError(f"Unexpected values in {column}: {unexpected_values}")

negative_assets = {
    column: int(data[column].lt(0).sum())
    for column in ASSET_FEATURES
    if data[column].lt(0).any()
}
if negative_assets:
    warnings.warn(
        "Negative asset values were found and retained so the source data is not "
        f"silently changed: {negative_assets}",
        stacklevel=2,
    )

target_counts = data[TARGET_COLUMN].value_counts()
print("\nMissing values:")
print(data.isna().sum())
print(f"\nDuplicate rows: {data.duplicated().sum()}")
print("\nTarget counts:")
print(target_counts)
print("\nTarget percentages:")
print((target_counts / len(data) * 100).round(2))
print("\nNumeric summary:")
print(data[NUMERIC_FEATURES].describe().T)


# %% 5. Separate features and target, then create the holdout set
# loan_id is deliberately excluded because it identifies a row rather than
# describing an applicant.
X = data[FEATURES]
y = data[TARGET_COLUMN]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

print(f"Training rows: {len(X_train)}")
print(f"Holdout test rows: {len(X_test)}")


# %% 6. Build preprocessing steps
# Imputation makes the saved pipeline robust to missing values in future input.
numeric_pipeline = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median"))]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
        ),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ]
)


# %% 7. Create the random-forest model and complete pipeline
classifier = RandomForestClassifier(
    n_estimators=400,
    min_samples_leaf=2,
    max_features="sqrt",
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=1,
)

# Keeping preprocessing and classification together prevents training-serving
# skew and lets the saved model accept raw pandas rows.
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ],
    memory=str(PIPELINE_CACHE),
)


# %% 8. Cross-validate only on the training partition
cross_validator = StratifiedKFold(
    n_splits=CV_FOLDS,
    shuffle=True,
    random_state=RANDOM_STATE,
)

cv_scores = cross_validate(
    model,
    X_train,
    y_train,
    cv=cross_validator,
    scoring={
        "accuracy": "accuracy",
        "macro_f1": "f1_macro",
        "roc_auc": "roc_auc",
    },
    n_jobs=1,
)

print(f"{CV_FOLDS}-fold cross-validation results:")
for metric_name in ("accuracy", "macro_f1", "roc_auc"):
    values = cv_scores[f"test_{metric_name}"]
    print(f"  {metric_name}: {values.mean():.4f} (+/- {values.std():.4f})")


# %% 9. Fit the model and evaluate the untouched holdout set
model.fit(X_train, y_train)
fitted_classifier = model.named_steps["classifier"]

predictions = model.predict(X_test)
class_names = list(model.classes_)
approved_index = class_names.index("Approved")
approval_probabilities = model.predict_proba(X_test)[:, approved_index]
binary_test_target = y_test.eq("Approved").astype(int)

accuracy = accuracy_score(y_test, predictions)
balanced_accuracy = balanced_accuracy_score(y_test, predictions)
macro_f1 = f1_score(y_test, predictions, average="macro")
roc_auc = roc_auc_score(binary_test_target, approval_probabilities)
confusion = confusion_matrix(y_test, predictions, labels=LABELS)
report = classification_report(
    y_test,
    predictions,
    labels=LABELS,
    output_dict=True,
    zero_division=0,
)

print(f"Holdout accuracy:          {accuracy:.4f}")
print(f"Holdout balanced accuracy: {balanced_accuracy:.4f}")
print(f"Holdout macro-F1:          {macro_f1:.4f}")
print(f"Holdout ROC-AUC:           {roc_auc:.4f}")
print(f"\nConfusion matrix (labels: {LABELS}):")
print(confusion)
print("\nClassification report:")
print(
    classification_report(
        y_test,
        predictions,
        labels=LABELS,
        digits=4,
        zero_division=0,
    )
)


# %% 10. Inspect which features the model uses most
transformed_features = model.named_steps["preprocessor"].get_feature_names_out()
feature_importance = sorted(
    zip(transformed_features, fitted_classifier.feature_importances_, strict=True),
    key=lambda item: item[1],
    reverse=True,
)

print("Top feature importance:")
for feature_name, importance in feature_importance[:10]:
    print(f"  {feature_name}: {importance:.4f}")


# %% 11. Save the trained pipeline and its metrics
cross_validation_metrics = {
    "folds": CV_FOLDS,
    **{
        metric_name: {
            "mean": float(cv_scores[f"test_{metric_name}"].mean()),
            "standard_deviation": float(
                cv_scores[f"test_{metric_name}"].std()
            ),
        }
        for metric_name in ("accuracy", "macro_f1", "roc_auc")
    },
}

metrics = {
    "data": {
        "rows": len(data),
        "input_features": len(FEATURES),
        "duplicate_rows": int(data.duplicated().sum()),
        "target_distribution": {
            str(label): int(count) for label, count in target_counts.items()
        },
        "negative_asset_values": negative_assets,
    },
    "split": {
        "training_rows": len(X_train),
        "test_rows": len(X_test),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
    },
    "model": {
        "type": "RandomForestClassifier",
        "parameters": {
            "n_estimators": fitted_classifier.n_estimators,
            "min_samples_leaf": fitted_classifier.min_samples_leaf,
            "max_features": fitted_classifier.max_features,
            "class_weight": fitted_classifier.class_weight,
        },
        "features": FEATURES,
        "target": TARGET_COLUMN,
        "classes": class_names,
    },
    "cross_validation_on_training_partition": cross_validation_metrics,
    "holdout_test": {
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "macro_f1": float(macro_f1),
        "roc_auc": float(roc_auc),
        "confusion_matrix": {
            "label_order": LABELS,
            "values": confusion.tolist(),
        },
        "classification_report": report,
    },
    "top_feature_importance": [
        {"feature": str(feature), "importance": float(importance)}
        for feature, importance in feature_importance[:10]
    ],
}

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(model, MODEL_PATH)
METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

print(f"Model saved to: {MODEL_PATH}")
print(f"Metrics saved to: {METRICS_PATH}")
