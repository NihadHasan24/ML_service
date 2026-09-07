"""Validation and conversion helpers for raw model features."""

from collections.abc import Mapping
from math import isfinite
from typing import Any

import pandas as pd

from .exceptions import InputValidationError
from .schemas import LoanApplication

MODEL_FEATURES = (
    "no_of_dependents",
    "income_annum",
    "loan_amount",
    "loan_term",
    "cibil_score",
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value",
    "education",
    "self_employed",
)
NUMERIC_FEATURES = MODEL_FEATURES[:9]


def validate_feature_payload(payload: Mapping[str, Any]) -> None:
    """Validate exact feature names and finite numeric values."""

    supplied = set(payload)
    expected = set(MODEL_FEATURES)
    missing = sorted(expected - supplied)
    unexpected = sorted(supplied - expected)
    if missing or unexpected:
        raise InputValidationError(
            f"Invalid feature set; missing={missing}, unexpected={unexpected}"
        )

    non_finite = [
        name
        for name in NUMERIC_FEATURES
        if isinstance(payload[name], (int, float))
        and not isinstance(payload[name], bool)
        and not isfinite(payload[name])
    ]
    if non_finite:
        raise InputValidationError(f"Non-finite numeric values: {non_finite}")


def to_model_frame(
    application: LoanApplication | Mapping[str, Any],
) -> pd.DataFrame:
    """Convert one validated application into the model's ordered DataFrame."""

    payload = (
        application.model_dump()
        if isinstance(application, LoanApplication)
        else dict(application)
    )
    validate_feature_payload(payload)
    return pd.DataFrame([payload], columns=list(MODEL_FEATURES))
