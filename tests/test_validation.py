"""Tests for request validation and model feature conversion."""

import pytest
from pydantic import ValidationError

from src.ml_service.exceptions import InputValidationError
from src.ml_service.schemas import LoanApplication
from src.ml_service.validation import MODEL_FEATURES, to_model_frame

VALID_APPLICATION = {
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


def test_application_is_converted_in_model_feature_order() -> None:
    application = LoanApplication(**VALID_APPLICATION)

    frame = to_model_frame(application)

    assert list(frame.columns) == list(MODEL_FEATURES)
    assert frame.iloc[0]["cibil_score"] == 750


@pytest.mark.parametrize(
    ("field", "value"),
    [("cibil_score", 250), ("loan_amount", 0), ("bank_asset_value", -1)],
)
def test_schema_rejects_out_of_range_values(field: str, value: int) -> None:
    payload = {**VALID_APPLICATION, field: value}

    with pytest.raises(ValidationError):
        LoanApplication(**payload)


def test_feature_conversion_rejects_missing_fields() -> None:
    payload = VALID_APPLICATION.copy()
    payload.pop("education")

    with pytest.raises(InputValidationError, match="missing=.*education"):
        to_model_frame(payload)
