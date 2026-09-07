"""Tests for prediction business logic."""

import pytest

from src.ml_service.schemas import LoanApplication
from src.ml_service.service import LoanApprovalService
from src.ml_service.validation import MODEL_FEATURES

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


class StubModel:
    classes_ = ("Approved", "Rejected")

    def predict(self, features: object) -> list[str]:
        assert hasattr(features, "columns")
        assert list(features.columns) == list(MODEL_FEATURES)
        return ["Approved"]

    def predict_proba(self, features: object) -> list[list[float]]:
        return [[0.91, 0.09]]


class StubLoader:
    def __init__(self) -> None:
        self.model = StubModel()
        self.calls = 0

    def load(self) -> StubModel:
        self.calls += 1
        return self.model


def test_service_returns_label_and_approval_probability() -> None:
    loader = StubLoader()
    service = LoanApprovalService(loader)
    application = LoanApplication(**VALID_APPLICATION)

    result = service.predict(application)

    assert result.loan_status == "Approved"
    assert result.approval_probability == pytest.approx(0.91)
    assert loader.calls == 1
