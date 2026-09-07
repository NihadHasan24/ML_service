"""Request and response schemas for the prediction API."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class LoanApplication(BaseModel):
    """Validated applicant features expected by the trained model."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
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
        },
    )

    no_of_dependents: int = Field(ge=0, le=20)
    education: Literal["Graduate", "Not Graduate"]
    self_employed: Literal["Yes", "No"]
    income_annum: int = Field(gt=0)
    loan_amount: int = Field(gt=0)
    loan_term: int = Field(gt=0, le=100)
    cibil_score: int = Field(ge=300, le=900)
    residential_assets_value: int = Field(ge=0)
    commercial_assets_value: int = Field(ge=0)
    luxury_assets_value: int = Field(ge=0)
    bank_asset_value: int = Field(ge=0)


class LoanPrediction(BaseModel):
    """Prediction returned by the service."""

    loan_status: Literal["Approved", "Rejected"]
    approval_probability: float = Field(ge=0.0, le=1.0)


class HealthResponse(BaseModel):
    """Health-check response."""

    status: Literal["ok"]
    model_loaded: bool
