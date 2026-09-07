"""FastAPI entry point for the loan approval service."""

from fastapi import FastAPI, HTTPException, status

from src.ml_service.exceptions import InputValidationError, ModelServiceError
from src.ml_service.schemas import HealthResponse, LoanApplication, LoanPrediction
from src.ml_service.service import LoanApprovalService

app = FastAPI(
    title="Loan Approval API",
    description="Predict loan approval with the trained random-forest pipeline.",
    version="1.0.0",
)
loan_service = LoanApprovalService()


@app.get("/", tags=["service"])
def root() -> dict[str, str]:
    return {"message": "Loan Approval API", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse, tags=["service"])
def health() -> HealthResponse:
    try:
        loan_service.ensure_model_loaded()
    except ModelServiceError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    return HealthResponse(status="ok", model_loaded=True)


@app.post("/predict", response_model=LoanPrediction, tags=["prediction"])
def predict(application: LoanApplication) -> LoanPrediction:
    try:
        return loan_service.predict(application)
    except InputValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc
    except ModelServiceError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
