"""Business logic for loan approval predictions."""

from typing import Any

from .exceptions import MLServiceError, PredictionError
from .model_loader import ModelLoader
from .schemas import LoanApplication, LoanPrediction
from .validation import to_model_frame


class LoanApprovalService:
    """Coordinate feature validation, model loading, and prediction."""

    def __init__(self, model_loader: ModelLoader | Any | None = None) -> None:
        self.model_loader = model_loader or ModelLoader()

    def ensure_model_loaded(self) -> None:
        """Load the artifact, raising a service exception if it is unavailable."""

        self.model_loader.load()

    def predict(self, application: LoanApplication) -> LoanPrediction:
        """Predict the status and approval probability for one application."""

        try:
            model = self.model_loader.load()
            features = to_model_frame(application)
            predicted_status = str(model.predict(features)[0])
            classes = [str(label) for label in model.classes_]
            approved_index = classes.index("Approved")
            approval_probability = float(
                model.predict_proba(features)[0][approved_index]
            )
        except MLServiceError:
            raise
        except Exception as exc:
            raise PredictionError("The model could not produce a prediction.") from exc

        if predicted_status not in {"Approved", "Rejected"}:
            raise PredictionError(
                f"Model returned an unexpected class: {predicted_status}"
            )
        if not 0.0 <= approval_probability <= 1.0:
            raise PredictionError("Model returned an invalid approval probability.")

        return LoanPrediction(
            loan_status=predicted_status,
            approval_probability=approval_probability,
        )
