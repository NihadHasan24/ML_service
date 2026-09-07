"""Service-specific exceptions."""


class MLServiceError(Exception):
    """Base exception for the loan approval service."""


class InputValidationError(MLServiceError):
    """Raised when model input does not match the expected feature contract."""


class ModelServiceError(MLServiceError):
    """Base exception for model loading and prediction failures."""


class ModelNotFoundError(ModelServiceError):
    """Raised when the trained model artifact is unavailable."""


class ModelLoadError(ModelServiceError):
    """Raised when a model artifact cannot be loaded or is incompatible."""


class PredictionError(ModelServiceError):
    """Raised when the model cannot produce a valid prediction."""
