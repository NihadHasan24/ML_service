"""Lazy and thread-safe loading of the trained model artifact."""

from pathlib import Path
from threading import Lock
from typing import Any

import joblib

from .exceptions import ModelLoadError, ModelNotFoundError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "model.pkl"
REQUIRED_MODEL_METHODS = ("predict", "predict_proba")


class ModelLoader:
    """Load the model once and reuse it for subsequent predictions."""

    def __init__(self, model_path: Path | str = DEFAULT_MODEL_PATH) -> None:
        self.model_path = Path(model_path)
        self._model: Any | None = None
        self._lock = Lock()

    def load(self) -> Any:
        """Return the cached model, loading and validating it when necessary."""

        if self._model is None:
            with self._lock:
                if self._model is None:
                    self._model = self._load_from_disk()
        return self._model

    def clear(self) -> None:
        """Remove the in-memory model so the next call reloads it."""

        with self._lock:
            self._model = None

    def _load_from_disk(self) -> Any:
        if not self.model_path.is_file():
            raise ModelNotFoundError(
                f"Model not found at {self.model_path}. Run train.py first."
            )

        try:
            model = joblib.load(self.model_path)
        except Exception as exc:
            raise ModelLoadError(
                f"Unable to load model artifact: {self.model_path}"
            ) from exc

        missing_attributes = [
            name
            for name in (*REQUIRED_MODEL_METHODS, "classes_")
            if not hasattr(model, name)
        ]
        if missing_attributes:
            raise ModelLoadError(
                f"Model is missing required attributes: {missing_attributes}"
            )

        classes = {str(label) for label in model.classes_}
        expected_classes = {"Approved", "Rejected"}
        if classes != expected_classes:
            raise ModelLoadError(
                f"Expected model classes {sorted(expected_classes)}; found {sorted(classes)}"
            )
        return model
