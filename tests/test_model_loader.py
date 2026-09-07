"""Tests for model artifact loading and caching."""

from pathlib import Path

import joblib
import pytest

from src.ml_service.exceptions import ModelLoadError, ModelNotFoundError
from src.ml_service.model_loader import ModelLoader


class CompatibleModel:
    classes_ = ("Approved", "Rejected")

    def predict(self, features: object) -> list[str]:
        return ["Approved"]

    def predict_proba(self, features: object) -> list[list[float]]:
        return [[0.8, 0.2]]


def test_loader_loads_and_caches_compatible_model(tmp_path: Path) -> None:
    model_path = tmp_path / "model.pkl"
    joblib.dump(CompatibleModel(), model_path)
    loader = ModelLoader(model_path)

    first = loader.load()
    second = loader.load()

    assert first is second
    assert first.classes_ == ("Approved", "Rejected")


def test_loader_reports_missing_model(tmp_path: Path) -> None:
    loader = ModelLoader(tmp_path / "missing.pkl")

    with pytest.raises(ModelNotFoundError, match="Run train.py first"):
        loader.load()


def test_loader_rejects_incompatible_artifact(tmp_path: Path) -> None:
    model_path = tmp_path / "model.pkl"
    joblib.dump({"not": "a model"}, model_path)

    with pytest.raises(ModelLoadError, match="required attributes"):
        ModelLoader(model_path).load()
