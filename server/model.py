"""ML model loader and inference engine."""

import hashlib
import json
import logging
import os
import pickle
from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

MODEL_DIR = os.environ.get("MODEL_DIR", "./models")


@dataclass
class ModelMetadata:
    name: str
    version: str
    description: str
    input_features: int
    output_classes: list[str]


@dataclass
class InferenceResult:
    predictions: list[float]
    predicted_class: str
    confidence: float


class ModelRegistry:
    def __init__(self):
        self._models: dict[str, object] = {}
        self._scalers: dict[str, StandardScaler] = {}
        self._metadata: dict[str, ModelMetadata] = {}

    def load_or_train(self, model_name: str) -> None:
        os.makedirs(MODEL_DIR, exist_ok=True)
        model_path = os.path.join(MODEL_DIR, f"{model_name}.pkl")
        scaler_path = os.path.join(MODEL_DIR, f"{model_name}_scaler.pkl")

        if os.path.exists(model_path) and os.path.exists(scaler_path):
            with open(model_path, "rb") as f:
                self._models[model_name] = pickle.load(f)
            with open(scaler_path, "rb") as f:
                self._scalers[model_name] = pickle.load(f)
            logger.info("Loaded model '%s' from disk", model_name)
        else:
            self._train_and_save(model_name, model_path, scaler_path)

        self._metadata[model_name] = ModelMetadata(
            name=model_name,
            version="1.0.0",
            description="Iris species classifier (RandomForest)",
            input_features=4,
            output_classes=["setosa", "versicolor", "virginica"],
        )

    def _train_and_save(self, model_name: str, model_path: str, scaler_path: str) -> None:
        logger.info("Training model '%s'...", model_name)
        iris = load_iris()
        X, y = iris.data, iris.target

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_scaled, y)

        with open(model_path, "wb") as f:
            pickle.dump(clf, f)
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)

        self._models[model_name] = clf
        self._scalers[model_name] = scaler
        logger.info("Model '%s' trained and saved", model_name)

    def predict(self, model_name: str, features: list[float]) -> InferenceResult:
        if model_name not in self._models:
            raise ValueError(f"Model '{model_name}' not found")

        model = self._models[model_name]
        scaler = self._scalers[model_name]
        metadata = self._metadata[model_name]

        X = np.array(features).reshape(1, -1)
        X_scaled = scaler.transform(X)

        proba = model.predict_proba(X_scaled)[0]
        class_idx = int(np.argmax(proba))
        confidence = float(proba[class_idx])
        predicted_class = metadata.output_classes[class_idx]

        return InferenceResult(
            predictions=proba.tolist(),
            predicted_class=predicted_class,
            confidence=confidence,
        )

    def get_metadata(self, model_name: str) -> Optional[ModelMetadata]:
        return self._metadata.get(model_name)

    @staticmethod
    def feature_hash(model_name: str, features: list[float]) -> str:
        key = json.dumps({"model": model_name, "features": features}, sort_keys=True)
        return hashlib.sha256(key.encode()).hexdigest()
