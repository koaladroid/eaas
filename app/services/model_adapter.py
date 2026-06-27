from pathlib import Path

import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

MODEL_FILES = {
    "random_forest": "random_forest.pkl",
    "logistic_regression": "logistic_regression.pkl",
}


class ModelAdapter:
    """Abstracts model loading, prediction, and training data access."""

    def __init__(self, model_name: str):
        if model_name not in MODEL_FILES:
            raise ValueError(f"Unsupported model: {model_name}")
        self.model_name = model_name
        self.model = None
        self.load_model()

    def load_model(self):
        """Load the serialized model from artifacts."""
        model_path = ARTIFACTS_DIR / MODEL_FILES[self.model_name]
        self.model = joblib.load(model_path)
        return self.model

    def _reshape(self, input_data):
        if len(input_data) != 30:
            raise ValueError("Expected 30 features.")
        return np.asarray(input_data, dtype=float).reshape(1, -1)

    def predict(self, input_data):
        """Return the predicted class for the given features."""
        X = self._reshape(input_data)
        return int(self.model.predict(X)[0])

    def predict_proba(self, input_data):
        """Return the probability of the predicted class."""
        X = self._reshape(input_data)
        prediction = int(self.model.predict(X)[0])
        probabilities = self.model.predict_proba(X)[0]
        class_index = np.where(self.model.classes_ == prediction)[0][0]
        return float(probabilities[class_index])

    def get_training_data(self):
        """Load and return the serialized training dataset."""
        return joblib.load(ARTIFACTS_DIR / "training_data.pkl")

    def get_feature_names(self):
        """Return feature names from the training dataset."""
        data = self.get_training_data()
        return data["feature_names"]
