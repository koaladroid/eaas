import numpy as np
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


class MethodManager:
    """Selects and executes explanation methods behind a common interface."""

    def explain(self, method, model_adapter, input_data, prediction):
        if method == "shap":
            return self._run_shap(model_adapter, input_data, prediction)
        raise ValueError(f"Unsupported explanation method: {method}")

    def _run_shap(self, model_adapter, input_data, prediction):
        model = model_adapter.model
        feature_names = model_adapter.get_feature_names()
        training_data = model_adapter.get_training_data()
        X = np.asarray(input_data, dtype=float).reshape(1, -1)

        if isinstance(model, RandomForestClassifier):
            explainer = shap.TreeExplainer(model)
        elif isinstance(model, LogisticRegression):
            explainer = shap.LinearExplainer(model, training_data["X_train"])
        else:
            explainer = shap.Explainer(model, training_data["X_train"])

        shap_values = explainer.shap_values(X)
        values = self._normalize_shap_values(shap_values, prediction)

        feature_importance = {
            name: float(value)
            for name, value in zip(feature_names, values)
        }
        feature_importance = dict(
            sorted(
                feature_importance.items(),
                key=lambda item: abs(item[1]),
                reverse=True,
            )
        )

        return {
            "method": "shap",
            "feature_importance": feature_importance,
        }

    def _normalize_shap_values(self, shap_values, prediction):
        if isinstance(shap_values, list):
            return shap_values[int(prediction)][0]

        arr = np.asarray(shap_values)
        if arr.ndim == 3:
            return arr[0, :, int(prediction)]
        return arr[0]
