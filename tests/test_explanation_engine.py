from numbers import Real

import pytest

from app.services.explanation_engine import ExplanationEngine
from app.services.model_adapter import ModelAdapter

SUPPORTED_MODELS = ("random_forest", "logistic_regression")
SUPPORTED_METHODS = ("shap", "lime")


@pytest.fixture(scope="module")
def sample_features():
    training_data = ModelAdapter("random_forest").get_training_data()
    return training_data["X_test"][0]


@pytest.mark.parametrize("model_name", SUPPORTED_MODELS)
@pytest.mark.parametrize("method", SUPPORTED_METHODS)
def test_generate_explanation_returns_complete_response(
    model_name, method, sample_features
):
    result = ExplanationEngine().generate_explanation(
        model=model_name,
        method=method,
        input_data=sample_features,
    )

    assert set(result.keys()) >= {
        "model",
        "method",
        "prediction",
        "probability",
        "feature_importance",
    }
    assert result["model"] == model_name
    assert result["method"] == method
    assert isinstance(result["prediction"], int)
    assert isinstance(result["probability"], Real)
    assert 0.0 <= result["probability"] <= 1.0
    assert isinstance(result["feature_importance"], dict)
    assert len(result["feature_importance"]) == 30
    assert all(isinstance(value, Real) for value in result["feature_importance"].values())


def test_unsupported_explanation_method_propagates_value_error(sample_features):
    with pytest.raises(
        ValueError,
        match=r"Unsupported explanation method: foo\. Supported methods: shap, lime",
    ):
        ExplanationEngine().generate_explanation(
            model="random_forest",
            method="foo",
            input_data=sample_features,
        )
