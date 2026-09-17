from numbers import Real

import pytest

from app.services.method_manager import MethodManager
from app.services.model_adapter import ModelAdapter

SUPPORTED_MODELS = ("random_forest", "logistic_regression")
SUPPORTED_METHODS = ("shap", "lime")


@pytest.fixture(scope="module")
def sample_features():
    training_data = ModelAdapter("random_forest").get_training_data()
    return training_data["X_test"][0]


@pytest.mark.parametrize("method", SUPPORTED_METHODS)
def test_method_manager_accepts_supported_explanation_methods(method, sample_features):
    adapter = ModelAdapter("random_forest")
    prediction = adapter.predict(sample_features)
    result = MethodManager().explain(
        method=method,
        model_adapter=adapter,
        input_data=sample_features,
        prediction=prediction,
    )
    assert result["method"] == method


@pytest.mark.parametrize("model_name", SUPPORTED_MODELS)
@pytest.mark.parametrize("method", SUPPORTED_METHODS)
def test_explain_returns_standardized_result_for_supported_combinations(
    model_name, method, sample_features
):
    adapter = ModelAdapter(model_name)
    prediction = adapter.predict(sample_features)
    result = MethodManager().explain(
        method=method,
        model_adapter=adapter,
        input_data=sample_features,
        prediction=prediction,
    )

    assert "method" in result
    assert "feature_importance" in result
    assert result["method"] == method
    assert isinstance(result["feature_importance"], dict)
    assert len(result["feature_importance"]) == 30
    assert all(isinstance(value, Real) for value in result["feature_importance"].values())


def test_unsupported_explanation_method_is_rejected(sample_features):
    adapter = ModelAdapter("random_forest")
    prediction = adapter.predict(sample_features)
    with pytest.raises(
        ValueError,
        match=r"Unsupported explanation method: foo\. Supported methods: shap, lime",
    ):
        MethodManager().explain(
            method="foo",
            model_adapter=adapter,
            input_data=sample_features,
            prediction=prediction,
        )
