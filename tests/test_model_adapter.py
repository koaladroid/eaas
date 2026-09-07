import pytest

from app.services.model_adapter import ModelAdapter

SUPPORTED_MODELS = ("random_forest", "logistic_regression")


@pytest.fixture(scope="module", params=SUPPORTED_MODELS)
def adapter(request):
    return ModelAdapter(request.param)


@pytest.fixture(scope="module")
def sample_features():
    training_data = ModelAdapter("random_forest").get_training_data()
    return training_data["X_test"][0]


def test_random_forest_can_be_instantiated_and_loaded():
    adapter = ModelAdapter("random_forest")
    assert adapter.model is not None
    assert adapter.model_name == "random_forest"


def test_logistic_regression_can_be_instantiated_and_loaded():
    adapter = ModelAdapter("logistic_regression")
    assert adapter.model is not None
    assert adapter.model_name == "logistic_regression"


def test_supported_models_produce_a_prediction(adapter, sample_features):
    prediction = adapter.predict(sample_features)
    assert isinstance(prediction, int)


def test_supported_models_produce_a_prediction_probability(adapter, sample_features):
    probability = adapter.predict_proba(sample_features)
    assert isinstance(probability, float)
    assert 0.0 <= probability <= 1.0


def test_supported_models_expose_training_data(adapter):
    training_data = adapter.get_training_data()
    assert "X_train" in training_data
    assert training_data["X_train"].shape[1] == 30


def test_supported_models_expose_feature_names(adapter):
    feature_names = adapter.get_feature_names()
    assert len(feature_names) == 30


def test_invalid_input_length_is_rejected(adapter):
    with pytest.raises(ValueError, match="Expected 30 features."):
        adapter.predict([0.0] * 29)
