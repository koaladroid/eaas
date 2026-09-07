from numbers import Real

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.model_adapter import ModelAdapter

SUPPORTED_MODELS = ("random_forest", "logistic_regression")
SUPPORTED_METHODS = ("shap", "lime")
EXPECTED_KEYS = {
    "model",
    "method",
    "prediction",
    "probability",
    "feature_importance",
}


@pytest.fixture(scope="module")
def sample_features():
    training_data = ModelAdapter("random_forest").get_training_data()
    return training_data["X_test"][0].tolist()


def _error_payload(response):
    body = response.json()
    assert "error" in body
    assert "type" in body["error"]
    assert "message" in body["error"]
    return body


@pytest.mark.parametrize("model_name", SUPPORTED_MODELS)
@pytest.mark.parametrize("method", SUPPORTED_METHODS)
def test_explain_endpoint_returns_standardized_success_response(
    model_name, method, sample_features
):
    client = TestClient(app)
    response = client.post(
        "/explain",
        json={
            "model": model_name,
            "method": method,
            "input_data": sample_features,
        },
    )

    assert response.status_code == 200
    assert "application/json" in response.headers["content-type"]
    result = response.json()
    assert EXPECTED_KEYS.issubset(result.keys())
    assert result["model"] == model_name
    assert result["method"] == method
    assert isinstance(result["prediction"], int)
    assert isinstance(result["probability"], Real)
    assert 0.0 <= result["probability"] <= 1.0
    assert isinstance(result["feature_importance"], dict)
    assert len(result["feature_importance"]) == 30
    assert all(isinstance(value, Real) for value in result["feature_importance"].values())


def test_explain_rejects_invalid_feature_count(sample_features):
    client = TestClient(app)
    response = client.post(
        "/explain",
        json={
            "model": "random_forest",
            "method": "shap",
            "input_data": sample_features[:29],
        },
    )

    assert response.status_code == 422
    body = _error_payload(response)
    assert body["error"]["type"] == "ValidationError"


def test_explain_rejects_invalid_model(sample_features):
    client = TestClient(app)
    response = client.post(
        "/explain",
        json={
            "model": "svm",
            "method": "shap",
            "input_data": sample_features,
        },
    )

    assert response.status_code == 422
    body = _error_payload(response)
    assert body["error"]["type"] == "ValidationError"


def test_explain_rejects_invalid_method(sample_features):
    client = TestClient(app)
    response = client.post(
        "/explain",
        json={
            "model": "random_forest",
            "method": "foo",
            "input_data": sample_features,
        },
    )

    assert response.status_code == 422
    body = _error_payload(response)
    assert body["error"]["type"] == "ValidationError"


def test_shap_and_lime_share_the_same_response_keys(sample_features):
    client = TestClient(app)
    shap_response = client.post(
        "/explain",
        json={
            "model": "random_forest",
            "method": "shap",
            "input_data": sample_features,
        },
    )
    lime_response = client.post(
        "/explain",
        json={
            "model": "random_forest",
            "method": "lime",
            "input_data": sample_features,
        },
    )

    assert shap_response.status_code == 200
    assert lime_response.status_code == 200
    assert set(shap_response.json().keys()) == set(lime_response.json().keys()) == EXPECTED_KEYS
