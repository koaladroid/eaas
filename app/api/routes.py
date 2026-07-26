from enum import Enum

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.services.explanation_engine import ExplanationEngine
from app.services.model_adapter import ModelAdapter

router = APIRouter()


class ModelEnum(str, Enum):
    random_forest = "random_forest"
    logistic_regression = "logistic_regression"


class MethodEnum(str, Enum):
    shap = "shap"
    lime = "lime"


# Preserve existing /predict query-parameter typing.
ModelType = ModelEnum


class PredictionRequest(BaseModel):
    features: list[float] = Field(min_length=30, max_length=30)


class PredictionResponse(BaseModel):
    model: str
    prediction: int
    probability: float


class ExplainRequest(BaseModel):
    model: ModelEnum
    method: MethodEnum
    input_data: list[float] = Field(min_length=30, max_length=30)


class ExplainResponse(BaseModel):
    model: str
    method: str
    prediction: int
    probability: float
    feature_importance: dict[str, float]


def _error_response(exc: Exception):
    if isinstance(exc, ValueError):
        message = str(exc)
        if message.startswith("Unsupported model"):
            return JSONResponse(
                status_code=400,
                content={"error": {"type": "InvalidModelError", "message": message}},
            )
        if message.startswith("Unsupported explanation method"):
            return JSONResponse(
                status_code=400,
                content={"error": {"type": "InvalidMethodError", "message": message}},
            )
        return JSONResponse(
            status_code=400,
            content={"error": {"type": "ValidationError", "message": message}},
        )
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "type": "InternalServerError",
                "message": "An unexpected error occurred.",
            }
        },
    )


@router.post("/predict", response_model=PredictionResponse)
def predict(model: ModelType, request: PredictionRequest):
    try:
        adapter = ModelAdapter(model.value)
        prediction = adapter.predict(request.features)
        probability = adapter.predict_proba(request.features)
    except Exception as e:
        return _error_response(e)

    return PredictionResponse(
        model=model.value,
        prediction=prediction,
        probability=probability,
    )


@router.post("/explain", response_model=ExplainResponse)
def explain(request: ExplainRequest):
    try:
        engine = ExplanationEngine()
        return engine.generate_explanation(
            model=request.model.value,
            method=request.method.value,
            input_data=request.input_data,
        )
    except Exception as e:
        return _error_response(e)
