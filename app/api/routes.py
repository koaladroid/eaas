from enum import Enum

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.services.model_adapter import ModelAdapter

router = APIRouter()


class ModelType(str, Enum):
    random_forest = "random_forest"
    logistic_regression = "logistic_regression"


class PredictionRequest(BaseModel):
    features: list[float] = Field(min_length=30, max_length=30)


class PredictionResponse(BaseModel):
    model: str
    prediction: int
    probability: float


@router.post("/predict", response_model=PredictionResponse)
def predict(model: ModelType, request: PredictionRequest):
    try:
        adapter = ModelAdapter(model.value)
        prediction = adapter.predict(request.features)
        probability = adapter.predict_proba(request.features)
    except ValueError as e:
        message = str(e)
        if message.startswith("Unsupported model"):
            return JSONResponse(
                status_code=400,
                content={"error": {"type": "InvalidModelError", "message": message}},
            )
        return JSONResponse(
            status_code=400,
            content={"error": {"type": "ValidationError", "message": message}},
        )
    except Exception:
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "type": "InternalServerError",
                    "message": "An unexpected error occurred.",
                }
            },
        )

    return PredictionResponse(
        model=model.value,
        prediction=prediction,
        probability=probability,
    )
