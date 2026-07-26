from app.services.method_manager import MethodManager
from app.services.model_adapter import ModelAdapter


class ExplanationEngine:
    """Orchestrates prediction and explanation into a unified EaaS response."""

    def generate_explanation(self, model: str, method: str, input_data: list[float]):
        adapter = ModelAdapter(model)

        prediction = adapter.predict(input_data)
        probability = adapter.predict_proba(input_data)

        manager = MethodManager()
        explanation = manager.explain(
            method=method,
            model_adapter=adapter,
            input_data=input_data,
            prediction=prediction,
        )

        response = {
            "model": model,
            "prediction": prediction,
            "probability": probability,
        }
        response.update(explanation)
        return response
