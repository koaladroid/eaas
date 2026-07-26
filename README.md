# Explainability-as-a-Service (EaaS)

Master's thesis prototype demonstrating that explainability can be operationalized as an independently orchestrated service capability through standardized interfaces and model-agnostic abstractions.

**Thesis title:** Explainability-as-a-Service (EaaS): Operationalizing Explainable AI Through a Service-Oriented Architectural Abstraction

---

## Architecture (frozen)

```
Client
  │
  ▼
FastAPI  (/predict, /explain)
  │
  ▼
ExplanationEngine
  │
  ├──────────────┐
  ▼              ▼
ModelAdapter   MethodManager
                 │
            ┌────┴────┐
            ▼         ▼
          SHAP       LIME
```

| Component | Responsibility |
|---|---|
| **API Layer** | HTTP transport, validation, error formatting |
| **ExplanationEngine** | Workflow orchestration and unified response assembly |
| **ModelAdapter** | Model loading, prediction, training data access |
| **MethodManager** | Explanation method selection, SHAP/LIME execution, output normalization |

---

## Supported models

- `random_forest`
- `logistic_regression`

## Supported explanation methods

- `shap`
- `lime`

Both methods return the **same standardized explanation schema**.

---

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | Service metadata |
| `GET` | `/health` | Health check |
| `POST` | `/predict` | Model inference |
| `POST` | `/explain` | Prediction + explanation |

Interactive docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## Quick start

```bash
pip install -r requirements.txt
python app/models/train.py
python -m uvicorn app.main:app --reload
```

### Example: predict

```bash
curl -X POST "http://127.0.0.1:8000/predict?model=random_forest" \
  -H "Content-Type: application/json" \
  -d "{\"features\": [<30 float values>]}"
```

### Example: explain (SHAP or LIME)

```bash
curl -X POST "http://127.0.0.1:8000/explain" \
  -H "Content-Type: application/json" \
  -d "{\"model\": \"random_forest\", \"method\": \"shap\", \"input_data\": [<30 float values>]}"
```

Replace `"method": "shap"` with `"method": "lime"` to use LIME. The response schema is identical.

---

## Standardized `/explain` response

```json
{
  "model": "random_forest",
  "method": "shap",
  "prediction": 0,
  "probability": 1.0,
  "feature_importance": {
    "worst concave points": 0.094,
    "worst area": 0.081
  }
}
```

---

## Project layout

```
app/
  api/routes.py              # Thin HTTP layer
  main.py                    # FastAPI application
  models/train.py            # Artifact generation
  services/
    model_adapter.py         # Model abstraction
    method_manager.py        # SHAP + LIME abstraction
    explanation_engine.py    # Orchestration
artifacts/                   # Serialized models and training data
docs/                        # Thesis architecture and contracts
```

---

## Documentation

- [Project Charter](docs/PROJECT_CHARTER.md)
- [Project Rules](docs/PROJECT_RULES.md)
- [Thesis Contribution](docs/THESIS_CONTRIBUTION.md)
- [Architecture](docs/ARCHITECTURE.md)
- [API Contracts](docs/API_CONTRACTS.md)
- [Implementation Plan](docs/IMPLEMENTATION_PLAN.md)
