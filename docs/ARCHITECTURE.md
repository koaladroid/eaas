# EaaS Architecture

## Architectural Principle

Explainability should be treated as an independently orchestrated service capability rather than an embedded model utility.

---

# High-Level Architecture

```
Client
  │
  ▼
API Layer  (/predict, /explain)
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

---

# Component Responsibilities

## API Layer

Responsibilities:

- HTTP request handling
- Input validation
- Response formatting
- Exposing endpoints

Must NOT:

- Execute SHAP
- Execute LIME
- Load models
- Contain business logic

---

## ExplanationEngine

Responsibilities:

- Central orchestration layer
- Coordinate explanation workflow
- Invoke ModelAdapter
- Invoke MethodManager
- Construct unified responses

This is the architectural centerpiece of EaaS.

Must NOT:

- Contain HTTP logic
- Implement SHAP or LIME directly
- Load models directly

---

## ModelAdapter

Responsibilities:

- Abstract model interaction
- Load models
- Perform predictions
- Provide training data and feature names

Methods:

- load_model()
- predict(input_data)
- predict_proba(input_data)
- get_training_data()
- get_feature_names()

Supported Models:

- random_forest
- logistic_regression

Purpose:

Demonstrates model decoupling and model agnosticism.

---

## MethodManager

Responsibilities:

- Select explanation method
- Execute SHAP or LIME
- Normalize outputs into a shared schema
- Reject unsupported methods with a descriptive error

Public interface:

- explain(method, model_adapter, input_data, prediction)

Supported Methods:

- shap
- lime

Purpose:

Demonstrates explanation interface standardization and extensibility.

Both SHAP and LIME are implemented. Adding a method requires extending MethodManager (and the API method enum) without changing ExplanationEngine or ModelAdapter.

---

# Dependency Rules

Allowed:

```
API Layer
  ↓
ExplanationEngine
  ↓
ModelAdapter + MethodManager
```

Forbidden:

- API → SHAP directly
- API → LIME directly
- API → Model loading directly
- MethodManager → HTTP objects
- ExplanationEngine → HTTP objects
- Route handlers containing explanation logic

---

# Data Flow

```
Client Request
  ↓
API Endpoint
  ↓
ExplanationEngine
  ↓
ModelAdapter (prediction + probability)
  ↓
MethodManager (SHAP or LIME explanation)
  ↓
Unified Response
  ↓
Client
```

---

# Architectural Goals

1. Service abstraction
2. Model agnosticism
3. Explanation standardization
4. Orchestration independence
5. Separation of concerns

---

# Architecture Status

The architecture is **frozen** for validation.

Public contracts of ModelAdapter, MethodManager, ExplanationEngine, and the API are stable.
