# EaaS Architecture

## Architectural Principle

Explainability should be treated as an independently orchestrated service capability rather than an embedded model utility.

---

# High-Level Architecture

API Layer

↓

ExplanationEngine

↓

ModelAdapter + MethodManager

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

- Provide training data

Methods:

- load_model(model_name)

- predict(input_data)

- get_training_data()

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

- Normalize outputs

Supported Methods:

- shap

- lime

Purpose:

Demonstrates explanation interface standardization.

---

# Dependency Rules

Allowed:

API Layer

↓

ExplanationEngine

↓

ModelAdapter + MethodManager

Forbidden:

❌ API → SHAP directly

❌ API → LIME directly

❌ API → Model loading directly

❌ MethodManager → HTTP objects

❌ ExplanationEngine → HTTP objects

❌ Route handlers containing explanation logic

---

# Data Flow

Client Request

↓

API Endpoint

↓

ExplanationEngine

↓

ModelAdapter (prediction)

↓

MethodManager (explanation)

↓

Unified Response

↓

Client

---

# Architectural Goals

1. Service abstraction

2. Model agnosticism

3. Explanation standardization

4. Orchestration independence

5. Separation of concerns