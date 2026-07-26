# EaaS Implementation Plan

This document defines the implementation roadmap for the Explainability-as-a-Service (EaaS) prototype.

Status legend: **Done** | **Next** | Planned

---

# Step 1 — Project Setup — Done

Objectives:

- Create project structure
- Configure FastAPI application
- Implement health check endpoint
- Verify server execution

Deliverable:

- Running FastAPI service.

---

# Step 2 — Model Training — Done

Objectives:

- Train RandomForest model
- Train LogisticRegression model
- Save models to artifacts
- Save training data

Deliverable:

- Serialized models and training dataset.

---

# Step 3 — ModelAdapter — Done

Objectives:

- Implement model abstraction layer.
- Standardize model interaction.

Methods:

- load_model()
- predict()
- predict_proba()
- get_training_data()
- get_feature_names()

Deliverable:

- Working adapter supporting both models.

---

# Step 4 — Prediction Endpoint — Done

Objectives:

- Implement POST /predict.
- Integrate ModelAdapter.

Deliverable:

- Prediction API with standardized response.

---

# Step 5 — MethodManager + SHAP — Done

Objectives:

- Implement method selection layer.
- Add SHAP explanations.

Deliverable:

- SHAP explanations with standardized schema.

---

# Step 6 — ExplanationEngine — Done

Objectives:

- Implement orchestration layer.
- Coordinate ModelAdapter and MethodManager.
- Assemble the unified explanation response.

Deliverable:

- Stateless ExplanationEngine as the architectural centerpiece.

---

# Step 7 — /explain Endpoint — Done

Objectives:

- Expose ExplanationEngine through POST /explain.
- Keep the API layer thin (validation and transport only).

Deliverable:

- Fully functional explanation API supporting the frozen architecture.

---

# Step 8 — LIME Integration — Done

Objectives:

- Integrate LIME into MethodManager.
- Preserve the public `explain(...)` interface.
- Preserve response schema consistency with SHAP.

Deliverable:

- SHAP and LIME producing identical response structures.
- MethodEnum exposing both `shap` and `lime`.

---

# Architecture Freeze — Done

Objectives:

- Synchronize documentation with the implemented architecture.
- Confirm unsupported-method rejection in MethodManager.
- Freeze public contracts before validation.

Deliverable:

- Documentation and code aligned; architecture ready for validation.

---

# Step 9 — Model-Agnostic Demonstration — Next

Objectives:

- Validate architecture against both models and both methods.

Scenarios:

1. RandomForest + SHAP
2. RandomForest + LIME
3. LogisticRegression + SHAP
4. LogisticRegression + LIME

Deliverable:

- Evidence that the same architecture supports multiple models and methods without structural changes.

---

# Step 10 — Error Handling Review — Planned

Objectives:

- Confirm standardized errors.
- Confirm input validation.
- Confirm graceful handling of invalid requests.

Deliverable:

- Stable API behavior under invalid inputs.

---

# Step 11 — End-to-End Testing — Planned

Objectives:

- Build demonstration / test coverage for thesis evaluation.
- Exercise all model × method combinations.

Deliverable:

- Demonstration script for thesis evaluation.

---

# Final Deliverable

A lightweight cloud-deployable prototype that demonstrates:

- explainability externalization,
- standardized explanation interfaces,
- model-agnostic orchestration,
- API-driven operationalization of explainability.
