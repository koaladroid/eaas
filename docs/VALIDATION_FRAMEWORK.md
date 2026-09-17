# Validation Framework

This document defines the formal validation plan for the Explainability-as-a-Service (EaaS) Master's thesis prototype.

It specifies **what** will be validated, **why**, and **what evidence** will be collected. It is not the results chapter. Experimental outcomes will be recorded later.

Authoritative sources for the implemented prototype:

- `docs/THESIS_CONTRIBUTION.md`
- `docs/PROJECT_CHARTER.md`
- `docs/PROJECT_RULES.md`
- `docs/ARCHITECTURE.md`
- `docs/API_CONTRACTS.md`
- `docs/IMPLEMENTATION_PLAN.md`
- `README.md`

The architecture is **frozen**. Validation inspects and exercises the existing system. It does not redesign it.

---

## 1. Purpose

The purpose of this framework is to produce reproducible, thesis-ready evidence that the implemented prototype supports the four primary research claims of the project.

The contribution under evaluation is **architectural**, not algorithmic. Validation therefore focuses on:

- service accessibility of explainability,
- interface standardization across SHAP and LIME,
- separation of orchestration from model and HTTP concerns,
- model-agnostic use of a single explanation architecture.

Validation does **not** evaluate explanation quality, fairness, fidelity, or comparative XAI performance.

---

## 2. Research Objective

Thesis title (project charter):

**Explainability-as-a-Service (EaaS): Operationalizing Explainable AI Through a Service-Oriented Architectural Abstraction**

Research question investigated by the prototype:

How can explainability be operationalized as a reusable, API-driven service capability analogous, at a conceptual level, to service-oriented offerings such as SaaS or MLaaS?

The implemented answer is a lightweight FastAPI service in which explainability is exposed through `POST /explain`, orchestrated by `ExplanationEngine`, and decoupled from model implementations via `ModelAdapter` and from explanation backends via `MethodManager`.

Success statement (`docs/THESIS_CONTRIBUTION.md`):

> Explainability can function as an independently orchestrated service capability rather than an embedded model utility.

---

## 3. Validation Scope

### In scope (implemented prototype)

- FastAPI service (`app.main:app`)
- `GET /`, `GET /health`
- `POST /predict`
- `POST /explain`
- `ExplanationEngine`
- `ModelAdapter`
- `MethodManager`
- Models: `random_forest`, `logistic_regression`
- Methods: `shap`, `lime`
- Standardized `/explain` response contract
- Tracked artifacts under `artifacts/`
- Docker packaging of the frozen service
- Planned lightweight public/cloud API deployment

### Out of scope (not claimed, not validated as implemented)

- New XAI algorithms or explanation-quality benchmarking
- MLOps platforms, CI/CD productization, model registries
- Kubernetes, serverless platforms, service meshes
- Databases, authentication, user management
- Queues, Redis, Celery, event-driven infrastructure
- Frontends
- Automated model-driven generation, meta-model/lifecycle layers
- Audit, provenance, or compliance platforms
- Enterprise production hardening

These items may appear in the broader research narrative as motivation or future work. They are **not** treated as implemented capabilities in this validation plan.

---

## 4. Primary Research Claims

Exactly four primary claims are validated. No additional primary claims are introduced.

### C1 — Service Abstraction

**Claim.** Explainability can be externalized into an independently deployable service capability.

**Architectural evidence to be confirmed.**

- Dedicated `POST /explain` endpoint
- Explanation accessible over HTTP without client-side SHAP/LIME usage
- Explanation logic not embedded in model-specific client code

**Mapped experiment:** V1 (supported by V6)

### C2 — Interface Standardization

**Claim.** Heterogeneous explanation methods can share a standardized service interface.

**Architectural evidence to be confirmed.**

- SHAP and LIME selected through the same `/explain` request contract (`method`)
- Both methods return the same response fields
- Only `method` and feature contribution values are expected to differ

**Mapped experiment:** V2

### C3 — Independent Orchestration

**Claim.** Explainability can be orchestrated independently of model implementations.

**Architectural evidence to be confirmed.**

- `ExplanationEngine` coordinates the workflow
- `ModelAdapter` hides model details
- `MethodManager` hides SHAP/LIME execution
- API routes do not implement explanation algorithms or orchestration beyond calling the engine

**Mapped experiment:** V3 (supported by end-to-end execution in V1/V4)

### C4 — Model Agnosticism

**Claim.** The same explainability architecture can support multiple model implementations without architectural redesign.

**Architectural evidence to be confirmed.**

- Random Forest and Logistic Regression use the same `/explain` endpoint
- Same `ExplanationEngine`, `ModelAdapter`, and `MethodManager`
- Model identity is a request parameter, not a separate architectural path

**Mapped experiment:** V4

Supporting (not primary) concerns:

- Reliability of the published API error envelope — V5
- Independent deployability — V6

---

## 5. Validation Questions

### Primary (claim-aligned)

**VQ1.** Can explainability be accessed through an independent service API rather than being embedded directly into model-specific code?

**VQ2.** Can SHAP and LIME be accessed through the same API interface and produce the same standardized response contract?

**VQ3.** Does the implementation preserve the intended architectural separation between API transport, orchestration, model interaction, and explanation-method execution?

**VQ4.** Can multiple heterogeneous ML models use the same explainability architecture without structural or architectural changes?

### Supporting (not additional primary claims)

**VQ5.** Does the service reject invalid requests using the published error JSON envelope, and fail in a predictable way?

**VQ6.** Can the prototype run as an independently packaged service (Docker), and later be accessed remotely as a public API?

---

## 6. Validation Strategy

Validation combines three complementary modes:

1. **Functional API experiments** against the frozen FastAPI service (`POST /explain`, plus `/health` and `/` for liveness).
2. **Architectural inspection** of source modules, imports, and call flow against `docs/ARCHITECTURE.md`.
3. **Deployment checks** using the existing Docker image and, later, a lightweight public deployment.

Dataset and inputs:

- sklearn Breast Cancer features (exactly 30 numerical values, dataset order)
- valid samples taken from existing `artifacts/training_data.pkl` (for example `X_test[0]`)
- models are **not** retrained during validation

API contracts used (`docs/API_CONTRACTS.md`):

- `POST /predict`: query `model`; body `{ "features": [ ... 30 floats ... ] }`
- `POST /explain`: body `{ "model", "method", "input_data": [ ... 30 floats ... ] }`

Standardized `/explain` success schema:

```json
{
  "model": "random_forest",
  "method": "shap",
  "prediction": 1,
  "probability": 0.87,
  "feature_importance": {
    "mean radius": 0.142,
    "mean texture": -0.031
  }
}
```

SHAP and LIME **must not** be compared for numerical equality of feature contributions. They are different techniques. Validation compares **schema and interface**, not explanation quality.

Existing automated tests under `tests/` may be cited later as supporting instrumentation. This document defines the formal experiment plan; it does not replace the results chapter.

---

## 7. Experiment V1 — API Functional / Service Validation

**Claim:** C1  
**Question:** VQ1

### Objective

Show that a client can obtain an explanation solely through the independent `/explain` service API.

### Procedure

Send a valid `POST /explain` request, for example:

```json
{
  "model": "random_forest",
  "method": "shap",
  "input_data": [<30 floats from artifacts>]
}
```

### Record

| Field | Value to record |
|---|---|
| Endpoint | `POST /explain` |
| Request body | full JSON |
| HTTP status | integer |
| Response body | full JSON |
| `model`, `method` | as returned |
| `prediction` | integer class |
| `probability` | numeric |
| `feature_importance` | dictionary (count of keys) |
| Success/failure | pass/fail against expected result |

### Expected result (pass)

- HTTP 200
- JSON containing `model`, `method`, `prediction`, `probability`, `feature_importance`
- Client did not import SHAP, LIME, or sklearn models
- Client did not load pickle artifacts

### Fail

Non-200 for a valid request, missing required fields, or any requirement that the client execute XAI locally.

### Evidence

API request, API response, HTTP status, endpoint execution log.

---

## 8. Experiment V2 — Explanation Interface Standardization

**Claim:** C2  
**Question:** VQ2

### Objective

Show that SHAP and LIME share one service interface and one response schema.

### Procedure

Using the **same** `POST /explain` endpoint and the **same** 30-feature instance:

1. `random_forest` + `shap`
2. `random_forest` + `lime`

### Schema comparison

Both responses must contain exactly the conceptual field set:

`model`, `method`, `prediction`, `probability`, `feature_importance`

### Expected differences

- `method`: `"shap"` vs `"lime"`
- values inside `feature_importance`

### Expected invariants

- same endpoint
- same request field names (`model`, `method`, `input_data`)
- same response field names
- same orchestration path (API → ExplanationEngine → ModelAdapter + MethodManager)

### Fail

Different endpoints per method, missing fields in one method, or a method-specific response type.

### Evidence

Paired SHAP/LIME request–response records, schema comparison table, note of invariant vs differing fields.

---

## 9. Experiment V3 — Architectural Boundary Validation

**Claim:** C3  
**Question:** VQ3

This experiment is **inspection-led**, not merely a functional ping.

### Objective

Determine whether the frozen implementation conforms to `docs/ARCHITECTURE.md`.

### Inspection targets

| Layer | File(s) | Must | Must not |
|---|---|---|---|
| API | `app/api/routes.py`, `app/main.py` | HTTP, validation, call `ExplanationEngine` for `/explain`, return response | import shap/lime; load artifacts; run explainers |
| ExplanationEngine | `app/services/explanation_engine.py` | create `ModelAdapter`, predict/proba, create `MethodManager`, assemble response | FastAPI/HTTP objects; shap/lime/sklearn explainers; direct pickle I/O |
| ModelAdapter | `app/services/model_adapter.py` | load models, predict, proba, training data, feature names | HTTP; SHAP/LIME |
| MethodManager | `app/services/method_manager.py` | select method, run SHAP/LIME, normalize `{method, feature_importance}` | HTTP; own `/explain` contract assembly of `model`/`prediction`/`probability` |

### Call flow to trace

```
POST /explain
  → ExplanationEngine.generate_explanation(model, method, input_data)
    → ModelAdapter.predict / predict_proba
    → MethodManager.explain(...)
      → _run_shap or _run_lime
    → unified dict returned to client
```

### Allowed dependency direction

`API → ExplanationEngine → ModelAdapter + MethodManager`

### Forbidden (document as defect if found; do not redesign during this plan)

- API → SHAP/LIME
- API → model loading on `/explain`
- ExplanationEngine → HTTP
- MethodManager → HTTP
- explanation algorithms inside route handlers

### Expected result (pass)

Source inspection matches the frozen architecture **and** a valid `/explain` request still completes end-to-end (reuse V1/V4 execution).

### Evidence

Import lists, method call traces, architecture-to-code mapping table, plus one successful E2E `/explain` result referenced from V1 or V4.

---

## 10. Experiment V4 — Model × Method Validation

**Claim:** C4  
**Question:** VQ4

### Objective

Show that two heterogeneous models and two explanation methods share one architecture.

### Procedure

Execute all four combinations through **only** `POST /explain`:

1. `random_forest` + `shap`
2. `random_forest` + `lime`
3. `logistic_regression` + `shap`
4. `logistic_regression` + `lime`

No model-specific explanation endpoint may be used or introduced.

### Results table (to be filled during execution)

| Model | Method | Endpoint | Status | Response schema | Result |
|---|---|---|---|---|---|
| random_forest | shap | POST /explain | | model, method, prediction, probability, feature_importance | |
| random_forest | lime | POST /explain | | same | |
| logistic_regression | shap | POST /explain | | same | |
| logistic_regression | lime | POST /explain | | same | |

For each row also store prediction, probability, and feature_importance key count (expected 30).

### Expected result (pass)

All four return HTTP 200 and the same response field set. Model and method vary by **request parameters**, not by architectural forks.

### Fail

Any combination requiring a different endpoint, engine, or adapter type, or a missing schema field.

### Evidence

Completed table plus the four request/response artifacts.

---

## 11. Experiment V5 — Error Handling / Input Validation

**Role:** supporting reliability validation  
**Question:** VQ5  
**Not a primary research claim.**

### Objective

Confirm that invalid requests fail predictably under the existing contract.

Published error envelope (`docs/API_CONTRACTS.md`):

```json
{
  "error": {
    "type": "...",
    "message": "..."
  }
}
```

Documented types: `InvalidModelError`, `InvalidMethodError`, `ValidationError`, `InternalServerError`.

### Cases

| Case | Example |
|---|---|
| Unsupported model | `"model": "svm"` |
| Unsupported method | `"method": "foo"` |
| Incorrect feature count | 29 or 31 values in `input_data` |
| Malformed body | missing `features`/`input_data`, or empty JSON object |
| Non-numerical input | string in a numeric field, where applicable |

### Record

Request, HTTP status, error JSON, pass/fail against **predictable failure** (structured error, no unexplained 500 for validation cases).

### Expected result

The service does not return a successful explanation for invalid input. Errors use the `{error: {type, message}}` envelope.

**Inspection note (do not fix during this plan):** request enums (`ModelEnum`, `MethodEnum`) may reject unsupported `model`/`method` at HTTP validation time (for example HTTP 422 `ValidationError`) before service-layer `InvalidModelError` / `InvalidMethodError` handlers run. Validation must **record the actual frozen behavior** and compare it to the published contract. If they differ, document the deviation; do not redesign the API in this stage.

### Fail

Unhandled crash, HTML/plain-text errors without the envelope, or a 200 response for invalid input.

---

## 12. Experiment V6 — Deployment Validation

**Role:** supporting deployability evidence for C1  
**Question:** VQ6

### Stage 1 — Docker (already implemented; record existing evidence)

Do not rebuild or redesign Docker unless a later instruction requires it.

Existing local evidence to capture in the results chapter:

- Image built (`eaas:frozen` or successor tag)
- Container started; Uvicorn bound to port 8000
- `GET /health` → HTTP 200
- `GET /` → HTTP 200
- `POST /explain` with `random_forest` + `shap` → HTTP 200, frozen schema
- `POST /explain` with `random_forest` + `lime` → HTTP 200, frozen schema

### Stage 2 — Lightweight public/cloud deployment (pending)

Demonstrate remote API access to the **same** frozen service.

In scope: a simple container or equivalent process reachable over HTTPS/HTTP.

Out of scope: Kubernetes, Redis, Celery, databases, service meshes, enterprise authentication.

### Expected result (pass)

- Stage 1: independently packaged process serving `/explain`
- Stage 2 (when executed): remote client can call `/explain` without local model/XAI libraries

### Evidence

Build/start logs, local HTTP transcripts, later remote URL + request/response.

---

## 13. Evidence Collection

### Functional / API (V1, V2, V4, V5)

Collect for each trial:

- endpoint and HTTP method
- request body
- HTTP status
- response body
- `model`, `method` (where applicable)
- `prediction`, `probability`
- `feature_importance` (or error object)
- success/failure against the criterion in this document
- timestamp and execution environment (local process, Docker, or remote)

### Interface standardization (V2)

- SHAP response
- LIME response
- list of shared keys
- list of differing fields (`method`, contribution values)

### Architecture (V3)

- module/class responsibilities
- imports and forbidden-dependency checks
- call-flow description
- mapping from `docs/ARCHITECTURE.md` to source files

### Model agnosticism (V4)

- four-row combination table
- schema identity across rows

### Deployment (V6)

- Docker build and startup evidence (Stage 1)
- local `/health`, `/`, `/explain` transcripts
- cloud/public URL and remote `/explain` transcript (Stage 2, later)

Evidence must be reproducible from tracked artifacts and the frozen codebase (same 30-feature sample, no retraining).

---

## 14. Success Criteria

Primary validation **passes** if all of the following hold.

1. `POST /explain` is usable as an independent API (V1).
2. SHAP and LIME use the same `/explain` interface (V2).
3. SHAP and LIME return the same response field set (V2, V4).
4. `ExplanationEngine` remains the orchestration component (V3).
5. `ModelAdapter` remains the model-interaction component (V3).
6. `MethodManager` remains the explanation-method component (V3).
7. `/explain` route handlers do not implement SHAP/LIME (V3).
8. Random Forest and Logistic Regression use the same explanation architecture (V4).
9. All four model × method combinations succeed on `/explain` (V4).
10. Invalid inputs fail according to the existing error envelope, with any contract-vs-implementation status-code nuances documented rather than silently “fixed” (V5).
11. The service runs in Docker as an independent process (V6 Stage 1).
12. The service can be shown as remotely accessible when Stage 2 is executed (V6 Stage 2).

The prototype **does not** succeed or fail based on SHAP vs LIME numerical agreement, explanation quality, latency SLOs, or enterprise security posture.

Overarching criterion:

The collected evidence should support, without overclaiming:

> Explainability can function as an independently orchestrated service capability rather than an embedded model utility.

This is evidence from a **proof-of-concept architectural prototype**, not a claim of production-grade cloud-native completeness.

---

## 15. Research Claim Traceability Matrix

| Research claim | Architectural element | Implementation | Experiment | Evidence | Conclusion if pass |
|---|---|---|---|---|---|
| **C1 Service abstraction** | Dedicated explanation API | FastAPI `POST /explain` | V1 (V6 supports deployability) | API request/response, HTTP status | Explainability can be accessed as an independent service capability |
| **C2 Interface standardization** | MethodManager + `/explain` contract | SHAP and LIME backends | V2 | Paired responses, schema comparison | Heterogeneous explanation methods can share a common interface |
| **C3 Independent orchestration** | ExplanationEngine | `app/services/explanation_engine.py` | V3 | Source/dependency inspection + E2E `/explain` | Explainability orchestration is separated from API and model implementations |
| **C4 Model agnosticism** | ModelAdapter + shared `/explain` path | Random Forest, Logistic Regression | V4 | Four model × method results | Multiple models can use the same explainability architecture |
| Reliability (supporting) | API validation / error envelope | Pydantic + error JSON | V5 | Invalid-request results | Invalid input fails predictably under the published envelope |
| Deployment (supporting) | Service/process boundary | Dockerfile + planned public host | V6 | Container logs + remote API (Stage 2) | The prototype can operate as an independently packaged service |

---

## 16. Implemented vs Conceptual Scope

### Implemented / to be validated

- FastAPI microservice
- `/predict`, `/explain`, `/health`, `/`
- ModelAdapter, MethodManager, ExplanationEngine
- Random Forest and Logistic Regression artifacts
- SHAP and LIME behind MethodManager
- Standardized `/explain` schema
- Docker packaging
- Planned lightweight public API deployment

### Conceptual / motivation / future work (not validated as implemented)

- Meta-model or lifecycle specification layers
- Model-driven automated generation
- Audit, traceability, provenance, and compliance platforms
- Kubernetes, serverless, and rich cloud topology
- Databases, authentication, queues, Redis, Celery
- Production MLOps
- New XAI algorithms
- Explanation-quality benchmarking
- Enterprise infrastructure and frontends

Validation text must not state that unimplemented items exist in the prototype.

---

## 17. Deployment Validation Plan

### Local container (complete; results to be archived)

1. Build image from existing `Dockerfile` (`python -m uvicorn app.main:app --host 0.0.0.0 --port 8000`).
2. Run container with `app/` and `artifacts/` present.
3. Confirm `/health`, `/`, and `/explain` (at least SHAP and LIME on Random Forest).

No Docker redesign unless later explicitly required.

### Public/cloud (not yet executed)

1. Deploy the **same** frozen image or equivalent process to a simple public host.
2. Call `POST /explain` from a remote client.
3. Record URL, status, and response schema.
4. Do not add Kubernetes, databases, auth products, or extra orchestration.

---

## 18. Expected Validation Outcome

If experiments V1–V4 pass, and V5–V6 are completed as specified, the thesis can report that the prototype provides architectural evidence for:

- externalizing explainability as an API service (C1),
- standardizing heterogeneous methods behind one contract (C2),
- orchestrating explanations independently of models and HTTP (C3),
- serving multiple models through one explanation architecture (C4).

Expected narrative outcome:

The EaaS prototype shows that SHAP and LIME can be consumed through a single `/explain` interface, that Random Forest and Logistic Regression can share that interface, and that orchestration resides in `ExplanationEngine` rather than in model code or route handlers.

What this framework will **not** conclude:

- that EaaS is production-ready,
- that SHAP or LIME were improved,
- that explanation quality was measured,
- that a full cloud-native platform was built.

Those conclusions would exceed the prototype and are excluded from this validation plan.
