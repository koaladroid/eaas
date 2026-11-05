# Evaluation Guidelines

This document governs the empirical and architectural evaluation of the existing Explainability-as-a-Service (EaaS) prototype.

It is **not** an implementation plan for new product features. Production architecture, public API contracts, models, and explanation backends remain **frozen** unless the user explicitly approves a change.

Authoritative implementation sources:

- `docs/ARCHITECTURE.md`
- `docs/API_CONTRACTS.md`
- `docs/THESIS_CONTRIBUTION.md`
- `docs/PROJECT_CHARTER.md`
- `docs/PROJECT_RULES.md`
- `docs/VALIDATION_FRAMEWORK.md`
- `docs/VALIDATION_RESULTS.md` (V1–V5 already recorded; not a substitute for E1–E4)

---

## 1. Thesis positioning

**Thesis title:** Explainability-as-a-Service (EaaS): Operationalizing Explainable AI as a Cloud-Native, Model-Driven Microservice

**Degree context:** MSc Software Engineering.

**Primary research question:**

How can explainability be operationalized as a reusable service rather than being tied to an individual application or model?

**Contribution of this thesis:**

A lightweight, **contract-oriented** EaaS **reference implementation** and its **empirical and architectural evaluation**. The contribution is architectural and evaluative, not algorithmic.

The work demonstrates that, in this prototype:

- explainability is accessible through `POST /explain`;
- SHAP and LIME share one response schema;
- orchestration is centralized in `ExplanationEngine`;
- Random Forest and Logistic Regression use the same explanation path.

**Prior art:** Service-oriented XAI, API-based explanation services, and related “explainability as a service” discussions already exist in the literature. This project **does not claim to have invented EaaS**. It implements a small, inspectable reference system and evaluates it against explicit claims (compatibility, contracts, runtime, architecture, invalid-input handling).

**Out of evaluation scope:** explanation quality, fidelity, fairness, SHAP-vs-LIME superiority, production scalability, and enterprise MLOps.

---

## 2. Frozen architecture

Do not modify these without explicit approval.

```
Client
  → FastAPI  (/predict, /explain)
    → ExplanationEngine
      → ModelAdapter + MethodManager
                    → SHAP / LIME
```

| Element | Location / contract |
|---|---|
| API layer | `app/main.py`, `app/api/routes.py` |
| ExplanationEngine | `app/services/explanation_engine.py` |
| ModelAdapter | `app/services/model_adapter.py` |
| MethodManager | `app/services/method_manager.py` |
| Models | `random_forest`, `logistic_regression` |
| Methods | `shap`, `lime` |
| Dataset features | **exactly 30** numerical Breast Cancer features, dataset order |

### `/explain` (evaluation target)

Request body:

```json
{
  "model": "random_forest",
  "method": "shap",
  "input_data": [ /* 30 floats */ ]
}
```

Success response keys (mandatory, identical for SHAP and LIME):

`model`, `method`, `prediction`, `probability`, `feature_importance`

### `/predict` (not the primary E1–E4 HTTP surface)

Query `model`; body `{ "features": [ /* 30 floats */ ] }`.  
Implemented via `ModelAdapter` directly. Do not treat this as the explanation orchestration path.

### Error envelope (documented)

```json
{ "error": { "type": "...", "message": "..." } }
```

Documented types: `InvalidModelError`, `InvalidMethodError`, `ValidationError`, `InternalServerError`.

**Known implementation note (V5):** unsupported `model` / `method` on `/explain` are rejected by Pydantic enums as HTTP **422** `ValidationError` (`Invalid request body.`) before service-layer 400 handlers. E4 must **record actual status and body**, then compare to the written contract. Do not change production code to force a match.

---

## 3. Experimental design

HTTP experiments (E1, E2, E4) must call the **running service** `POST /explain`. They must not invoke SHAP or LIME in the evaluation process except through that API.

Use existing `artifacts/training_data.pkl`. **Do not retrain.**

### Combinations (E1, E2)

1. `random_forest` + `shap`  
2. `random_forest` + `lime`  
3. `logistic_regression` + `shap`  
4. `logistic_regression` + `lime`

### E1 — Model–method compatibility and response-contract consistency

| Item | Value |
|---|---|
| Samples | **First 30 rows** of `X_test` (`X_test[0:30]`) |
| Per combination | 30 requests |
| Total | **120** valid requests |
| Same rows | Identical 30 vectors for every combination |

For each response, check:

- HTTP status (expect 200 for valid requests);
- keys: `model`, `method`, `prediction`, `probability`, `feature_importance`;
- `model` and `method` match the request;
- `prediction` is an integer class label;
- `probability` is numeric and in `[0, 1]`;
- `feature_importance` is a dict of **30** entries;
- keys correspond to dataset feature names;
- values are numeric and finite;
- schema is the same across all four combinations.

Do **not** require SHAP and LIME (or repeated LIME calls) to have equal contribution values.

### E2 — Runtime performance

| Item | Value |
|---|---|
| HTTP timeout | **120 seconds** per request |
| Sessions | **2**, started **at least 30 minutes apart** |
| Combination order | Randomized **between sessions**; sample order **fixed** within a combination |
| Warm-up | **10** requests per combination per session using `X_test[0:10]` |
| Warm-up in statistics | **Excluded** from all measured results |
| Measured attempts | **Exactly 100** per combination per session |
| Measured total | **800** attempts (4 combinations × 100 × 2 sessions) |
| Measured inputs | `X_test[10:110]` (100 rows), **identical** across combinations and sessions |
| Failures | HTTP errors, **timeouts**, and **connection errors** count inside the 100; **never retry or replace** |
| Protocol | **Sequential** HTTP `POST /explain` (no parallel client load) |
| Latency clock | Python `time.perf_counter()` (client-observed) |

Record **per attempt** (raw), including failures:

- combination, session id, sample index (`10..109` for measured rows);
- wall-clock start/end timestamps;
- `perf_counter` elapsed latency (milliseconds) where a request completed or timed out;
- HTTP status, or timeout/connection-error class;
- success/failure;
- error body if any;
- test configuration (base URL, client, **120 s** timeout).

Also record session start/end timestamps and the **actual** gap between sessions.

**Success rate / error rate** (per combination, per session, and overall):

- Denominator = all **100** measured **attempts** (warm-up excluded).
- Numerator for success = HTTP **200** only.
- Timeouts, connection errors, and non-200 responses remain in the denominator. Do **not** hide failures by dropping them from rate calculations.

**Latency statistics** (mean, median, minimum, maximum, P95):

- Use **only** measured attempts that returned HTTP **200**.
- Explicitly report **n** = number of successful observations used.
- P95: `np.percentile(latencies, 95, method="linear")`.

**Interpretation:** report **sequential, end-to-end, client-observed API latency**. Do **not** claim isolated SHAP/LIME time, concurrency capacity, throughput, scalability, high availability, or long-term reliability.

### E3 — Architectural evaluation

Inspect the actual source files listed in §2.

Verify, with file references:

- API: HTTP, validation, `/explain` delegates to `ExplanationEngine`;
- ExplanationEngine: orchestration only; no HTTP, shap, lime, or pickle I/O;
- ModelAdapter: load/predict/data for both models;
- MethodManager: method selection, SHAP/LIME, normalized `{method, feature_importance}`.

Document any boundary violations. V3 in `docs/VALIDATION_RESULTS.md` may be reused as a starting inspection; re-check the current tree.

A **controlled change-impact analysis** (e.g. “what files would change to add a method”) is optional and **must not modify production code**. Record it as a paper exercise if done.

### E4 — Invalid-input handling

Six categories, **three repetitions** each, **18** requests, against `POST /explain`:

| ID | Category | Example |
|---|---|---|
| E4.1 | Unsupported model | `"model": "unsupported_model"` |
| E4.2 | Unsupported method | `"method": "unsupported_method"` |
| E4.3 | Incorrect feature count | 29 floats in `input_data` |
| E4.4 | Nonnumeric feature | one string in the 30-length list |
| E4.5 | Missing required `input_data` | omit the field |
| E4.6 | Malformed JSON body | not valid JSON |

Record: request, HTTP status, response body, whether `{error: {type, message}}` is present, whether rejection is correct (not HTTP 200 with an explanation), whether the **documented** type (`InvalidModelError` / `InvalidMethodError` / `ValidationError`) matches **observed** type, and the layer if inferable (Pydantic vs service).

Preserve mismatches as **deviations**, not as silent passes.

---

## 4. Sampling and stochasticity

- Source: `artifacts/training_data.pkl` (`X_test` shape **114 × 30**).
- Keep the original train/test split; do not reshuffle or retrain.
- E1 uses indices `0..29`. E2 warm-up uses `0..9`; E2 measured uses `10..109`. The same index means the same feature vector for every model–method pair.
- If a run needs a stored fixture, export those rows once under `evaluation/fixtures/` without changing artifacts.
- **LIME** (and any non-deterministic backend) may vary numerically across repeats. Evaluate **structure and finiteness**, not bitwise equality of maps.
- SHAP vs LIME values must not be tested for equality.

---

## 5. Deployment and environment

**Evaluation HTTP target:**

`https://eaas-production-df33.up.railway.app/`

**Owner-confirmed:** `GET /health` succeeds; `GET /docs` is accessible.

**Independently smoke-tested (one request per combination, not experimental evidence):** all four model–method pairs on `POST /explain` returned HTTP 200 with the frozen schema. That probe is **not** E1, E2, or E4.

**Unverified:** deployed Git commit; Railway runtime / image Python version.

- Local uvicorn or Docker (`python:3.12-slim`, host `0.0.0.0`, port **8000**) may be used only if the public service is unavailable; that substitution **must** be recorded. A successful local run **does not** prove public deployment.
- Dockerfile `CMD`: `python -m uvicorn app.main:app --host 0.0.0.0 --port 8000`. The image does **not** read `$PORT`.
- Record: Python version, `pip freeze` or image digest, OS, client library, base URL, date/time (UTC), git commit (`HEAD` at run time).
- Observed versions at inspection: **local Python 3.14.4**; **Docker base Python 3.12**. Do **not** change `requirements.txt` or pin versions without approval.
- Dependencies are unpinned; record resolved versions per session.

---

## 6. Evaluation directory structure

Create later (not in this task). Keep evaluation **out of** `app/`.

```
evaluation/
  README.md                 # how to run E1–E4; environment checklist
  fixtures/                 # exported X_test rows used by E1/E2 (including 0:10 and 10:110)
  scripts/                  # HTTP clients and summary generators only
  results/raw/              # JSONL/CSV including failures
  results/summaries/        # derived tables; cite raw files
  results/figures/          # optional plots generated from raw data
  architecture-audit.md     # E3 notes and source citations
```

Scripts may use HTTP against the service. They must not patch production modules, retrain models, or import SHAP/LIME to “help” the API.

Existing `tests/` remain architectural/unit checks. They are **not** the E1–E4 protocol.

---

## 7. Reproducibility and evidence

- Keep **raw** rows for every request, including failures and timeouts.
- Never fabricate, interpolate, or drop measurements to improve rates.
- Derived statistics must be recomputable from raw files. P95 is `np.percentile(..., 95, method="linear")`. Never retry or replace failed E2 attempts.
- Log: git commit, base URL, Python, dependency versions, session id, start/end time, protocol deviations (retries, restarts, wrong index).
- Label text as **observation**, **interpretation**, or **limitation**.
- Do not overwrite raw files; use timestamped names.

---

## 8. Future work (not current requirements)

The following are **proposals for later research**, not tasks for this evaluation or for production code:

- Schema-aware EaaS (typed feature schemas beyond a fixed length-30 vector);
- Semantic contract verification (richer than key-set checks);
- Reproducible explanations and provenance (seeds, explainer versions, audit trails).

Do not implement these unless separately approved.

---

## 9. Cursor working rules

1. No production architecture changes.
2. No public API request/response contract changes.
3. No new models, methods, endpoints, databases, auth, queues, or Kubernetes.
4. Evaluation scripts stay under `evaluation/` (or equivalent), separate from `app/`.
5. No commits, pushes, merges, or deployments without explicit user approval.
6. If implementation disagrees with documentation, **document** the disagreement; do not “fix” the service during evaluation.
7. All experimental numbers come from actual execution.

---

## Checklist before first E1/E2 run

- [ ] Confirm git commit to be evaluated.
- [ ] Confirm HTTP base URL (`https://eaas-production-df33.up.railway.app/` unless a documented fallback is required).
- [ ] Confirm `X_test` has at least 110 rows (currently 114) for `X_test[10:110]`.
- [ ] Confirm client timeout is **120 seconds** per request.
- [ ] Confirm raw output path exists and will not be overwritten.
