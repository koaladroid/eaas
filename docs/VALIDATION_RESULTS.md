# Validation Results

This document records executed validation evidence for the frozen EaaS prototype, following `docs/VALIDATION_FRAMEWORK.md`.

Execution environment:

- Docker container `eaas-frozen` on `http://127.0.0.1:8000`
- Input instance: `artifacts/training_data.pkl` → `X_test[0]` (30 Breast Cancer features)
- Date of execution: 2026-09-08

No application source code was modified during these experiments.

---

## V1 — API Functional / Service Validation

### Test V1.1 — RandomForest + SHAP

- Request:

```http
POST /explain
Content-Type: application/json
```

```json
{
  "model": "random_forest",
  "method": "shap",
  "input_data": [
    19.55, 28.77, 133.6, 1207.0, 0.0926, 0.2063, 0.1784, 0.1144, 0.1893, 0.06232,
    0.8426, 1.199, 7.158, 106.4, 0.006356, 0.04765, 0.03863, 0.01519, 0.01936, 0.005252,
    25.05, 36.27, 178.6, 1926.0, 0.1281, 0.5329, 0.4251, 0.1941, 0.2818, 0.1005
  ]
}
```

- HTTP Status: `200`

- Response:

```json
{
  "model": "random_forest",
  "method": "shap",
  "prediction": 0,
  "probability": 1.0,
  "feature_importance": {
    "worst concave points": 0.0939898161429568,
    "worst area": 0.08109447529100106,
    "worst radius": 0.055731161709292765,
    "mean concave points": 0.05514700419162046,
    "worst perimeter": 0.05125068485207676,
    "mean perimeter": 0.041712915472802696,
    "mean radius": 0.03863510040255016,
    "mean concavity": 0.038563861770973595,
    "mean area": 0.03588464548053283,
    "area error": 0.03468424356569953,
    "worst concavity": 0.02486326956991555,
    "radius error": 0.01766599605064866,
    "worst compactness": 0.01309379662663422,
    "worst texture": 0.013066358948923731,
    "perimeter error": 0.00969749910493148,
    "mean texture": 0.008168439585172884,
    "mean compactness": 0.00533976872873891,
    "worst smoothness": -0.0021260262935812214,
    "worst fractal dimension": 0.0018502791695782627,
    "texture error": 0.0018344582546897617,
    "compactness error": -0.001815399217655654,
    "concavity error": 0.0014996742323926111,
    "concave points error": 0.0012910572337011925,
    "symmetry error": 0.0009495188945498279,
    "fractal dimension error": 0.0007031282010600899,
    "mean fractal dimension": 0.0005418379613945962,
    "mean smoothness": 0.0005193152032181277,
    "smoothness error": 0.0004822762215586709,
    "mean symmetry": 0.00027982982003698417,
    "worst symmetry": -0.00011547069189882745
  }
}
```

- Result: The independent `/explain` API returned a complete standardized explanation. The client did not invoke SHAP, load model artifacts, or execute model-specific explanation code. Response fields present: `model`, `method`, `prediction`, `probability`, `feature_importance` (30 numeric contributions). `prediction` = `0`, `probability` = `1.0`.

- Pass/Fail: **Pass**

---

### Test V1.2 — RandomForest + LIME

- Request:

```http
POST /explain
Content-Type: application/json
```

```json
{
  "model": "random_forest",
  "method": "lime",
  "input_data": [
    19.55, 28.77, 133.6, 1207.0, 0.0926, 0.2063, 0.1784, 0.1144, 0.1893, 0.06232,
    0.8426, 1.199, 7.158, 106.4, 0.006356, 0.04765, 0.03863, 0.01519, 0.01936, 0.005252,
    25.05, 36.27, 178.6, 1926.0, 0.1281, 0.5329, 0.4251, 0.1941, 0.2818, 0.1005
  ]
}
```

- HTTP Status: `200`

- Response:

```json
{
  "model": "random_forest",
  "method": "lime",
  "prediction": 0,
  "probability": 1.0,
  "feature_importance": {
    "worst concave points": 0.13941677594992605,
    "worst area": 0.13818678283881805,
    "worst radius": 0.10255419088392344,
    "worst perimeter": 0.08629581774047046,
    "area error": 0.05675559885094926,
    "mean concave points": 0.05632954316003043,
    "mean radius": 0.05052285884064057,
    "mean perimeter": 0.04850672443148771,
    "worst texture": 0.047061259155076876,
    "mean concavity": 0.03733921494458518,
    "worst concavity": 0.0339006893386852,
    "mean texture": 0.027923713252670547,
    "mean area": 0.027904621789135654,
    "radius error": 0.021637268996338773,
    "perimeter error": 0.020940658684534534,
    "worst smoothness": -0.012619570606574022,
    "worst compactness": 0.010786868526366924,
    "compactness error": -0.00944150532897597,
    "fractal dimension error": -0.009217949282060323,
    "worst symmetry": -0.008331849765982396,
    "mean compactness": 0.007825927621948206,
    "symmetry error": 0.0028896162351588353,
    "texture error": 0.002770082836439627,
    "mean symmetry": -0.0024399036134118168,
    "concavity error": 0.0020536973118297835,
    "concave points error": 0.00164266999482657,
    "smoothness error": -0.0015982795730442715,
    "mean fractal dimension": 0.0010675304314478043,
    "worst fractal dimension": 0.0009624057591450346,
    "mean smoothness": -0.0002146896580585782
  }
}
```

- Result: The same `/explain` endpoint returned a complete standardized explanation for LIME. Response fields present: `model`, `method`, `prediction`, `probability`, `feature_importance` (30 numeric contributions). `prediction` = `0`, `probability` = `1.0`. The client did not invoke LIME locally.

- Pass/Fail: **Pass**

---

## V2 — Explanation Interface Standardization

Comparison uses V1.1 and V1.2: identical endpoint, identical 30-feature instance, `random_forest`, methods `shap` vs `lime`.

### Schema Comparison

| Field | SHAP | LIME | Result |
|---|---|---|---|
| model | `"random_forest"` (string) | `"random_forest"` (string) | Same field present; same requested model |
| method | `"shap"` (string) | `"lime"` (string) | Same field present; values differ as expected |
| prediction | `0` (integer) | `0` (integer) | Same field present; same type |
| probability | `1.0` (number in [0, 1]) | `1.0` (number in [0, 1]) | Same field present; same type |
| feature_importance | object, 30 numeric entries | object, 30 numeric entries | Same field present; same structure; **values differ** (expected) |

### Invariants

- Endpoint: `POST /explain` for both
- Request interface: `{ "model", "method", "input_data" }` with exactly 30 floats
- Response structure: `{ "model", "method", "prediction", "probability", "feature_importance" }`
- `feature_importance` is a dictionary of feature name → numeric contribution (30 keys)

### Expected differences (observed)

- `method`: `"shap"` vs `"lime"`
- Numeric feature contribution values (and ranking magnitudes) differ between SHAP and LIME

No comparison of explanation quality, accuracy, or superiority is made.

### V2 Conclusion

The evidence supports the claim that SHAP and LIME can share the same standardized service interface. Both methods were selected through `POST /explain` using the same request schema and both returned the same response field structure. Differences were limited to the `method` identifier and the feature contribution values, which is consistent with two distinct explanation techniques behind one service contract.

- Pass/Fail: **Pass**

---

## V3 — Architectural Boundary Validation

Inspection date: 2026-09-08. Source was read as implemented; no code was modified.

End-to-end behavior referenced from V1.1 and V1.2 (`POST /explain`, `random_forest` + `shap` / `lime`, HTTP 200). Successful HTTP execution is **not** treated as sufficient by itself. C3 is assessed from inspection **plus** that E2E path.

### V3.1 API Layer

**Inspected files:** `app/main.py`, `app/api/routes.py`

**Modules / functions:**

| Symbol | Role |
|---|---|
| `app` (`FastAPI`) | Application object; mounts router; `GET /`, `GET /health`; validation exception handler |
| `router` (`APIRouter`) | HTTP routes |
| `predict()` | `POST /predict` |
| `explain()` | `POST /explain` |
| Pydantic models | Request/response schemas |
| `_error_response()` | Maps exceptions to the published error JSON envelope |

**Responsibilities observed:**

- HTTP handling and status codes
- Request validation (`PredictionRequest` / `ExplainRequest`, enums, 30-feature length)
- Response models (`PredictionResponse` / `ExplainResponse`)
- `/explain` invokes only `ExplanationEngine().generate_explanation(...)` and returns that result (or an error envelope)

**Checks against frozen “API must not” list:**

| Check | `/explain` | `/predict` |
|---|---|---|
| Direct SHAP | No `shap` import or call | No |
| Direct LIME | No `lime` import or call | No |
| Load ML artifacts / pickle | No | Indirectly via `ModelAdapter(...)` |
| Model-specific explanation logic | No | No |
| Explanation orchestration | No; delegates to `ExplanationEngine` | N/A (prediction only) |

**Finding:** For explainability (C3), `/explain` is a transport layer only. `/predict` calls `ModelAdapter` directly for inference (Step 4 of the implementation plan). That is a prediction-path dependency, not explanation orchestration, and it does not execute SHAP/LIME.

**Pass/Fail:** **Pass** (explanation path). `/predict`→`ModelAdapter` is recorded as a scoped observation, not a C3 failure.

---

### V3.2 ExplanationEngine

**Inspected file:** `app/services/explanation_engine.py`

**Class / method:** `ExplanationEngine.generate_explanation(model, method, input_data)`

**Imports:** `MethodManager`, `ModelAdapter` only. No FastAPI, no HTTP, no `shap`, no `lime`, no `joblib`, no sklearn.

**Actual call flow:**

1. `adapter = ModelAdapter(model)`
2. `prediction = adapter.predict(input_data)`
3. `probability = adapter.predict_proba(input_data)`
4. `manager = MethodManager()`
5. `explanation = manager.explain(method, adapter, input_data, prediction)`
6. Merge `{model, prediction, probability}` with `{method, feature_importance}` and return

**Checks:**

- Central orchestration: yes
- Coordinates ModelAdapter and MethodManager: yes
- Independent of HTTP: yes
- Does not implement SHAP/LIME: yes
- Does not load pickles itself: yes (loading is inside ModelAdapter)

**Pass/Fail:** **Pass**

---

### V3.3 ModelAdapter

**Inspected file:** `app/services/model_adapter.py`

**Class:** `ModelAdapter`

**Public surface:** `__init__(model_name)`, `load_model()`, `predict()`, `predict_proba()`, `get_training_data()`, `get_feature_names()`

**Observed behavior:**

- Selects artifact by name (`random_forest.pkl` / `logistic_regression.pkl`)
- Loads via `joblib`
- Predicts and returns predicted-class probability
- Supplies training data and feature names used by MethodManager
- Rejects unknown model names with `ValueError`

Random Forest and Logistic Regression are both accessed through **the same class and methods**; the model name is a constructor argument. Orchestration (`ExplanationEngine`) does not import sklearn model classes.

**Pass/Fail:** **Pass**

---

### V3.4 MethodManager

**Inspected file:** `app/services/method_manager.py`

**Class / public method:** `MethodManager.explain(method, model_adapter, input_data, prediction)`

**Imports of XAI libraries:** `shap`, `lime.lime_tabular.LimeTabularExplainer` — confined to this module. No FastAPI/HTTP imports.

**Observed behavior:**

- Selects `shap` or `lime`
- Executes `_run_shap` / `_run_lime`
- Uses ModelAdapter for `model`, `get_training_data()`, `get_feature_names()`
- Normalizes to `{ "method", "feature_importance" }`
- Rejects other methods with `ValueError` listing supported methods

SHAP/LIME objects are not returned to the API or ExplanationEngine. sklearn type checks (`RandomForestClassifier` / `LogisticRegression`) appear only inside `_run_shap` to choose a SHAP explainer; they do not leak into routes or ExplanationEngine.

**Pass/Fail:** **Pass**

---

### V3.5 Dependency / Call Flow

**Explainability path (frozen architecture):**

```
Client
  → POST /explain  (app/api/routes.py::explain)
    → ExplanationEngine.generate_explanation
      → ModelAdapter.predict / predict_proba   (artifacts via joblib)
      → MethodManager.explain
        → shap.TreeExplainer / shap.LinearExplainer  or  LimeTabularExplainer
      → unified JSON returned by the route
```

**Prediction path (not explanation orchestration):**

```
POST /predict  (app/api/routes.py::predict)
  → ModelAdapter.predict / predict_proba
```

**Import evidence:**

| Module | Imports services / XAI |
|---|---|
| `app/main.py` | `app.api.routes` only |
| `app/api/routes.py` | `ExplanationEngine`, `ModelAdapter`; no shap/lime |
| `explanation_engine.py` | `ModelAdapter`, `MethodManager`; no shap/lime/HTTP |
| `model_adapter.py` | joblib, numpy; no shap/lime/HTTP |
| `method_manager.py` | shap, lime, sklearn classifiers; no HTTP |

**Architectural violations (C3 / `/explain` path):** **None.**

Observation (not a C3 violation): `/predict` depends on `ModelAdapter` without `ExplanationEngine`, which matches the dedicated prediction endpoint, not the explanation workflow.

---

### V3.6 Architectural Boundary Assessment

The `/explain` implementation conforms to the frozen architecture:

`API → ExplanationEngine → ModelAdapter + MethodManager → SHAP/LIME`

V1.1 and V1.2 show that RF+SHAP and RF+LIME complete through that **same** HTTP interface. Combined with the import/call-flow inspection, this supports independent orchestration: the client and the API layer do not execute explainers; `ExplanationEngine` sequences adapter and manager; model and method details stay in their respective components.

**Pass/Fail:** **Pass**

---

### V3 Conclusion

The evidence supports **C3**: explainability is orchestrated independently of model implementations. `ExplanationEngine` coordinates prediction (via `ModelAdapter`) and explanation (via `MethodManager`) without HTTP or explainer implementations. Random Forest and Logistic Regression are selected by name through one adapter. SHAP and LIME execute only inside `MethodManager`. Successful V1 `/explain` calls show that this path is the one exercised at runtime.

This does not claim that API success alone proves separation; the claim rests on **source inspection and** those end-to-end results.

- Pass/Fail: **Pass**

---

## V4 — Model × Method Validation

**Claim:** C4 — Explainability can be model-agnostic.  
**Date:** 2026-09-17  
**Environment:** Frozen application `python -m uvicorn app.main:app --host 127.0.0.1 --port 8000` (Docker daemon was not running; same source and artifacts as V1–V3).  
**Input:** `artifacts/training_data.pkl` → `X_test[0]` (30 features). No retraining. No source-code changes between combinations.

### Objective

Show that the **same** frozen `POST /explain` contract and architectural path support both models and both explanation methods without structural change.

### Test matrix

| ID | Model | Method | Endpoint | Expected schema |
|---|---|---|---|---|
| V4.1 | random_forest | shap | POST /explain | model, method, prediction, probability, feature_importance |
| V4.2 | random_forest | lime | POST /explain | same |
| V4.3 | logistic_regression | shap | POST /explain | same |
| V4.4 | logistic_regression | lime | POST /explain | same |

### Test input (shared)

```json
{
  "model": "<varies>",
  "method": "<varies>",
  "input_data": [
    19.55, 28.77, 133.6, 1207.0, 0.0926, 0.2063, 0.1784, 0.1144, 0.1893, 0.06232,
    0.8426, 1.199, 7.158, 106.4, 0.006356, 0.04765, 0.03863, 0.01519, 0.01936, 0.005252,
    25.05, 36.27, 178.6, 1926.0, 0.1281, 0.5329, 0.4251, 0.1941, 0.2818, 0.1005
  ]
}
```

Routes present in `app/api/routes.py`: `POST /predict`, `POST /explain` only. No model-specific or method-specific explanation endpoints.

---

### V4.1 — random_forest + shap

- HTTP status: `200`
- `model`: `random_forest`
- `method`: `shap`
- `prediction`: `0`
- `probability`: `1.0`
- `feature_importance`: 30 numeric entries

```json
{
  "model": "random_forest",
  "method": "shap",
  "prediction": 0,
  "probability": 1.0,
  "feature_importance": {
    "worst concave points": 0.0939898161429568,
    "worst area": 0.08109447529100106,
    "worst radius": 0.055731161709292765,
    "mean concave points": 0.05514700419162046,
    "worst perimeter": 0.05125068485207676,
    "mean perimeter": 0.041712915472802696,
    "mean radius": 0.03863510040255016,
    "mean concavity": 0.038563861770973595,
    "mean area": 0.03588464548053283,
    "area error": 0.03468424356569953,
    "worst concavity": 0.02486326956991555,
    "radius error": 0.01766599605064866,
    "worst compactness": 0.01309379662663422,
    "worst texture": 0.013066358948923731,
    "perimeter error": 0.00969749910493148,
    "mean texture": 0.008168439585172884,
    "mean compactness": 0.00533976872873891,
    "worst smoothness": -0.0021260262935812214,
    "worst fractal dimension": 0.0018502791695782627,
    "texture error": 0.0018344582546897617,
    "compactness error": -0.001815399217655654,
    "concavity error": 0.0014996742323926111,
    "concave points error": 0.0012910572337011925,
    "symmetry error": 0.0009495188945498279,
    "fractal dimension error": 0.0007031282010600899,
    "mean fractal dimension": 0.0005418379613945962,
    "mean smoothness": 0.0005193152032181277,
    "smoothness error": 0.0004822762215586709,
    "mean symmetry": 0.00027982982003698417,
    "worst symmetry": -0.00011547069189882745
  }
}
```

- Result: **Pass**

---

### V4.2 — random_forest + lime

- HTTP status: `200`
- `model`: `random_forest`
- `method`: `lime`
- `prediction`: `0`
- `probability`: `1.0`
- `feature_importance`: 30 numeric entries

```json
{
  "model": "random_forest",
  "method": "lime",
  "prediction": 0,
  "probability": 1.0,
  "feature_importance": {
    "worst area": 0.14295949192757362,
    "worst concave points": 0.14140898821782125,
    "worst radius": 0.10483809341002351,
    "worst perimeter": 0.08843799761724062,
    "area error": 0.05248120294024532,
    "mean concave points": 0.047444109689006767,
    "mean radius": 0.04692460773478402,
    "worst texture": 0.046909225006062046,
    "mean perimeter": 0.04463507382242734,
    "worst concavity": 0.03879309264876107,
    "mean concavity": 0.0369951855967841,
    "mean area": 0.03390037049579456,
    "mean texture": 0.023889120062995584,
    "radius error": 0.023310828895641556,
    "perimeter error": 0.01984546857064807,
    "worst compactness": 0.016555559750294924,
    "fractal dimension error": -0.012860763925058671,
    "compactness error": -0.008527473431846339,
    "mean compactness": 0.006877857305043569,
    "concavity error": 0.005022848412348569,
    "worst symmetry": -0.0042371200736800495,
    "smoothness error": 0.004227247612199207,
    "symmetry error": 0.003148279121765801,
    "worst fractal dimension": 0.0018311083442295595,
    "worst smoothness": -0.0017360985561069775,
    "concave points error": -0.0010139377480292036,
    "mean fractal dimension": -0.00074407090720568,
    "mean symmetry": 0.0006813622342713298,
    "mean smoothness": 0.0006246876984975621,
    "texture error": -0.0005720228294644424
  }
}
```

- Result: **Pass**  
- Note: LIME numeric values may differ from V1.2 on the same instance because LIME is stochastic. Schema and endpoint are unchanged.

---

### V4.3 — logistic_regression + shap

- HTTP status: `200`
- `model`: `logistic_regression`
- `method`: `shap`
- `prediction`: `0`
- `probability`: `0.9999999999980681`
- `feature_importance`: 30 numeric entries

```json
{
  "model": "logistic_regression",
  "method": "shap",
  "prediction": 0,
  "probability": 0.9999999999980681,
  "feature_importance": {
    "worst area": -27.707552496848898,
    "mean radius": 10.91324749731411,
    "worst radius": 10.558734947905483,
    "worst perimeter": -7.281718661884001,
    "area error": -6.037003125372009,
    "worst texture": -3.199221814905269,
    "mean perimeter": -2.2078166572287197,
    "mean area": -0.952428601709524,
    "mean texture": 0.6743749863758655,
    "worst compactness": -0.3196215721967923,
    "worst concavity": -0.24457144531866382,
    "mean concavity": -0.056571083941258364,
    "perimeter error": -0.05318090321369739,
    "worst concave points": -0.050456849259310106,
    "mean compactness": -0.0394716585456803,
    "texture error": 0.024407204748117556,
    "mean concave points": -0.02092316964753116,
    "radius error": -0.006729053459782726,
    "mean symmetry": -0.003363210155527704,
    "worst symmetry": 0.002292073981388085,
    "worst fractal dimension": -0.001803348768782215,
    "worst smoothness": 0.00127039319773829,
    "mean smoothness": 0.00040595536576133355,
    "concavity error": -0.00038776143747283517,
    "compactness error": -0.0001438488957353102,
    "concave points error": -0.00014124755035262813,
    "smoothness error": 1.2905445788059113e-05,
    "symmetry error": 1.2028004421909083e-05,
    "fractal dimension error": 8.173807401235254e-06,
    "mean fractal dimension": 5.150374085683448e-06
  }
}
```

- Result: **Pass**

---

### V4.4 — logistic_regression + lime

- HTTP status: `200`
- `model`: `logistic_regression`
- `method`: `lime`
- `prediction`: `0`
- `probability`: `0.9999999999980681`
- `feature_importance`: 30 numeric entries

```json
{
  "model": "logistic_regression",
  "method": "lime",
  "prediction": 0,
  "probability": 0.9999999999980681,
  "feature_importance": {
    "worst area": 0.58747113542082,
    "mean radius": -0.31909420926246346,
    "worst radius": -0.27155782455675476,
    "area error": 0.20927749832341458,
    "worst perimeter": 0.17390303904175886,
    "worst texture": 0.10427595565541606,
    "mean perimeter": 0.06845623731841134,
    "mean area": 0.03623756396319702,
    "mean texture": -0.03616461450373735,
    "worst fractal dimension": -0.025938665158129783,
    "mean compactness": 0.02272811511862888,
    "mean smoothness": -0.014516252306904835,
    "worst concavity": 0.012880272805760762,
    "texture error": -0.012333749125028472,
    "worst concave points": 0.009685042297673507,
    "perimeter error": 0.009512166975234784,
    "worst symmetry": 0.009335636401013144,
    "mean concavity": 0.007008817477632721,
    "mean concave points": -0.006269399351282275,
    "fractal dimension error": -0.005526486468249973,
    "worst smoothness": -0.004740993093178946,
    "mean fractal dimension": 0.003744456427421612,
    "symmetry error": -0.0037133580466681584,
    "radius error": 0.003570828971399648,
    "concave points error": 0.003114301136590591,
    "concavity error": 0.0023803854508459487,
    "mean symmetry": -0.0016139610997194056,
    "compactness error": 0.0009106561208982063,
    "smoothness error": 0.0005632536069950854,
    "worst compactness": -7.082640611034392e-05
  }
}
```

- Result: **Pass**

---

### Response-schema comparison

| Combination | Endpoint | Request fields | Response fields | Status | FI count | Result |
|---|---|---|---|---|---|---|
| RF + SHAP | POST /explain | model, method, input_data | model, method, prediction, probability, feature_importance | 200 | 30 | Pass |
| RF + LIME | POST /explain | same | same | 200 | 30 | Pass |
| LR + SHAP | POST /explain | same | same | 200 | 30 | Pass |
| LR + LIME | POST /explain | same | same | 200 | 30 | Pass |

All four responses contain only the contract fields above (plus nested feature-name keys under `feature_importance`).

Feature **values** differ across model and method, as expected. They were not compared for equality or quality.

### Architectural observations

- Model and method were selected only via request parameters.
- No additional routes, engines, adapters, or managers were introduced between trials.
- V3 already established `/explain` → `ExplanationEngine` → `ModelAdapter` + `MethodManager` → SHAP/LIME. V4 exercises that path for both supported models and both methods.
- `POST /predict` remains a separate prediction endpoint; it is not an explanation route.

### Deviations / failures

None. All four combinations returned HTTP 200 and the published `/explain` schema.

### Evidence for C4

C4 is: “Explainability can be model-agnostic.”

This experiment shows that **Random Forest and Logistic Regression** can obtain SHAP and LIME explanations through the **same** frozen `/explain` API and the **same** orchestration path, without architectural modification. It does **not** show that arbitrary models, additional backends, or production-scale registries are supported.

### V4 Conclusion

C4 is **supported** at the scope of this prototype: RF+SHAP, RF+LIME, LR+SHAP, and LR+LIME all succeed on one endpoint and one response contract.

- Pass/Fail: **Pass**

---

## V5 — Error Handling Validation

**Role:** supporting reliability check (not a primary research claim).  
**Date:** 2026-09-17  
**Environment:** Frozen application at `http://127.0.0.1:8000` (`python -m uvicorn app.main:app`).  
**Endpoint:** `POST /explain` only.  
**Valid 30-feature baseline** (used where a full vector is required): `artifacts/training_data.pkl` → `X_test[0]`.

No application code, routes, Pydantic models, or error handlers were modified.

### Objective

Record how the frozen API rejects invalid `/explain` requests, and compare that behavior to `docs/API_CONTRACTS.md`.

### Documented expected behavior (`API_CONTRACTS.md`)

Error envelope (all endpoints):

```json
{
  "error": {
    "type": "...",
    "message": "..."
  }
}
```

Documented types: `InvalidModelError`, `InvalidMethodError`, `ValidationError`, `InternalServerError`.

The contract example for an unsupported model uses `type: "InvalidModelError"` and `message: "Unsupported model: svm"`.

`input_data` must be exactly 30 numerical values.

Service-layer code (`app/api/routes.py` `_error_response`) maps:

- `ValueError` starting with `"Unsupported model"` → HTTP **400** `InvalidModelError`
- `ValueError` starting with `"Unsupported explanation method"` → HTTP **400** `InvalidMethodError`

Those handlers run only if the request reaches the route body and a `ValueError` is raised. `ExplainRequest` uses `ModelEnum` and `MethodEnum`, and `input_data: list[float]` with `min_length=30, max_length=30`. Pydantic failures are handled in `app/main.py` as HTTP **422** `ValidationError` / `"Invalid request body."`

### Test cases and observed behavior

#### V5.1 — Unsupported model

- Request:

```json
{
  "model": "unsupported_model",
  "method": "shap",
  "input_data": [19.55, 28.77, 133.6, 1207.0, 0.0926, 0.2063, 0.1784, 0.1144, 0.1893, 0.06232, 0.8426, 1.199, 7.158, 106.4, 0.006356, 0.04765, 0.03863, 0.01519, 0.01936, 0.005252, 25.05, 36.27, 178.6, 1926.0, 0.1281, 0.5329, 0.4251, 0.1941, 0.2818, 0.1005]
}
```

- HTTP status: **422**
- Response body:

```json
{
  "error": {
    "type": "ValidationError",
    "message": "Invalid request body."
  }
}
```

- Error envelope `{error: {type, message}}`: **yes**
- Matches documented `InvalidModelError` example: **no**
- Layer: **Pydantic / FastAPI request validation** (`ModelEnum`). Route handler, `ExplanationEngine`, and `ModelAdapter` are not reached. Service-layer `InvalidModelError` is **not** raised.

#### V5.2 — Unsupported explanation method

- Request:

```json
{
  "model": "random_forest",
  "method": "unsupported_method",
  "input_data": [19.55, 28.77, 133.6, 1207.0, 0.0926, 0.2063, 0.1784, 0.1144, 0.1893, 0.06232, 0.8426, 1.199, 7.158, 106.4, 0.006356, 0.04765, 0.03863, 0.01519, 0.01936, 0.005252, 25.05, 36.27, 178.6, 1926.0, 0.1281, 0.5329, 0.4251, 0.1941, 0.2818, 0.1005]
}
```

- HTTP status: **422**
- Response body:

```json
{
  "error": {
    "type": "ValidationError",
    "message": "Invalid request body."
  }
}
```

- Error envelope: **yes**
- Matches documented `InvalidMethodError`: **no**
- Layer: **Pydantic / FastAPI request validation** (`MethodEnum`). `MethodManager.explain` is not called. Service-layer `InvalidMethodError` is **not** raised.

#### V5.3 — Wrong feature count (29 values)

- Request: same as a valid RF+SHAP body except `input_data` is the first **29** floats of `X_test[0]`.
- HTTP status: **422**
- Response body:

```json
{
  "error": {
    "type": "ValidationError",
    "message": "Invalid request body."
  }
}
```

- Error envelope: **yes**
- Matches documented `ValidationError` type: **yes** (type name). Message is generic (`Invalid request body.`), not `Expected 30 features.`
- Layer: **Pydantic** (`Field(min_length=30, max_length=30)`). `ModelAdapter._reshape` is not reached.

#### V5.4 — Non-numeric feature

- Request: 29 valid floats plus the string `"not-a-number"` (length 30, one invalid type).
- HTTP status: **422**
- Response body:

```json
{
  "error": {
    "type": "ValidationError",
    "message": "Invalid request body."
  }
}
```

- Error envelope: **yes**
- Matches documented `ValidationError` type: **yes** (type name); message remains generic.
- Layer: **Pydantic** (`list[float]`). Service layer not reached.

---

### Contract comparison

| Case | Rejects request? | Envelope `{error.type, error.message}`? | Documented example type | Observed type | Observed status | Layer |
|---|---|---|---|---|---|---|
| Unsupported model | Yes | Yes | `InvalidModelError` | `ValidationError` | 422 | Pydantic enum |
| Unsupported method | Yes | Yes | `InvalidMethodError` | `ValidationError` | 422 | Pydantic enum |
| Wrong feature count | Yes | Yes | `ValidationError` | `ValidationError` | 422 | Pydantic length |
| Non-numeric feature | Yes | Yes | `ValidationError` | `ValidationError` | 422 | Pydantic types |

### Deviations (implementation vs `API_CONTRACTS.md`)

1. **Unsupported model / method** are rejected at the **HTTP validation layer** with **422** `ValidationError` / `"Invalid request body."` They do **not** produce **400** `InvalidModelError` / `InvalidMethodError` or the example messages (`Unsupported model: …` / `Unsupported explanation method: …`), because those `ValueError`s never fire when the enum already rejects the payload.
2. **Wrong feature count** is also 422 `ValidationError` at Pydantic, not the adapter message `"Expected 30 features."`

These are **documentation-vs-implementation deviations**. They are **not** treated as a successful match to the `InvalidModelError` / `InvalidMethodError` examples merely because the API refused the request.

What **does** match the contract: invalid `/explain` bodies do not return HTTP 200; they use the `{error: {type, message}}` envelope; `ValidationError` is a documented error type.

### V5 Conclusion

Invalid `/explain` requests are **rejected** with a structured error JSON envelope. That is successful **failure-mode** behavior for a service API.

Relative to the **written** contract examples for unsupported model/method, the frozen implementation **deviates**: Pydantic enums intercept those cases as **422 ValidationError** before service-layer `InvalidModelError` / `InvalidMethodError` handlers. Feature-count and type errors are 422 `ValidationError`, which is consistent with the documented type name for validation failures.

V5 therefore records: **error-handling is operational (requests fail predictably with the envelope)** and **contract examples for InvalidModelError/InvalidMethodError are not exercised by these `/explain` payloads**. No code was changed to close that gap.

- Outcome: **Documented — deviations recorded; invalid requests did not succeed**

---

## Deviations

None observed for V1/V2 success paths relative to `docs/API_CONTRACTS.md` (required `/explain` success fields were present; HTTP 200).

V3: no forbidden `/explain` dependencies. `/predict` calling `ModelAdapter` directly is noted in V3.1/V3.5 as the existing prediction endpoint, not an explanation-orchestration defect.

V4: none. All four model × method combinations returned HTTP 200 with the frozen `/explain` schema.

V5: invalid `/explain` requests return **422** `{error: {type: "ValidationError", message: "Invalid request body."}}`. That uses the documented envelope and `ValidationError` type, but **does not match** the contract examples for `InvalidModelError` / `InvalidMethodError` on unsupported model/method, because Pydantic enums reject those payloads before the service-layer handlers run.
