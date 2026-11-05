# EaaS evaluation framework

This directory holds the **empirical evaluation** of the frozen Explainability-as-a-Service (EaaS) prototype. It is **not** part of the production application (`app/`).

Production architecture, public API contracts, models, and explanation backends must remain unchanged. Evaluation talks to the running service over HTTP (or inspects source for E3). It must not import SHAP, LIME, or application modules to “drive” explanations.

**Authoritative protocol:** [`EVALUATION_GUIDELINES.md`](../EVALUATION_GUIDELINES.md) at the repository root.

**Research question:** How can explainability be operationalized as a reusable service rather than being tied to an individual application or model?

This framework does **not** claim that experiments have already been executed. V1–V5 in `docs/VALIDATION_RESULTS.md` are preliminary validation, not E1–E4 results.

---

## Experiments

| ID | Purpose | Script |
|---|---|---|
| **E1** | Model–method compatibility and response-contract consistency: 4 combinations × 30 samples = **120** valid `POST /explain` requests (`X_test[0:30]`). | `scripts/run_e1_contract.py` |
| **E2** | Sequential end-to-end client latency: 2 sessions, 10 warm-up + 100 measured attempts per combination per session = **800** measured attempts (`X_test[0:10]` warm-up, `X_test[10:110]` measured). Failures count; no retries. | `scripts/run_e2_performance.py` |
| **E3** | Source-level architectural inspection of API, ExplanationEngine, ModelAdapter, MethodManager (no production edits). | not implemented yet |
| **E4** | Invalid-input handling: 6 categories × 3 = **18** `POST /explain` requests. Record actual status/body vs documented error types. | `scripts/run_e4_invalid_inputs.py` |

E4 has **not** been executed against Railway. Presence of the script is not experimental evidence. E1 evidence, if present, lives under `results/` and must not be overwritten.

### Running E1

The evaluation script uses **httpx**, which is listed in `requirements-dev.txt` (not in production `requirements.txt`). Install development dependencies before a live run:

```bash
pip install -r requirements-dev.txt
```

Importing the script does not send HTTP requests. The default invocation also does nothing.

```bash
python evaluation/scripts/run_e1_contract.py --help
python evaluation/scripts/run_e1_contract.py --offline-check
python evaluation/scripts/run_e1_contract.py --execute --confirm E1
```

`--execute --confirm E1` is the explicit live command. It sends **120** sequential `POST /explain` requests (4 combinations × 30 samples from `X_test[0:30]`) to `BASE_URL`. Each request uses the **120-second** client timeout from `HTTP_TIMEOUT_SECONDS`. There are **no automatic retries**; failed attempts remain in the 120-attempt total.

Raw results are opened **before** the first HTTP request and **appended and flushed after every attempt** to:

`evaluation/results/raw/e1_contract_YYYYMMDD_HHMMSS.jsonl`

The summary is written to:

`evaluation/results/summaries/e1_contract_YYYYMMDD_HHMMSS.json`

If the run is interrupted (Ctrl+C), completed JSONL rows are kept and a **partial** summary is written with `execution_status: interrupted` and `overall_outcome: incomplete`. That is **not** a completed E1 result. A finished run that fails HTTP or contract checks is `execution_status: completed` with `overall_outcome: not_fully_successful`. Only all 120 HTTP 200 and contract-valid responses yield `fully_successful`.

Do not run `--execute --confirm E1` unless a formal E1 session is intended.

Constants come from `evaluation/config.py`. Filenames are timestamped and must not be overwritten.

### Running E4

httpx is required (`pip install -r requirements-dev.txt`). Import and default invocation send no HTTP.

```bash
python evaluation/scripts/run_e4_invalid_inputs.py --help
python evaluation/scripts/run_e4_invalid_inputs.py --offline-check
python evaluation/scripts/run_e4_invalid_inputs.py --execute --confirm E4
```

`--execute --confirm E4` sends **18** sequential invalid `POST /explain` requests (6 categories × 3) to `BASE_URL`, with the **120-second** timeout and **no retries**. It records observed HTTP status and body; HTTP 4xx is not treated as an automatic pass. Malformed JSON is sent as raw bytes with `Content-Type: application/json`.

Raw JSONL is opened before the first request and flushed after every attempt:

`evaluation/results/raw/e4_invalid_inputs_YYYYMMDD_HHMMSS.jsonl`

Summary:

`evaluation/results/summaries/e4_invalid_inputs_YYYYMMDD_HHMMSS.json`

Interrupted runs keep completed JSONL rows and write `execution_status: interrupted` / `overall_outcome: incomplete`. That is not a completed E4 result. Filenames are allocated so existing E1/E4 files are never overwritten.

Do not run `--execute --confirm E4` unless a formal E4 session is intended.

### Running E2

httpx is required (`pip install -r requirements-dev.txt`). Import and default invocation send no HTTP. Sessions are **separate commands**; the script does not wait 30 minutes and start session 2 by itself.

```bash
python evaluation/scripts/run_e2_performance.py --help
python evaluation/scripts/run_e2_performance.py --offline-check
python evaluation/scripts/run_e2_performance.py --execute --session 1 --confirm E2
python evaluation/scripts/run_e2_performance.py --execute --session 2 --confirm E2 --session-1-summary evaluation/results/summaries/e2_performance_session1_<stamp>.json
```

Each session sends 4 combinations × (10 warm-up + 100 measured) = **440** sequential `POST /explain` requests, timeout **120 s**, **no retries**. Warm-up is recorded but excluded from latency statistics. Measured failures stay in the 100-attempt denominator.

Session 2 refuses to run unless the session-1 summary is completed, the matching raw JSONL has 40 warm-up and 400 measured rows, and at least **30 minutes** (UTC `session_end_timestamp_utc`) have elapsed.

Raw/summary names look like `e2_performance_session1_YYYYMMDD_HHMMSS.jsonl`. Existing files are never overwritten.

Do not run `--execute --confirm E2` unless a formal E2 session is intended.

---

## Directory structure

```
evaluation/
  README.md
  config.py                 # all experiment constants
  fixtures/                 # optional exported request vectors
  scripts/                  # HTTP clients / summarizers (`run_e1_contract.py`)
  results/raw/              # one row per attempt, including failures
  results/summaries/        # derived tables from raw files
  results/figures/          # optional plots from raw files
  architecture-audit/       # E3 notes
```

---

## Configuration

Import constants from `evaluation/config.py` (or `config.py` when running with this directory on `PYTHONPATH`). Do not duplicate slice indices, timeouts, or totals in scripts.

Examples: `BASE_URL`, `COMBINATIONS`, `E1_SLICE_START` / `E1_SLICE_END`, `HTTP_TIMEOUT_SECONDS`, `RAW_RESULTS_DIR`.

`config.py` does not import production code.

---

## Dataset and sampling

- Source: `artifacts/training_data.pkl` (`X_test` has 114 × 30 rows).
- Do not retrain models or reshuffle the original split.
- E1: `X_test[0:30]`.
- E2 warm-up: `X_test[0:10]` (excluded from statistics).
- E2 measured: `X_test[10:110]`.
- The same index is the same feature vector for every model–method pair.
- LIME values need not match across repeats; check structure and finite numerics.

---

## Evaluation target

`https://eaas-production-df33.up.railway.app/`

A successful local or Docker run does **not** substitute for public-deployment evidence unless the public service is unavailable and that fallback is recorded.

The deployed Git commit and Railway Python/runtime version are **unverified**. Record environment metadata with every run.

---

## Raw-result preservation

- Keep every attempted request, including HTTP errors, timeouts, and connection failures.
- Never fabricate, interpolate, retry, or replace failed E2 attempts.
- Derived statistics must be recomputable from raw files.
- Use timestamped filenames; do not overwrite raw outputs.
- Distinguish observation, interpretation, and limitation.

---

## Reproducibility

Each run should log at least: git commit, `BASE_URL`, client timeout (120 s for E2), Python version, date/time (UTC), session ids, combination order, and protocol deviations.

E2 P95: `np.percentile(latencies, 95, method="linear")` on HTTP 200 measured rows only; report **n**. Success/error rates use all 100 measured **attempts** as the denominator.

---

## Frozen production system

Do not modify `app/`, API contracts, artifacts, dependencies, Dockerfile, or Railway settings as part of evaluation. If implementation disagrees with `docs/API_CONTRACTS.md` (for example HTTP 422 `ValidationError` vs documented `InvalidModelError`), document the deviation.
