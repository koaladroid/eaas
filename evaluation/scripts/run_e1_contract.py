"""E1: Model–method compatibility and response-contract consistency.

Talks only to the running service via POST /explain. Does not import app,
SHAP, or LIME. Does not retrain models.

This module performs no HTTP I/O on import. Full execution requires
--execute --confirm E1.

Protocol: EVALUATION_GUIDELINES.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from typing import Any, TextIO

import joblib
import numpy as np

EVALUATION_DIR = Path(__file__).resolve().parent.parent
if str(EVALUATION_DIR) not in sys.path:
    sys.path.insert(0, str(EVALUATION_DIR))

import config  # noqa: E402

EXPERIMENT_ID = "E1"
REQUIRED_RESPONSE_KEYS = (
    "model",
    "method",
    "prediction",
    "probability",
    "feature_importance",
)
CONFIRM_TOKEN = "E1"
MAX_STORED_BODY_CHARS = 50_000
EVALUATION_DEPENDENCY_PACKAGES = ("httpx", "joblib", "numpy")
SHA256_CHUNK_SIZE = 1024 * 1024


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    return utc_now().strftime("%Y-%m-%dT%H:%M:%SZ")


RESULT_FILENAME_PREFIX = "e1_contract"


def stamp_for_filename(moment: datetime | None = None) -> str:
    """Default run id: second-level UTC stamp (YYYYMMDD_HHMMSS)."""
    moment = moment or utc_now()
    return moment.strftime("%Y%m%d_%H%M%S")


def result_paths_for_stamp(stamp: str) -> tuple[Path, Path]:
    raw_path = config.RAW_RESULTS_DIR / f"{RESULT_FILENAME_PREFIX}_{stamp}.jsonl"
    summary_path = config.SUMMARY_RESULTS_DIR / f"{RESULT_FILENAME_PREFIX}_{stamp}.json"
    return raw_path, summary_path


def _run_stamp_candidates(preferred_stamp: str) -> Any:
    yield preferred_stamp
    yield f"{preferred_stamp}_{utc_now().strftime('%f')}"
    suffix = 1
    while suffix <= 10_000:
        yield f"{preferred_stamp}_{suffix}"
        suffix += 1


def allocate_run_output_paths(preferred_stamp: str | None = None) -> tuple[str, Path, Path]:
    """Choose a stamp whose raw and summary paths do not already exist.

    Prefers the default second-level filename. On collision, uses a
    microsecond suffix, then numeric suffixes. Never reuses an occupied pair.
    """
    preferred_stamp = preferred_stamp or stamp_for_filename()
    config.RAW_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    config.SUMMARY_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for stamp in _run_stamp_candidates(preferred_stamp):
        raw_path, summary_path = result_paths_for_stamp(stamp)
        if raw_path.exists() or summary_path.exists():
            continue
        return stamp, raw_path, summary_path
    raise RuntimeError(
        "Unable to allocate unique E1 raw/summary filenames without overwrite."
    )


def combination_label(model: str, method: str) -> str:
    return f"{model}+{method}"


def git_head() -> str | None:
    """Local HEAD at execution time. Not the deployed Railway revision."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=config.PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    value = completed.stdout.strip()
    return value or None


def python_version() -> str:
    return sys.version.replace("\n", " ")


def os_platform_info() -> dict[str, str]:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python_implementation": platform.python_implementation(),
    }


def installed_evaluation_package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in EVALUATION_DEPENDENCY_PACKAGES:
        try:
            versions[name] = package_version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(SHA256_CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def is_finite_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


def is_json_integer(value: Any) -> bool:
    """True only for JSON integers. Rejects bool (a subclass of int) and floats."""
    return isinstance(value, int) and not isinstance(value, bool)


def load_e1_dataset(
    artifact_path: Path | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Load X_test[E1 slice] and feature names from the frozen artifact."""
    path = artifact_path or config.TRAINING_DATA_PATH
    payload = joblib.load(path)
    if "X_test" not in payload or "feature_names" not in payload:
        raise ValueError(
            "training_data.pkl must contain 'X_test' and 'feature_names'."
        )

    x_test = np.asarray(payload["X_test"])
    raw_names = payload["feature_names"]
    feature_names = [str(name) for name in list(raw_names)]

    if x_test.ndim != 2:
        raise ValueError(f"X_test must be 2-dimensional, got shape {x_test.shape}.")
    if x_test.shape[1] != config.FEATURE_COUNT:
        raise ValueError(
            f"Expected {config.FEATURE_COUNT} features per row, got {x_test.shape[1]}."
        )
    if x_test.shape[0] < config.E1_SLICE_END:
        raise ValueError(
            f"X_test has {x_test.shape[0]} rows; need at least {config.E1_SLICE_END}."
        )
    if len(feature_names) != config.FEATURE_COUNT:
        raise ValueError(
            f"Expected {config.FEATURE_COUNT} feature names, got {len(feature_names)}."
        )
    if len(set(feature_names)) != config.FEATURE_COUNT:
        raise ValueError("Feature names must be unique.")

    samples = x_test[config.E1_SLICE_START : config.E1_SLICE_END]
    if samples.shape != (config.E1_REQUESTS_PER_COMBINATION, config.FEATURE_COUNT):
        raise ValueError(f"Unexpected E1 slice shape: {samples.shape}.")
    if not np.isfinite(samples).all():
        raise ValueError("E1 input slice contains non-finite values.")
    if not np.issubdtype(samples.dtype, np.number):
        raise ValueError("E1 input slice must be numerical.")

    return samples, feature_names


def sample_as_input_data(row: np.ndarray) -> list[float]:
    values = [float(v) for v in np.asarray(row).tolist()]
    if len(values) != config.FEATURE_COUNT:
        raise ValueError("Serialized input_data does not have 30 features.")
    if not all(math.isfinite(v) for v in values):
        raise ValueError("Serialized input_data contains non-finite values.")
    return values


def validate_contract(
    body: Any,
    expected_model: str,
    expected_method: str,
    feature_names: list[str],
) -> tuple[bool, list[str]]:
    """Return (ok, failure codes) for an HTTP 200 JSON body."""
    failures: list[str] = []

    if not isinstance(body, dict):
        return False, ["response_not_json_object"]

    for key in REQUIRED_RESPONSE_KEYS:
        if key not in body:
            failures.append(f"missing_key:{key}")

    if "model" in body and body["model"] != expected_model:
        failures.append("model_mismatch")
    if "method" in body and body["method"] != expected_method:
        failures.append("method_mismatch")

    if "prediction" in body and not is_json_integer(body["prediction"]):
        failures.append("prediction_not_integer")

    if "probability" in body:
        probability = body["probability"]
        if not is_finite_number(probability):
            failures.append("probability_not_finite_numeric")
        elif not (0.0 <= float(probability) <= 1.0):
            failures.append("probability_out_of_unit_interval")

    if "feature_importance" not in body:
        return (len(failures) == 0), failures

    importance = body["feature_importance"]
    if not isinstance(importance, dict):
        failures.append("feature_importance_not_dict")
        return False, failures

    if len(importance) != config.FEATURE_COUNT:
        failures.append(
            f"feature_importance_count:{len(importance)}!={config.FEATURE_COUNT}"
        )

    expected_set = set(feature_names)
    observed_set = {str(key) for key in importance.keys()}
    missing = sorted(expected_set - observed_set)
    extra = sorted(observed_set - expected_set)
    if missing:
        failures.append("feature_importance_missing_names")
    if extra:
        failures.append("feature_importance_unexpected_names")

    non_finite = [
        str(key)
        for key, value in importance.items()
        if not is_finite_number(value)
    ]
    if non_finite:
        failures.append("feature_importance_non_finite_values")

    return (len(failures) == 0), failures


def _clip_text(text: str) -> str:
    if len(text) <= MAX_STORED_BODY_CHARS:
        return text
    return text[:MAX_STORED_BODY_CHARS] + "...[truncated]"


def parse_response_body(text: str) -> tuple[Any | None, str | None]:
    try:
        return json.loads(text), None
    except json.JSONDecodeError as exc:
        return None, str(exc)


def classify_attempt(
    *,
    transport_error: str | None,
    http_status: int | None,
    json_error: str | None,
    contract_failures: list[str],
) -> str | None:
    if transport_error:
        return "transport"
    if http_status is None or http_status != 200:
        return "http"
    if json_error:
        return "json_parse"
    if contract_failures:
        return "contract"
    return None


def explain_url() -> str:
    return f"{config.BASE_URL.rstrip('/')}{config.EXPLAIN_PATH}"


def post_explain(client: Any, payload: dict[str, Any]) -> Any:
    """Single POST /explain. Callers must not retry."""
    return client.post(explain_url(), json=payload)


class RawJsonlWriter:
    """Append-only JSONL writer. Creates the file on enter; flushes every record."""

    def __init__(self, path: Path):
        self.path = path
        self._handle: TextIO | None = None
        self.records_written = 0

    def __enter__(self) -> RawJsonlWriter:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("x", encoding="utf-8")
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        return False

    def append(self, record: dict[str, Any]) -> None:
        if self._handle is None:
            raise RuntimeError("RawJsonlWriter is not open.")
        self._handle.write(json.dumps(record, default=str) + "\n")
        self._handle.flush()
        self.records_written += 1


def execute_one_request(
    *,
    client: Any,
    model: str,
    method: str,
    sample_index: int,
    input_data: list[float],
    feature_names: list[str],
) -> dict[str, Any]:
    payload = {
        "model": model,
        "method": method,
        "input_data": input_data,
    }

    http_status: int | None = None
    response_text = ""
    parsed_body: Any | None = None
    json_error: str | None = None
    transport_error: str | None = None
    contract_failures: list[str] = []

    try:
        response = post_explain(client, payload)
        http_status = int(response.status_code)
        response_text = response.text or ""
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        transport_error = f"{type(exc).__name__}: {exc}"

    if transport_error is None:
        parsed_body, json_error = parse_response_body(response_text)
        if http_status == 200 and json_error is None:
            _, contract_failures = validate_contract(
                parsed_body, model, method, feature_names
            )

    error_category = classify_attempt(
        transport_error=transport_error,
        http_status=http_status,
        json_error=json_error,
        contract_failures=contract_failures,
    )
    request_success = transport_error is None and http_status == 200
    contract_valid = request_success and json_error is None and not contract_failures

    stored_body: Any
    if parsed_body is not None:
        stored_body = parsed_body
    elif response_text:
        stored_body = _clip_text(response_text)
    else:
        stored_body = None

    prediction = None
    probability = None
    fi_count = None
    if isinstance(parsed_body, dict):
        prediction = parsed_body.get("prediction")
        probability = parsed_body.get("probability")
        importance = parsed_body.get("feature_importance")
        if isinstance(importance, dict):
            fi_count = len(importance)

    error_message = transport_error or json_error
    if error_category == "http" and http_status is not None:
        error_message = f"HTTP {http_status}"
    if error_category == "contract":
        error_message = ";".join(contract_failures)

    return {
        "experiment_id": EXPERIMENT_ID,
        "timestamp_utc": utc_now_iso(),
        "combination": combination_label(model, method),
        "model": model,
        "method": method,
        "sample_index": sample_index,
        "http_status": http_status,
        "request_success": request_success,
        "contract_valid": contract_valid,
        "validation_failures": list(contract_failures),
        "prediction": prediction,
        "probability": probability,
        "feature_importance_count": fi_count,
        "response_body": stored_body,
        "error_category": error_category,
        "error_message": error_message,
    }


def run_e1(
    client: Any,
    samples: np.ndarray,
    feature_names: list[str],
    raw_writer: RawJsonlWriter,
) -> tuple[list[dict[str, Any]], bool]:
    """Run sequential attempts. Returns (records, interrupted).

    KeyboardInterrupt is handled here so completed rows already flushed to
    JSONL are summarized. Other exceptions propagate.
    """
    records: list[dict[str, Any]] = []
    interrupted = False
    try:
        for model, method in config.COMBINATIONS:
            for offset, row in enumerate(samples):
                sample_index = config.E1_SLICE_START + offset
                record = execute_one_request(
                    client=client,
                    model=model,
                    method=method,
                    sample_index=sample_index,
                    input_data=sample_as_input_data(row),
                    feature_names=feature_names,
                )
                records.append(record)
                raw_writer.append(record)
    except KeyboardInterrupt:
        interrupted = True
    return records, interrupted


def _combination_breakdown(records: list[dict[str, Any]]) -> dict[str, Any]:
    breakdown: dict[str, Any] = {}
    for model, method in config.COMBINATIONS:
        label = combination_label(model, method)
        subset = [row for row in records if row["combination"] == label]
        n = len(subset)
        http_ok = sum(1 for row in subset if row["request_success"])
        contract_ok = sum(1 for row in subset if row["contract_valid"])
        planned = config.E1_REQUESTS_PER_COMBINATION
        breakdown[label] = {
            "model": model,
            "method": method,
            "planned_attempts": planned,
            "attempts": n,
            "http_200": http_ok,
            "contract_valid": contract_ok,
            "failed_requests": n - http_ok,
            "http_success_rate": (http_ok / planned) if planned else 0.0,
            "contract_conformance_rate": (contract_ok / planned) if planned else 0.0,
        }
    return breakdown


def build_summary(
    records: list[dict[str, Any]],
    metadata: dict[str, Any],
    *,
    interrupted: bool = False,
) -> dict[str, Any]:
    completed = len(records)
    planned = config.E1_TOTAL_REQUESTS
    http_200 = sum(1 for row in records if row["request_success"])
    contract_valid = sum(1 for row in records if row["contract_valid"])
    failed_requests = completed - http_200
    http_errors = sum(1 for row in records if row["error_category"] == "http")
    transport_errors = sum(1 for row in records if row["error_category"] == "transport")
    json_errors = sum(1 for row in records if row["error_category"] == "json_parse")
    contract_errors = sum(1 for row in records if row["error_category"] == "contract")

    fully_successful = (
        not interrupted
        and completed == planned
        and http_200 == planned
        and contract_valid == planned
    )
    if interrupted:
        execution_status = "interrupted"
        overall_outcome = "incomplete"
    elif fully_successful:
        execution_status = "completed"
        overall_outcome = "fully_successful"
    else:
        execution_status = "completed"
        overall_outcome = "not_fully_successful"

    return {
        "experiment_id": EXPERIMENT_ID,
        "execution_status": execution_status,
        "overall_outcome": overall_outcome,
        "fully_successful": fully_successful,
        "success_criterion": (
            "All 120 requests must return HTTP 200 and pass every contract check."
        ),
        "counts": {
            "planned_attempts": planned,
            "attempts_completed": completed,
            "successful_http_responses": http_200,
            "contract_valid_responses": contract_valid,
            "failed_requests": failed_requests,
            "http_error_count": http_errors,
            "transport_error_count": transport_errors,
            "json_parsing_error_count": json_errors,
            "contract_validation_failure_count": contract_errors,
        },
        "rates": {
            "success_rate": (http_200 / planned) if planned else 0.0,
            "contract_conformance_rate": (contract_valid / planned) if planned else 0.0,
            "denominator": planned,
        },
        "by_combination": _combination_breakdown(records),
        "metadata": metadata,
    }


def write_summary(summary: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(summary, indent=2, default=str) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(payload)
    return path


def build_run_metadata(
    *,
    feature_names: list[str],
    n_samples: int,
    run_stamp: str,
    artifact_sha256: str,
) -> dict[str, Any]:
    return {
        "experiment_timestamp_utc": utc_now_iso(),
        "run_stamp": run_stamp,
        "base_url": config.BASE_URL,
        "explain_url": explain_url(),
        "git_head_local": git_head(),
        "deployed_git_revision": "unverified",
        "python_version": python_version(),
        "os": os_platform_info(),
        "installed_package_versions": installed_evaluation_package_versions(),
        "http_client": "httpx",
        "http_timeout_seconds": config.HTTP_TIMEOUT_SECONDS,
        "retries": False,
        "parallel": False,
        "dataset_artifact_path": str(config.TRAINING_DATA_PATH),
        "dataset_artifact_sha256": artifact_sha256,
        "n_samples": n_samples,
        "sample_index_start": config.E1_SLICE_START,
        "sample_index_end_exclusive": config.E1_SLICE_END,
        "feature_count": config.FEATURE_COUNT,
        "feature_names": feature_names,
        "combinations": [combination_label(m, h) for m, h in config.COMBINATIONS],
        "expected_total_requests": config.E1_TOTAL_REQUESTS,
    }


def _valid_body(feature_names: list[str]) -> dict[str, Any]:
    return {
        "model": config.MODEL_RANDOM_FOREST,
        "method": config.METHOD_SHAP,
        "prediction": 1,
        "probability": 0.87,
        "feature_importance": {name: 0.1 for name in feature_names},
    }


def run_offline_checks() -> int:
    """Load the artifact and exercise contract/IO checks with mocked HTTP."""
    import tempfile

    samples, feature_names = load_e1_dataset()
    assert samples.shape[0] == config.E1_REQUESTS_PER_COMBINATION
    assert len(feature_names) == config.FEATURE_COUNT
    assert config.E1_TOTAL_REQUESTS == 120

    artifact_sha256 = sha256_file(config.TRAINING_DATA_PATH)
    assert len(artifact_sha256) == 64
    int(artifact_sha256, 16)

    valid_body = _valid_body(feature_names)
    ok, failures = validate_contract(
        valid_body,
        config.MODEL_RANDOM_FOREST,
        config.METHOD_SHAP,
        feature_names,
    )
    assert ok and not failures, failures

    ok, failures = validate_contract(
        {**valid_body, "model": config.MODEL_LOGISTIC_REGRESSION},
        config.MODEL_RANDOM_FOREST,
        config.METHOD_SHAP,
        feature_names,
    )
    assert not ok and "model_mismatch" in failures

    missing = dict(valid_body)
    del missing["probability"]
    ok, failures = validate_contract(
        missing,
        config.MODEL_RANDOM_FOREST,
        config.METHOD_SHAP,
        feature_names,
    )
    assert not ok and "missing_key:probability" in failures

    ok, failures = validate_contract(
        {**valid_body, "prediction": "malignant"},
        config.MODEL_RANDOM_FOREST,
        config.METHOD_SHAP,
        feature_names,
    )
    assert not ok and "prediction_not_integer" in failures

    ok, failures = validate_contract(
        {**valid_body, "prediction": 1.0},
        config.MODEL_RANDOM_FOREST,
        config.METHOD_SHAP,
        feature_names,
    )
    assert not ok and "prediction_not_integer" in failures

    ok, failures = validate_contract(
        {**valid_body, "prediction": True},
        config.MODEL_RANDOM_FOREST,
        config.METHOD_SHAP,
        feature_names,
    )
    assert not ok and "prediction_not_integer" in failures

    ok, failures = validate_contract(
        {**valid_body, "probability": 1.5},
        config.MODEL_RANDOM_FOREST,
        config.METHOD_SHAP,
        feature_names,
    )
    assert not ok and "probability_out_of_unit_interval" in failures

    short_fi = {name: 0.1 for name in feature_names[:-1]}
    ok, failures = validate_contract(
        {**valid_body, "feature_importance": short_fi},
        config.MODEL_RANDOM_FOREST,
        config.METHOD_SHAP,
        feature_names,
    )
    assert not ok
    assert any(item.startswith("feature_importance_count:") for item in failures)
    assert "feature_importance_missing_names" in failures

    transport_record_category = classify_attempt(
        transport_error="TimeoutException: timed out",
        http_status=None,
        json_error=None,
        contract_failures=[],
    )
    assert transport_record_category == "transport"

    class FakeResp:
        def __init__(self, code: int, body: str):
            self.status_code = code
            self.text = body

    class CountingClient:
        def __init__(self, interrupt_after: int | None = None):
            self.calls = 0
            self.interrupt_after = interrupt_after

        def post(self, url: str, json: dict[str, Any] | None = None) -> FakeResp:
            self.calls += 1
            if self.interrupt_after is not None and self.calls > self.interrupt_after:
                raise KeyboardInterrupt
            assert "/explain" in url
            body = {
                "model": json["model"],
                "method": json["method"],
                "prediction": 0,
                "probability": 0.5,
                "feature_importance": {name: 0.01 for name in feature_names},
            }
            return FakeResp(200, json_module.dumps(body))

    import json as json_module

    with tempfile.TemporaryDirectory() as tmp:
        raw_path = Path(tmp) / "e1_offline.jsonl"
        client = CountingClient()
        with RawJsonlWriter(raw_path) as writer:
            assert raw_path.exists()
            assert raw_path.stat().st_size == 0
            records, interrupted = run_e1(client, samples, feature_names, writer)
        assert interrupted is False
        assert client.calls == config.E1_TOTAL_REQUESTS
        assert len(records) == config.E1_TOTAL_REQUESTS
        assert writer.records_written == config.E1_TOTAL_REQUESTS
        lines = raw_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == config.E1_TOTAL_REQUESTS
        summary = build_summary(records, {"n_samples": 30}, interrupted=False)
        assert summary["execution_status"] == "completed"
        assert summary["overall_outcome"] == "fully_successful"
        assert summary["fully_successful"] is True
        assert summary["counts"]["planned_attempts"] == 120
        assert summary["counts"]["attempts_completed"] == 120

        interrupt_client = CountingClient(interrupt_after=5)
        partial_path = Path(tmp) / "e1_partial.jsonl"
        with RawJsonlWriter(partial_path) as writer:
            partial_records, interrupted = run_e1(
                interrupt_client, samples, feature_names, writer
            )
        assert interrupted is True
        assert len(partial_records) == 5
        assert interrupt_client.calls == 6
        assert writer.records_written == 5
        partial_lines = partial_path.read_text(encoding="utf-8").splitlines()
        assert len(partial_lines) == 5
        partial_summary = build_summary(
            partial_records, {"n_samples": 30}, interrupted=True
        )
        assert partial_summary["execution_status"] == "interrupted"
        assert partial_summary["overall_outcome"] == "incomplete"
        assert partial_summary["fully_successful"] is False
        assert partial_summary["counts"]["planned_attempts"] == 120
        assert partial_summary["counts"]["attempts_completed"] == 5
        assert partial_summary["counts"]["successful_http_responses"] == 5
        assert partial_summary["rates"]["denominator"] == 120

        completed_failed = build_summary(
            [{**row, "contract_valid": i != 0} for i, row in enumerate(records)],
            {"n_samples": 30},
            interrupted=False,
        )
        assert completed_failed["execution_status"] == "completed"
        assert completed_failed["overall_outcome"] == "not_fully_successful"
        assert completed_failed["fully_successful"] is False

        saved_raw_dir = config.RAW_RESULTS_DIR
        saved_summary_dir = config.SUMMARY_RESULTS_DIR
        try:
            config.RAW_RESULTS_DIR = Path(tmp) / "raw_results"
            config.SUMMARY_RESULTS_DIR = Path(tmp) / "summary_results"
            config.RAW_RESULTS_DIR.mkdir()
            config.SUMMARY_RESULTS_DIR.mkdir()

            free_stamp = "20260101_120001"
            allocated_stamp, allocated_raw, allocated_summary = (
                allocate_run_output_paths(free_stamp)
            )
            assert allocated_stamp == free_stamp
            assert allocated_raw == result_paths_for_stamp(free_stamp)[0]
            assert allocated_summary == result_paths_for_stamp(free_stamp)[1]
            assert not allocated_raw.exists()
            assert not allocated_summary.exists()

            occupied_stamp = "20260101_120000"
            occupied_raw, occupied_summary = result_paths_for_stamp(occupied_stamp)
            occupied_raw.write_text("EXISTING_RAW\n", encoding="utf-8")
            occupied_summary.write_text("EXISTING_SUMMARY\n", encoding="utf-8")

            unique_stamp, unique_raw, unique_summary = allocate_run_output_paths(
                occupied_stamp
            )
            assert unique_stamp != occupied_stamp
            assert unique_stamp.startswith(occupied_stamp)
            assert unique_raw.name == f"{RESULT_FILENAME_PREFIX}_{unique_stamp}.jsonl"
            assert unique_summary.name == f"{RESULT_FILENAME_PREFIX}_{unique_stamp}.json"
            assert not unique_raw.exists()
            assert not unique_summary.exists()
            assert occupied_raw.read_text(encoding="utf-8") == "EXISTING_RAW\n"
            assert occupied_summary.read_text(encoding="utf-8") == "EXISTING_SUMMARY\n"

            with RawJsonlWriter(unique_raw) as collision_writer:
                collision_writer.append({"probe": True})
            write_summary({"probe": True}, unique_summary)
            assert unique_raw.exists()
            assert unique_summary.exists()
            assert occupied_raw.read_text(encoding="utf-8") == "EXISTING_RAW\n"
            assert occupied_summary.read_text(encoding="utf-8") == "EXISTING_SUMMARY\n"

            try:
                with RawJsonlWriter(occupied_raw):
                    raise AssertionError("RawJsonlWriter overwrote an existing file.")
            except FileExistsError:
                pass
            assert occupied_raw.read_text(encoding="utf-8") == "EXISTING_RAW\n"

            try:
                write_summary({"must_not_write": True}, occupied_summary)
                raise AssertionError("write_summary overwrote an existing file.")
            except FileExistsError:
                pass
            assert occupied_summary.read_text(encoding="utf-8") == "EXISTING_SUMMARY\n"
        finally:
            config.RAW_RESULTS_DIR = saved_raw_dir
            config.SUMMARY_RESULTS_DIR = saved_summary_dir

    print("offline-check: dataset load OK")
    print(
        f"offline-check: X_test E1 slice shape {tuple(samples.shape)}; "
        f"{len(feature_names)} feature names"
    )
    print(f"offline-check: artifact sha256 {artifact_sha256}")
    print("offline-check: mocked contract validation OK")
    print("offline-check: incremental JSONL and KeyboardInterrupt OK")
    print("offline-check: filename collision / no-overwrite OK")
    print("offline-check: no live HTTP client constructed")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "E1 contract evaluation: 4 model–method combinations × 30 samples "
            f"= {config.E1_TOTAL_REQUESTS} sequential POST /explain requests."
        )
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run the full E1 HTTP experiment (requires --confirm E1).",
    )
    parser.add_argument(
        "--confirm",
        metavar="TOKEN",
        help=f"Safety token. Must be '{CONFIRM_TOKEN}' together with --execute.",
    )
    parser.add_argument(
        "--offline-check",
        action="store_true",
        help="Load the dataset and run mocked contract checks. No HTTP.",
    )
    return parser.parse_args(argv)


def execute_experiment() -> int:
    import httpx

    samples, feature_names = load_e1_dataset()
    artifact_sha256 = sha256_file(config.TRAINING_DATA_PATH)
    run_stamp, raw_path, summary_path = allocate_run_output_paths(
        stamp_for_filename()
    )
    metadata = build_run_metadata(
        feature_names=feature_names,
        n_samples=int(samples.shape[0]),
        run_stamp=run_stamp,
        artifact_sha256=artifact_sha256,
    )
    timeout = httpx.Timeout(config.HTTP_TIMEOUT_SECONDS)
    with RawJsonlWriter(raw_path) as writer:
        with httpx.Client(timeout=timeout) as client:
            records, interrupted = run_e1(client, samples, feature_names, writer)
    summary = build_summary(records, metadata, interrupted=interrupted)
    write_summary(summary, summary_path)
    print(f"raw: {raw_path}")
    print(f"summary: {summary_path}")
    print(f"execution_status: {summary['execution_status']}")
    print(f"overall_outcome: {summary['overall_outcome']}")
    print(
        f"attempts_completed: {summary['counts']['attempts_completed']}/"
        f"{summary['counts']['planned_attempts']}"
    )
    if interrupted:
        return 130
    return 0 if summary["fully_successful"] else 1


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.offline_check:
        return run_offline_checks()
    if args.execute:
        if args.confirm != CONFIRM_TOKEN:
            print(
                "Refusing to run E1: pass --execute --confirm E1 "
                "to send 120 sequential requests to the public evaluation URL.",
                file=sys.stderr,
            )
            return 2
        return execute_experiment()
    print(
        "No action taken. Use --help, --offline-check, "
        "or --execute --confirm E1 to run the full experiment."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
