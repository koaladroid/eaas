"""E2: Sequential client-observed runtime performance.

Talks only to the running service via POST /explain. Does not import app,
SHAP, or LIME. Does not retrain models.

Sessions are independent CLI invocations. Session 2 requires a completed
session-1 summary and a UTC gap of at least 30 minutes.

This module performs no HTTP I/O on import. Live execution requires
--execute --confirm E2.

Protocol: EVALUATION_GUIDELINES.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from typing import Any, Iterator, TextIO

import joblib
import numpy as np

EVALUATION_DIR = Path(__file__).resolve().parent.parent
if str(EVALUATION_DIR) not in sys.path:
    sys.path.insert(0, str(EVALUATION_DIR))

import config  # noqa: E402

EXPERIMENT_ID = "E2"
CONFIRM_TOKEN = "E2"
MAX_STORED_BODY_CHARS = 50_000
EVALUATION_DEPENDENCY_PACKAGES = ("httpx", "joblib", "numpy")
RESULT_FILENAME_PREFIX = "e2_performance"
SHA256_CHUNK_SIZE = 1024 * 1024
WARMUP_PER_SESSION = len(config.COMBINATIONS) * config.E2_WARMUP_PER_COMBINATION
MEASURED_PER_SESSION = (
    len(config.COMBINATIONS) * config.E2_MEASURED_ATTEMPTS_PER_COMBINATION_PER_SESSION
)
ATTEMPTS_PER_SESSION = WARMUP_PER_SESSION + MEASURED_PER_SESSION


class Session1ValidationError(ValueError):
    """Session 2 may not start: session-1 evidence is missing or invalid."""


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    return utc_now().strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_utc(value: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    moment = datetime.fromisoformat(text)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def stamp_for_filename(moment: datetime | None = None) -> str:
    moment = moment or utc_now()
    return moment.strftime("%Y%m%d_%H%M%S")


def combination_label(model: str, method: str) -> str:
    return f"{model}+{method}"


def result_paths_for_stamp(session_number: int, stamp: str) -> tuple[Path, Path]:
    name = f"{RESULT_FILENAME_PREFIX}_session{session_number}_{stamp}"
    raw_path = config.RAW_RESULTS_DIR / f"{name}.jsonl"
    summary_path = config.SUMMARY_RESULTS_DIR / f"{name}.json"
    return raw_path, summary_path


def _run_stamp_candidates(preferred_stamp: str) -> Iterator[str]:
    yield preferred_stamp
    yield f"{preferred_stamp}_{utc_now().strftime('%f')}"
    suffix = 1
    while suffix <= 10_000:
        yield f"{preferred_stamp}_{suffix}"
        suffix += 1


def allocate_run_output_paths(
    session_number: int,
    preferred_stamp: str | None = None,
) -> tuple[str, Path, Path]:
    preferred_stamp = preferred_stamp or stamp_for_filename()
    config.RAW_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    config.SUMMARY_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for stamp in _run_stamp_candidates(preferred_stamp):
        raw_path, summary_path = result_paths_for_stamp(session_number, stamp)
        if raw_path.exists() or summary_path.exists():
            continue
        return stamp, raw_path, summary_path
    raise RuntimeError(
        "Unable to allocate unique E2 raw/summary filenames without overwrite."
    )


def git_head() -> str | None:
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


def explain_url() -> str:
    return f"{config.BASE_URL.rstrip('/')}{config.EXPLAIN_PATH}"


def session_combination_orders(
    seed: int = config.E2_RANDOMIZATION_SEED,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Deterministic per-session combination orders. Sample order is not shuffled."""
    rng = random.Random(seed)
    order_1 = list(config.COMBINATIONS)
    rng.shuffle(order_1)
    order_2 = list(config.COMBINATIONS)
    rng.shuffle(order_2)
    while order_2 == order_1:
        rng.shuffle(order_2)
    return order_1, order_2


def combination_order_for_session(session_number: int) -> list[tuple[str, str]]:
    order_1, order_2 = session_combination_orders()
    if session_number == 1:
        return order_1
    if session_number == 2:
        return order_2
    raise ValueError(f"Unsupported E2 session: {session_number}")


def load_e2_dataset(
    artifact_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    path = artifact_path or config.TRAINING_DATA_PATH
    payload = joblib.load(path)
    x_test = np.asarray(payload["X_test"])
    if x_test.ndim != 2 or x_test.shape[1] != config.FEATURE_COUNT:
        raise ValueError(f"Unexpected X_test shape: {x_test.shape}.")
    if x_test.shape[0] < config.MEASURED_SLICE_END:
        raise ValueError(
            f"X_test has {x_test.shape[0]} rows; need at least {config.MEASURED_SLICE_END}."
        )
    feature_names = [str(name) for name in list(payload["feature_names"])]
    warmup = x_test[config.WARMUP_SLICE_START : config.WARMUP_SLICE_END]
    measured = x_test[config.MEASURED_SLICE_START : config.MEASURED_SLICE_END]
    if warmup.shape[0] != config.E2_WARMUP_PER_COMBINATION:
        raise ValueError(f"Unexpected warm-up slice: {warmup.shape}.")
    if measured.shape[0] != config.E2_MEASURED_ATTEMPTS_PER_COMBINATION_PER_SESSION:
        raise ValueError(f"Unexpected measured slice: {measured.shape}.")
    if not np.isfinite(warmup).all() or not np.isfinite(measured).all():
        raise ValueError("E2 input slices contain non-finite values.")
    return warmup, measured, feature_names


def sample_as_input_data(row: np.ndarray) -> list[float]:
    values = [float(v) for v in np.asarray(row).tolist()]
    if len(values) != config.FEATURE_COUNT:
        raise ValueError("Serialized input_data does not have 30 features.")
    return values


def _clip_text(text: str) -> str:
    if len(text) <= MAX_STORED_BODY_CHARS:
        return text
    return text[:MAX_STORED_BODY_CHARS] + "...[truncated]"


def parse_response_body(text: str) -> tuple[Any | None, str | None]:
    try:
        return json.loads(text), None
    except json.JSONDecodeError as exc:
        return None, str(exc)


def classify_http_outcome(
    *,
    transport_error: str | None,
    exception_name: str | None,
    http_status: int | None,
    json_error: str | None,
) -> str:
    if transport_error is not None:
        name = exception_name or ""
        if "Timeout" in name:
            return "timeout"
        if "Connect" in name:
            return "connection_error"
        return "transport_error"
    if http_status is None:
        return "transport_error"
    if http_status == 200 and json_error:
        return "json_parse_error"
    if http_status == 200:
        return "http_200"
    if 400 <= http_status <= 499:
        return "http_client_error"
    if 500 <= http_status <= 599:
        return "http_server_error"
    return "unexpected_http_status"


class RawJsonlWriter:
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


def post_explain(client: Any, payload: dict[str, Any]) -> Any:
    """Single POST /explain. Callers must not retry."""
    return client.post(explain_url(), json=payload)


def execute_one_attempt(
    *,
    client: Any,
    session_number: int,
    phase: str,
    model: str,
    method: str,
    sample_index: int,
    attempt_number: int,
    session_attempt_index: int,
    input_data: list[float],
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
    exception_name: str | None = None
    start_utc = utc_now_iso()
    started = time.perf_counter()
    try:
        response = post_explain(client, payload)
        http_status = int(response.status_code)
        response_text = response.text or ""
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        exception_name = type(exc).__name__
        transport_error = f"{exception_name}: {exc}"
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    end_utc = utc_now_iso()

    if transport_error is None:
        parsed_body, json_error = parse_response_body(response_text)

    outcome = classify_http_outcome(
        transport_error=transport_error,
        exception_name=exception_name,
        http_status=http_status,
        json_error=json_error,
    )
    http_200 = transport_error is None and http_status == 200
    # Protocol: latency uses measured HTTP 200 only. JSON parse errors stay diagnostic.
    latency_eligible = http_200

    stored_body: Any
    if parsed_body is not None:
        stored_body = parsed_body
    elif response_text:
        stored_body = _clip_text(response_text)
    else:
        stored_body = None

    return {
        "experiment_id": EXPERIMENT_ID,
        "session_number": session_number,
        "phase": phase,
        "combination": combination_label(model, method),
        "model": model,
        "method": method,
        "sample_index": sample_index,
        "attempt_number": attempt_number,
        "session_attempt_index": session_attempt_index,
        "start_timestamp_utc": start_utc,
        "end_timestamp_utc": end_utc,
        "latency_ms": elapsed_ms,
        "outcome": outcome,
        "http_status": http_status,
        "http_200": http_200,
        "latency_eligible": latency_eligible,
        "transport_error": transport_error,
        "response_json_parse_error": json_error,
        "response_body": stored_body,
        "retries": False,
        "parallel": False,
    }


def run_e2_session(
    client: Any,
    *,
    session_number: int,
    warmup_rows: np.ndarray,
    measured_rows: np.ndarray,
    raw_writer: RawJsonlWriter,
) -> tuple[list[dict[str, Any]], bool]:
    records: list[dict[str, Any]] = []
    interrupted = False
    session_attempt_index = 0
    try:
        for model, method in combination_order_for_session(session_number):
            for offset, row in enumerate(warmup_rows):
                session_attempt_index += 1
                record = execute_one_attempt(
                    client=client,
                    session_number=session_number,
                    phase="warmup",
                    model=model,
                    method=method,
                    sample_index=config.WARMUP_SLICE_START + offset,
                    attempt_number=offset + 1,
                    session_attempt_index=session_attempt_index,
                    input_data=sample_as_input_data(row),
                )
                records.append(record)
                raw_writer.append(record)
            for offset, row in enumerate(measured_rows):
                session_attempt_index += 1
                record = execute_one_attempt(
                    client=client,
                    session_number=session_number,
                    phase="measured",
                    model=model,
                    method=method,
                    sample_index=config.MEASURED_SLICE_START + offset,
                    attempt_number=offset + 1,
                    session_attempt_index=session_attempt_index,
                    input_data=sample_as_input_data(row),
                )
                records.append(record)
                raw_writer.append(record)
    except KeyboardInterrupt:
        interrupted = True
    return records, interrupted


def latency_metrics(latencies_ms: list[float]) -> dict[str, Any]:
    if not latencies_ms:
        return {
            "n": 0,
            "mean_ms": None,
            "median_ms": None,
            "min_ms": None,
            "max_ms": None,
            "p95_ms": None,
            "p95_method": config.E2_P95_PERCENTILE_METHOD,
        }
    values = np.asarray(latencies_ms, dtype=float)
    return {
        "n": int(values.size),
        "mean_ms": float(np.mean(values)),
        "median_ms": float(np.median(values)),
        "min_ms": float(np.min(values)),
        "max_ms": float(np.max(values)),
        "p95_ms": float(
            np.percentile(values, 95, method=config.E2_P95_PERCENTILE_METHOD)
        ),
        "p95_method": config.E2_P95_PERCENTILE_METHOD,
    }


def _outcome_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "http_200": sum(1 for row in records if row.get("http_200")),
        "http_client_error": sum(
            1 for row in records if row.get("outcome") == "http_client_error"
        ),
        "http_server_error": sum(
            1 for row in records if row.get("outcome") == "http_server_error"
        ),
        "timeout": sum(1 for row in records if row.get("outcome") == "timeout"),
        "connection_error": sum(
            1 for row in records if row.get("outcome") == "connection_error"
        ),
        "transport_error": sum(
            1 for row in records if row.get("outcome") == "transport_error"
        ),
        "unexpected_http_status": sum(
            1 for row in records if row.get("outcome") == "unexpected_http_status"
        ),
        "json_parse_error": sum(
            1 for row in records if row.get("outcome") == "json_parse_error"
        ),
    }


def combination_metrics(
    records: list[dict[str, Any]],
    model: str,
    method: str,
) -> dict[str, Any]:
    planned = config.E2_MEASURED_ATTEMPTS_PER_COMBINATION_PER_SESSION
    measured = [
        row
        for row in records
        if row["phase"] == "measured" and row["model"] == model and row["method"] == method
    ]
    warmup = [
        row
        for row in records
        if row["phase"] == "warmup" and row["model"] == model and row["method"] == method
    ]
    counts = _outcome_counts(measured)
    http_200 = counts["http_200"]
    completed = len(measured)
    missing = max(planned - completed, 0)
    # HTTP/transport failures: not HTTP 200. JSON parse errors on HTTP 200 are not included.
    http_or_transport_failures = (completed - http_200) + missing
    latencies = [
        float(row["latency_ms"])
        for row in measured
        if row.get("http_200") and row.get("latency_ms") is not None
    ]
    return {
        "model": model,
        "method": method,
        "combination": combination_label(model, method),
        "warmup_recorded": len(warmup),
        "measured_planned": planned,
        "measured_completed": completed,
        "measured_missing": missing,
        "http_200_count": http_200,
        "http_200_rate": http_200 / planned if planned else 0.0,
        "http_client_error_count": counts["http_client_error"],
        "http_server_error_count": counts["http_server_error"],
        "timeout_count": counts["timeout"],
        "connection_error_count": counts["connection_error"],
        "transport_error_count": counts["transport_error"],
        "unexpected_status_count": counts["unexpected_http_status"],
        "json_parse_error_count": counts["json_parse_error"],
        "http_or_transport_failures": http_or_transport_failures,
        "http_or_transport_failure_rate": (
            http_or_transport_failures / planned if planned else 0.0
        ),
        "rate_denominator": planned,
        "http_or_transport_failures_definition": (
            "Non-200 HTTP responses, timeouts, connection/transport errors, "
            "unexpected statuses, and missing measured attempts. "
            "Does not include HTTP 200 JSON parse errors. "
            "Not an application-success or scalability metric. "
            "Denominator is 100 planned measured attempts."
        ),
        "latency_denominator_note": (
            "Latency uses measured HTTP 200 responses only, including HTTP 200 "
            "bodies that fail JSON parsing. Warm-ups and non-200 outcomes are excluded. "
            "All rates use 100 planned measured attempts."
        ),
        "latency": latency_metrics(latencies),
    }


def build_summary(
    records: list[dict[str, Any]],
    metadata: dict[str, Any],
    *,
    interrupted: bool,
) -> dict[str, Any]:
    warmup = [row for row in records if row["phase"] == "warmup"]
    measured = [row for row in records if row["phase"] == "measured"]
    completed = not interrupted and len(warmup) == WARMUP_PER_SESSION and len(measured) == MEASURED_PER_SESSION
    by_combination = {}
    for model, method in config.COMBINATIONS:
        label = combination_label(model, method)
        stats = combination_metrics(records, model, method)
        stats.pop("unused", None)
        by_combination[label] = stats

    overall_counts = _outcome_counts(measured)
    overall_http_200 = overall_counts["http_200"]
    missing_measured = max(MEASURED_PER_SESSION - len(measured), 0)
    overall_http_or_transport_failures = (
        sum(1 for row in measured if not row.get("http_200")) + missing_measured
    )
    overall_latencies = [
        float(row["latency_ms"])
        for row in measured
        if row.get("http_200") and row.get("latency_ms") is not None
    ]

    return {
        "experiment_id": EXPERIMENT_ID,
        "session_number": metadata.get("session_number"),
        "execution_status": "completed" if completed else "interrupted",
        "overall_outcome": "completed" if completed else "incomplete",
        "completed_session": completed,
        "interpretation_note": (
            "Sequential, end-to-end, client-observed API latency. Not isolated "
            "SHAP/LIME time, throughput, or scalability."
        ),
        "counts": {
            "warmup_planned": WARMUP_PER_SESSION,
            "warmup_completed": len(warmup),
            "measured_planned": MEASURED_PER_SESSION,
            "measured_completed": len(measured),
            "measured_missing": missing_measured,
            "http_200_count": overall_http_200,
            "http_200_rate": overall_http_200 / MEASURED_PER_SESSION,
            "http_client_error_count": overall_counts["http_client_error"],
            "http_server_error_count": overall_counts["http_server_error"],
            "timeout_count": overall_counts["timeout"],
            "connection_error_count": overall_counts["connection_error"],
            "transport_error_count": overall_counts["transport_error"],
            "unexpected_status_count": overall_counts["unexpected_http_status"],
            "json_parse_error_count": overall_counts["json_parse_error"],
            "latency_observation_count": len(overall_latencies),
            "http_or_transport_failures": overall_http_or_transport_failures,
            "http_or_transport_failure_rate": (
                overall_http_or_transport_failures / MEASURED_PER_SESSION
            ),
            "rate_denominator": MEASURED_PER_SESSION,
            "http_or_transport_failures_definition": (
                "Non-200 HTTP responses, timeouts, connection/transport errors, "
                "unexpected statuses, and missing measured attempts. "
                "Does not include HTTP 200 JSON parse errors. "
                "Not an application-success or scalability metric. "
                "Session-level denominator is 400 planned measured attempts."
            ),
        },
        "overall_latency": latency_metrics(overall_latencies),
        "by_combination": by_combination,
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
    session_number: int,
    run_stamp: str,
    raw_path: Path,
    summary_path: Path,
    feature_names: list[str],
    artifact_sha256: str,
    session_start_utc: str,
    combination_order: list[tuple[str, str]],
    session_1_summary_path: str | None,
    session_1_end_utc: str | None,
    gap_seconds: float | None,
) -> dict[str, Any]:
    order_1, order_2 = session_combination_orders()
    return {
        "session_number": session_number,
        "session_start_timestamp_utc": session_start_utc,
        "session_end_timestamp_utc": None,
        "run_stamp": run_stamp,
        "raw_results_path": str(raw_path),
        "summary_results_path": str(summary_path),
        "base_url": config.BASE_URL,
        "explain_url": explain_url(),
        "git_head_local": git_head(),
        "deployed_git_revision": "unverified",
        "python_version": python_version(),
        "os": os_platform_info(),
        "installed_package_versions": installed_evaluation_package_versions(),
        "http_client": "httpx",
        "http_timeout_seconds": config.HTTP_TIMEOUT_SECONDS,
        "sequential": True,
        "retries": False,
        "parallel": False,
        "random_seed": config.E2_RANDOMIZATION_SEED,
        "combination_order": [combination_label(m, h) for m, h in combination_order],
        "session_1_combination_order": [combination_label(m, h) for m, h in order_1],
        "session_2_combination_order": [combination_label(m, h) for m, h in order_2],
        "warmup_sample_indices": list(
            range(config.WARMUP_SLICE_START, config.WARMUP_SLICE_END)
        ),
        "measured_sample_indices": list(
            range(config.MEASURED_SLICE_START, config.MEASURED_SLICE_END)
        ),
        "p95_method": config.E2_P95_PERCENTILE_METHOD,
        "dataset_artifact_path": str(config.TRAINING_DATA_PATH),
        "dataset_artifact_sha256": artifact_sha256,
        "feature_names": feature_names,
        "session_1_summary_path": session_1_summary_path,
        "session_1_end_timestamp_utc": session_1_end_utc,
        "session_gap_seconds": gap_seconds,
        "min_session_separation_seconds": config.E2_MIN_SESSION_SEPARATION_SECONDS,
        "planned_warmup_attempts": WARMUP_PER_SESSION,
        "planned_measured_attempts": MEASURED_PER_SESSION,
    }


def load_raw_jsonl_records(raw_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with raw_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise Session1ValidationError(
                    f"Session 1 raw file has a blank line at {line_number}."
                )
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise Session1ValidationError(
                    f"Session 1 raw file line {line_number} is not valid JSON: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise Session1ValidationError(
                    f"Session 1 raw file line {line_number} is not a JSON object."
                )
            records.append(row)
    return records


def validate_session_1_raw_records(records: list[dict[str, Any]]) -> None:
    """Structural checks for a completed E2 session-1 JSONL file."""
    expected_total = ATTEMPTS_PER_SESSION
    if len(records) != expected_total:
        raise Session1ValidationError(
            f"Session 1 raw file has {len(records)} records; expected {expected_total}."
        )

    allowed_combos = set(config.COMBINATIONS)
    allowed_phases = {"warmup", "measured"}
    sample_keys: list[tuple[Any, ...]] = []
    attempt_keys: list[tuple[Any, ...]] = []
    warmup_total = 0
    measured_total = 0

    for index, row in enumerate(records, start=1):
        if row.get("experiment_id") != EXPERIMENT_ID:
            raise Session1ValidationError(
                f"Session 1 raw record {index} has experiment_id={row.get('experiment_id')!r}."
            )
        if row.get("session_number") != 1:
            raise Session1ValidationError(
                f"Session 1 raw record {index} has session_number={row.get('session_number')!r}."
            )
        phase = row.get("phase")
        if phase not in allowed_phases:
            raise Session1ValidationError(
                f"Session 1 raw record {index} has unexpected phase={phase!r}."
            )
        if phase == "warmup":
            warmup_total += 1
        else:
            measured_total += 1
        model = row.get("model")
        method = row.get("method")
        if (model, method) not in allowed_combos:
            raise Session1ValidationError(
                f"Session 1 raw record {index} has unexpected combination "
                f"{model!r}+{method!r}."
            )
        sample_keys.append((model, method, phase, row.get("sample_index")))
        attempt_keys.append((model, method, phase, row.get("attempt_number")))

    if warmup_total != WARMUP_PER_SESSION or measured_total != MEASURED_PER_SESSION:
        raise Session1ValidationError(
            f"Session 1 raw file has warmup={warmup_total}, measured={measured_total}; "
            f"expected {WARMUP_PER_SESSION} and {MEASURED_PER_SESSION}."
        )
    if len(sample_keys) != len(set(sample_keys)):
        raise Session1ValidationError(
            "Session 1 raw file contains duplicate combination/phase/sample_index records."
        )
    if len(attempt_keys) != len(set(attempt_keys)):
        raise Session1ValidationError(
            "Session 1 raw file contains duplicate combination/phase/attempt_number records."
        )

    warmup_indices = list(range(config.WARMUP_SLICE_START, config.WARMUP_SLICE_END))
    measured_indices = list(
        range(config.MEASURED_SLICE_START, config.MEASURED_SLICE_END)
    )
    warmup_attempts = list(range(1, config.E2_WARMUP_PER_COMBINATION + 1))
    measured_attempts = list(
        range(1, config.E2_MEASURED_ATTEMPTS_PER_COMBINATION_PER_SESSION + 1)
    )

    for model, method in config.COMBINATIONS:
        label = combination_label(model, method)
        combo_rows = [
            row
            for row in records
            if row.get("model") == model and row.get("method") == method
        ]
        warmup = [row for row in combo_rows if row.get("phase") == "warmup"]
        measured = [row for row in combo_rows if row.get("phase") == "measured"]
        if len(warmup) != config.E2_WARMUP_PER_COMBINATION:
            raise Session1ValidationError(
                f"{label} has {len(warmup)} warm-up records; expected "
                f"{config.E2_WARMUP_PER_COMBINATION}."
            )
        if len(measured) != config.E2_MEASURED_ATTEMPTS_PER_COMBINATION_PER_SESSION:
            raise Session1ValidationError(
                f"{label} has {len(measured)} measured records; expected "
                f"{config.E2_MEASURED_ATTEMPTS_PER_COMBINATION_PER_SESSION}."
            )
        if [row.get("sample_index") for row in warmup] != warmup_indices:
            raise Session1ValidationError(
                f"{label} warm-up sample indices are not exactly {warmup_indices}."
            )
        if [row.get("sample_index") for row in measured] != measured_indices:
            raise Session1ValidationError(
                f"{label} measured sample indices are not exactly 10–109 in order."
            )
        if [row.get("attempt_number") for row in warmup] != warmup_attempts:
            raise Session1ValidationError(
                f"{label} warm-up attempt_number values are not 1–10 in order."
            )
        if [row.get("attempt_number") for row in measured] != measured_attempts:
            raise Session1ValidationError(
                f"{label} measured attempt_number values are not 1–100 in order."
            )


def write_valid_session1_raw(path: Path) -> None:
    """Write structurally valid synthetic session-1 JSONL for offline tests."""
    with path.open("w", encoding="utf-8") as handle:
        for model, method in config.COMBINATIONS:
            for offset in range(config.E2_WARMUP_PER_COMBINATION):
                handle.write(
                    json.dumps(
                        {
                            "experiment_id": EXPERIMENT_ID,
                            "session_number": 1,
                            "phase": "warmup",
                            "model": model,
                            "method": method,
                            "sample_index": config.WARMUP_SLICE_START + offset,
                            "attempt_number": offset + 1,
                        }
                    )
                    + "\n"
                )
            for offset in range(config.E2_MEASURED_ATTEMPTS_PER_COMBINATION_PER_SESSION):
                handle.write(
                    json.dumps(
                        {
                            "experiment_id": EXPERIMENT_ID,
                            "session_number": 1,
                            "phase": "measured",
                            "model": model,
                            "method": method,
                            "sample_index": config.MEASURED_SLICE_START + offset,
                            "attempt_number": offset + 1,
                        }
                    )
                    + "\n"
                )


def validate_session_1_for_session_2(
    summary_path: Path,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    if not summary_path.is_file():
        raise Session1ValidationError(
            f"Session 1 summary not found: {summary_path}"
        )
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise Session1ValidationError(f"Session 1 summary is not valid JSON: {exc}") from exc

    if summary.get("experiment_id") != EXPERIMENT_ID:
        raise Session1ValidationError("Session 1 summary is not an E2 summary.")
    if summary.get("session_number") != 1:
        raise Session1ValidationError("Supplied summary is not session 1.")
    if summary.get("execution_status") != "completed" or not summary.get(
        "completed_session"
    ):
        raise Session1ValidationError(
            "Session 1 is not completed; session 2 must not start."
        )

    counts = summary.get("counts") or {}
    if counts.get("warmup_completed") != WARMUP_PER_SESSION:
        raise Session1ValidationError(
            f"Session 1 warm-up count is {counts.get('warmup_completed')}, expected {WARMUP_PER_SESSION}."
        )
    if counts.get("measured_completed") != MEASURED_PER_SESSION:
        raise Session1ValidationError(
            f"Session 1 measured count is {counts.get('measured_completed')}, expected {MEASURED_PER_SESSION}."
        )

    metadata = summary.get("metadata") or {}
    raw_value = metadata.get("raw_results_path")
    if not raw_value:
        raise Session1ValidationError("Session 1 summary does not record a raw results path.")
    raw_path = Path(raw_value)
    if not raw_path.is_file():
        raise Session1ValidationError(f"Session 1 raw file not found: {raw_path}")

    try:
        raw_records = load_raw_jsonl_records(raw_path)
    except OSError as exc:
        raise Session1ValidationError(
            f"Session 1 raw file could not be read: {raw_path}"
        ) from exc
    validate_session_1_raw_records(raw_records)

    end_raw = metadata.get("session_end_timestamp_utc")
    if not end_raw:
        raise Session1ValidationError("Session 1 summary lacks session_end_timestamp_utc.")
    try:
        session_1_end = parse_utc(str(end_raw))
    except ValueError as exc:
        raise Session1ValidationError(
            f"Session 1 end timestamp is invalid: {end_raw}"
        ) from exc

    now_utc = now or utc_now()
    gap_seconds = (now_utc - session_1_end).total_seconds()
    if gap_seconds < config.E2_MIN_SESSION_SEPARATION_SECONDS:
        raise Session1ValidationError(
            f"Only {gap_seconds:.1f}s have elapsed since session 1 ended; "
            f"need at least {config.E2_MIN_SESSION_SEPARATION_SECONDS}s."
        )
    return {
        "summary": summary,
        "raw_path": str(raw_path),
        "session_1_end_timestamp_utc": session_1_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "gap_seconds": gap_seconds,
    }


def execute_experiment(
    session_number: int,
    session_1_summary_path: Path | None,
) -> int:
    import httpx

    session_1_end_utc = None
    gap_seconds = None
    session_1_summary_str = None
    session_start_utc = utc_now_iso()
    if session_number == 2:
        if session_1_summary_path is None:
            raise Session1ValidationError(
                "Session 2 requires --session-1-summary <path>."
            )
        check = validate_session_1_for_session_2(
            session_1_summary_path, now=parse_utc(session_start_utc)
        )
        session_1_end_utc = check["session_1_end_timestamp_utc"]
        gap_seconds = check["gap_seconds"]
        session_1_summary_str = str(session_1_summary_path)

    warmup_rows, measured_rows, feature_names = load_e2_dataset()
    artifact_sha256 = sha256_file(config.TRAINING_DATA_PATH)
    run_stamp, raw_path, summary_path = allocate_run_output_paths(session_number)
    combination_order = combination_order_for_session(session_number)
    metadata = build_run_metadata(
        session_number=session_number,
        run_stamp=run_stamp,
        raw_path=raw_path,
        summary_path=summary_path,
        feature_names=feature_names,
        artifact_sha256=artifact_sha256,
        session_start_utc=session_start_utc,
        combination_order=combination_order,
        session_1_summary_path=session_1_summary_str,
        session_1_end_utc=session_1_end_utc,
        gap_seconds=gap_seconds,
    )
    timeout = httpx.Timeout(config.HTTP_TIMEOUT_SECONDS)
    transport = httpx.HTTPTransport(retries=0)
    with RawJsonlWriter(raw_path) as writer:
        with httpx.Client(timeout=timeout, transport=transport) as client:
            records, interrupted = run_e2_session(
                client,
                session_number=session_number,
                warmup_rows=warmup_rows,
                measured_rows=measured_rows,
                raw_writer=writer,
            )
    metadata["session_end_timestamp_utc"] = utc_now_iso()
    if interrupted:
        metadata["interruption"] = "KeyboardInterrupt"
    summary = build_summary(records, metadata, interrupted=interrupted)
    write_summary(summary, summary_path)
    print(f"raw: {raw_path}")
    print(f"summary: {summary_path}")
    print(f"execution_status: {summary['execution_status']}")
    print(
        f"measured: {summary['counts']['measured_completed']}/"
        f"{summary['counts']['measured_planned']}"
    )
    if interrupted or not summary["completed_session"]:
        return 130 if interrupted else 1
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "E2 runtime evaluation: two independent sessions of 4 combinations "
            "× (10 warm-up + 100 measured) sequential POST /explain requests."
        )
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run one E2 session (requires --confirm E2 and --session).",
    )
    parser.add_argument(
        "--session",
        type=int,
        choices=(1, 2),
        help="Session number to execute.",
    )
    parser.add_argument(
        "--session-1-summary",
        type=Path,
        help="Path to completed session 1 summary JSON (required for session 2).",
    )
    parser.add_argument(
        "--confirm",
        metavar="TOKEN",
        help=f"Safety token. Must be '{CONFIRM_TOKEN}' together with --execute.",
    )
    parser.add_argument(
        "--offline-check",
        action="store_true",
        help="Run mocked protocol/metrics checks. No HTTP.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.offline_check:
        return run_offline_checks()
    if args.execute:
        if args.confirm != CONFIRM_TOKEN:
            print(
                "Refusing to run E2: pass --execute --session N --confirm E2.",
                file=sys.stderr,
            )
            return 2
        if args.session is None:
            print("Refusing to run E2: --session 1 or --session 2 is required.", file=sys.stderr)
            return 2
        try:
            return execute_experiment(args.session, args.session_1_summary)
        except Session1ValidationError as exc:
            print(f"Refusing session 2: {exc}", file=sys.stderr)
            return 2
    print(
        "No action taken. Use --help, --offline-check, "
        "or --execute --session 1 --confirm E2."
    )
    return 0


def expected_e1_e4_evidence_paths() -> list[Path]:
    return [
        config.RAW_RESULTS_DIR / "e1_contract_20261102_204631.jsonl",
        config.SUMMARY_RESULTS_DIR / "e1_contract_20261102_204631.json",
        config.RAW_RESULTS_DIR / "e4_invalid_inputs_20261102_211901.jsonl",
        config.SUMMARY_RESULTS_DIR / "e4_invalid_inputs_20261102_211901.json",
    ]


def require_evidence_files(paths: list[Path]) -> None:
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise AssertionError(
            "Offline check requires existing E1/E4 evidence files; missing: "
            + ", ".join(str(path) for path in missing)
        )


def run_offline_checks() -> int:
    import tempfile
    from collections import Counter

    def file_sha256(path: Path) -> str | None:
        if not path.exists():
            return None
        return hashlib.sha256(path.read_bytes()).hexdigest()

    evidence_files = expected_e1_e4_evidence_paths()
    require_evidence_files(evidence_files)
    evidence_hashes = {str(path): file_sha256(path) for path in evidence_files}
    assert all(digest is not None for digest in evidence_hashes.values())
    try:
        require_evidence_files(
            evidence_files + [Path("__missing_e1_e4_evidence__.json")]
        )
        raise AssertionError("missing evidence file should fail closed")
    except AssertionError as exc:
        if "__missing_e1_e4_evidence__" not in str(exc):
            raise

    assert len(config.COMBINATIONS) == 4
    warmup_rows, measured_rows, feature_names = load_e2_dataset()
    assert warmup_rows.shape == (10, 30)
    assert measured_rows.shape == (100, 30)
    assert len(feature_names) == 30
    assert list(range(config.WARMUP_SLICE_START, config.WARMUP_SLICE_END)) == list(
        range(10)
    )
    assert list(range(config.MEASURED_SLICE_START, config.MEASURED_SLICE_END)) == list(
        range(10, 110)
    )

    order_1, order_2 = session_combination_orders()
    assert order_1 != order_2
    assert sorted(order_1) == sorted(config.COMBINATIONS)
    assert sorted(order_2) == sorted(config.COMBINATIONS)
    again_1, again_2 = session_combination_orders()
    assert again_1 == order_1 and again_2 == order_2
    assert combination_order_for_session(1) == order_1
    assert combination_order_for_session(2) == order_2

    latencies = [10.0, 20.0, 30.0, 40.0, 50.0]
    metrics = latency_metrics(latencies)
    expected_p95 = float(np.percentile(np.asarray(latencies), 95, method="linear"))
    assert metrics["p95_ms"] == expected_p95
    assert metrics["n"] == 5
    empty = latency_metrics([])
    assert empty["n"] == 0 and empty["p95_ms"] is None

    combo = config.COMBINATIONS[0]
    model, method = combo
    synthetic: list[dict[str, Any]] = []
    for i in range(10):
        synthetic.append(
            {
                "phase": "warmup",
                "model": model,
                "method": method,
                "http_200": True,
                "latency_eligible": True,
                "latency_ms": 9999.0,
                "outcome": "http_200",
            }
        )
    for i in range(97):
        synthetic.append(
            {
                "phase": "measured",
                "model": model,
                "method": method,
                "http_200": True,
                "latency_eligible": True,
                "latency_ms": float(i + 1),
                "outcome": "http_200",
            }
        )
    synthetic.append(
        {
            "phase": "measured",
            "model": model,
            "method": method,
            "http_200": True,
            "latency_eligible": True,
            "latency_ms": 12.0,
            "outcome": "json_parse_error",
        }
    )
    synthetic.append(
        {
            "phase": "measured",
            "model": model,
            "method": method,
            "http_200": False,
            "latency_eligible": False,
            "latency_ms": 15.0,
            "outcome": "http_server_error",
        }
    )
    synthetic.append(
        {
            "phase": "measured",
            "model": model,
            "method": method,
            "http_200": False,
            "latency_eligible": False,
            "latency_ms": 120000.0,
            "outcome": "timeout",
        }
    )
    # 97 HTTP 200 JSON + 1 HTTP 200 JSON parse + 1 5xx + 1 timeout = 100.
    stats = combination_metrics(synthetic, model, method)
    assert stats["measured_planned"] == 100
    assert stats["rate_denominator"] == 100
    assert stats["http_200_count"] == 98
    assert stats["http_200_rate"] == 0.98
    assert stats["json_parse_error_count"] == 1
    assert stats["http_server_error_count"] == 1
    assert stats["timeout_count"] == 1
    assert stats["http_or_transport_failures"] == 2
    assert stats["http_or_transport_failure_rate"] == 0.02
    assert stats["latency"]["n"] == 98
    assert 12.0 in [
        float(row["latency_ms"])
        for row in synthetic
        if row["phase"] == "measured" and row.get("http_200")
    ]
    assert 9999.0 not in [
        float(row["latency_ms"])
        for row in synthetic
        if row["phase"] == "measured" and row.get("http_200")
    ]

    assert classify_http_outcome(
        transport_error="TimeoutException: timed out",
        exception_name="TimeoutException",
        http_status=None,
        json_error=None,
    ) == "timeout"
    assert classify_http_outcome(
        transport_error="ConnectError: refused",
        exception_name="ConnectError",
        http_status=None,
        json_error=None,
    ) == "connection_error"
    assert classify_http_outcome(
        transport_error=None,
        exception_name=None,
        http_status=422,
        json_error=None,
    ) == "http_client_error"
    assert classify_http_outcome(
        transport_error=None,
        exception_name=None,
        http_status=500,
        json_error=None,
    ) == "http_server_error"
    assert classify_http_outcome(
        transport_error=None,
        exception_name=None,
        http_status=200,
        json_error="expecting value",
    ) == "json_parse_error"
    assert classify_http_outcome(
        transport_error=None,
        exception_name=None,
        http_status=204,
        json_error=None,
    ) == "unexpected_http_status"

    class ParseFailClient:
        def post(self, url: str, json: dict[str, Any] | None = None) -> Any:
            class Resp:
                status_code = 200
                text = "not-json{"
            return Resp()

    parse_fail_record = execute_one_attempt(
        client=ParseFailClient(),
        session_number=1,
        phase="measured",
        model=model,
        method=method,
        sample_index=10,
        attempt_number=1,
        session_attempt_index=1,
        input_data=sample_as_input_data(measured_rows[0]),
    )
    assert parse_fail_record["http_200"] is True
    assert parse_fail_record["outcome"] == "json_parse_error"
    assert parse_fail_record["latency_eligible"] is True
    assert parse_fail_record["latency_ms"] is not None

    class OutcomeClient:
        def __init__(self, status: int | None = None, body: str = "", error: Exception | None = None):
            self.status = status
            self.body = body
            self.error = error

        def post(self, url: str, json: dict[str, Any] | None = None) -> Any:
            if self.error is not None:
                raise self.error
            class Resp:
                status_code = self.status
                text = self.body
            return Resp()

    def one_attempt(client: Any) -> dict[str, Any]:
        return execute_one_attempt(
            client=client,
            session_number=1,
            phase="measured",
            model=model,
            method=method,
            sample_index=10,
            attempt_number=1,
            session_attempt_index=1,
            input_data=sample_as_input_data(measured_rows[0]),
        )

    valid_json_body = json.dumps(
        {
            "model": model,
            "method": method,
            "prediction": 0,
            "probability": 0.5,
            "feature_importance": {"mean radius": 0.1},
        }
    )
    rec_ok = one_attempt(OutcomeClient(200, valid_json_body))
    assert rec_ok["outcome"] == "http_200" and rec_ok["latency_eligible"] is True
    rec_4xx = one_attempt(OutcomeClient(422, "{}"))
    assert rec_4xx["outcome"] == "http_client_error" and rec_4xx["latency_eligible"] is False
    rec_5xx = one_attempt(OutcomeClient(500, "{}"))
    assert rec_5xx["outcome"] == "http_server_error" and rec_5xx["latency_eligible"] is False
    rec_timeout = one_attempt(OutcomeClient(error=TimeoutError("timed out")))
    assert rec_timeout["outcome"] == "timeout" and rec_timeout["latency_eligible"] is False

    class ConnectError(Exception):
        pass

    rec_conn = one_attempt(OutcomeClient(error=ConnectError("refused")))
    assert rec_conn["outcome"] == "connection_error" and rec_conn["latency_eligible"] is False
    rec_transport = one_attempt(OutcomeClient(error=RuntimeError("socket reset")))
    assert rec_transport["outcome"] == "transport_error" and rec_transport["latency_eligible"] is False
    rec_unexpected = one_attempt(OutcomeClient(204, ""))
    assert rec_unexpected["outcome"] == "unexpected_http_status"
    assert rec_unexpected["latency_eligible"] is False

    mixed_records = [
        {**rec_ok, "phase": "measured", "model": model, "method": method},
        {**parse_fail_record, "phase": "measured", "model": model, "method": method},
        {**rec_4xx, "phase": "measured", "model": model, "method": method},
        {**rec_5xx, "phase": "measured", "model": model, "method": method},
        {**rec_timeout, "phase": "measured", "model": model, "method": method},
        {**rec_conn, "phase": "measured", "model": model, "method": method},
        {**rec_transport, "phase": "measured", "model": model, "method": method},
        {**rec_unexpected, "phase": "measured", "model": model, "method": method},
    ]
    mixed_stats = combination_metrics(mixed_records, model, method)
    assert mixed_stats["measured_completed"] == 8
    assert mixed_stats["measured_missing"] == 92
    assert mixed_stats["http_200_count"] == 2
    assert mixed_stats["json_parse_error_count"] == 1
    assert mixed_stats["http_client_error_count"] == 1
    assert mixed_stats["http_server_error_count"] == 1
    assert mixed_stats["timeout_count"] == 1
    assert mixed_stats["connection_error_count"] == 1
    assert mixed_stats["transport_error_count"] == 1
    assert mixed_stats["unexpected_status_count"] == 1
    assert mixed_stats["http_or_transport_failures"] == 6 + 92
    assert mixed_stats["latency"]["n"] == 2
    assert mixed_stats["rate_denominator"] == 100

    json_dumps = json.dumps

    class FakeResp:
        def __init__(self, code: int, body: str):
            self.status_code = code
            self.text = body

    class CountingClient:
        def __init__(self, interrupt_after: int | None = None):
            self.calls = 0
            self.interrupt_after = interrupt_after
            self.urls: list[str] = []

        def post(self, url: str, json: dict[str, Any] | None = None) -> FakeResp:
            self.calls += 1
            self.urls.append(url)
            if self.interrupt_after is not None and self.calls > self.interrupt_after:
                raise KeyboardInterrupt
            assert json is not None
            payload = json
            assert len(payload["input_data"]) == 30
            body = {
                "model": payload["model"],
                "method": payload["method"],
                "prediction": 0,
                "probability": 0.5,
                "feature_importance": {name: 0.01 for name in feature_names},
            }
            return FakeResp(200, json_dumps(body))

    with tempfile.TemporaryDirectory() as tmp:
        raw_path = Path(tmp) / "e2_offline.jsonl"
        client = CountingClient()
        with RawJsonlWriter(raw_path) as writer:
            records, interrupted = run_e2_session(
                client,
                session_number=1,
                warmup_rows=warmup_rows,
                measured_rows=measured_rows,
                raw_writer=writer,
            )
        assert interrupted is False
        assert client.calls == ATTEMPTS_PER_SESSION
        assert len(records) == ATTEMPTS_PER_SESSION
        phases = Counter(row["phase"] for row in records)
        assert phases["warmup"] == 40
        assert phases["measured"] == 400
        combos = [row["combination"] for row in records if row["phase"] == "warmup"]
        expected_warmup_order = []
        for model, method in order_1:
            expected_warmup_order.extend([combination_label(model, method)] * 10)
        assert combos == expected_warmup_order
        measured_indices = [
            row["sample_index"]
            for row in records
            if row["phase"] == "measured" and row["combination"] == combination_label(*order_1[0])
        ]
        assert measured_indices == list(range(10, 110))
        assert writer.records_written == ATTEMPTS_PER_SESSION
        summary = build_summary(
            records,
            {"session_number": 1},
            interrupted=False,
        )
        assert summary["completed_session"] is True
        assert summary["counts"]["measured_completed"] == 400
        assert summary["counts"]["http_200_count"] == 400
        assert summary["overall_latency"]["n"] == 400

        interrupt_client = CountingClient(interrupt_after=5)
        partial_path = Path(tmp) / "e2_partial.jsonl"
        with RawJsonlWriter(partial_path) as writer:
            partial_records, interrupted = run_e2_session(
                interrupt_client,
                session_number=1,
                warmup_rows=warmup_rows,
                measured_rows=measured_rows,
                raw_writer=writer,
            )
        assert interrupted is True
        assert len(partial_records) == 5
        assert len(partial_path.read_text(encoding="utf-8").splitlines()) == 5
        partial_summary = build_summary(
            partial_records, {"session_number": 1}, interrupted=True
        )
        assert partial_summary["execution_status"] == "interrupted"
        assert partial_summary["completed_session"] is False
        assert partial_summary["counts"]["measured_missing"] == MEASURED_PER_SESSION
        assert (
            partial_summary["counts"]["http_or_transport_failures"]
            == MEASURED_PER_SESSION
        )
        first_combo_stats = partial_summary["by_combination"][
            combination_label(*order_1[0])
        ]
        assert first_combo_stats["measured_missing"] == 100
        assert first_combo_stats["http_or_transport_failures"] == 100

        occupied_raw_dir = config.RAW_RESULTS_DIR
        occupied_sum_dir = config.SUMMARY_RESULTS_DIR
        try:
            config.RAW_RESULTS_DIR = Path(tmp) / "raw"
            config.SUMMARY_RESULTS_DIR = Path(tmp) / "summaries"
            config.RAW_RESULTS_DIR.mkdir()
            config.SUMMARY_RESULTS_DIR.mkdir()
            stamp = "20260101_120000"
            occ_raw, occ_sum = result_paths_for_stamp(1, stamp)
            occ_raw.write_text("EXISTING_RAW\n", encoding="utf-8")
            occ_sum.write_text("EXISTING_SUMMARY\n", encoding="utf-8")
            new_stamp, new_raw, new_sum = allocate_run_output_paths(1, stamp)
            assert new_stamp != stamp
            try:
                with RawJsonlWriter(occ_raw):
                    raise AssertionError("overwrote raw")
            except FileExistsError:
                pass
            assert occ_raw.read_text(encoding="utf-8") == "EXISTING_RAW\n"

            fake_raw = Path(tmp) / "session1.jsonl"
            write_valid_session1_raw(fake_raw)
            validate_session_1_raw_records(load_raw_jsonl_records(fake_raw))

            def mutate_raw(name: str, mutator: Any) -> Path:
                dest = Path(tmp) / name
                rows = load_raw_jsonl_records(fake_raw)
                mutator(rows)
                dest.write_text(
                    "".join(json.dumps(row) + "\n" for row in rows),
                    encoding="utf-8",
                )
                return dest

            def expect_raw_error(path: Path) -> None:
                try:
                    validate_session_1_raw_records(load_raw_jsonl_records(path))
                except Session1ValidationError:
                    return
                raise AssertionError(f"expected structural rejection for {path}")

            wrong_id = mutate_raw(
                "wrong_id.jsonl",
                lambda rows: rows.__setitem__(
                    0, {**rows[0], "experiment_id": "E1"}
                ),
            )
            expect_raw_error(wrong_id)

            wrong_session = mutate_raw(
                "wrong_session.jsonl",
                lambda rows: rows.__setitem__(
                    0, {**rows[0], "session_number": 2}
                ),
            )
            expect_raw_error(wrong_session)

            missing_rows = load_raw_jsonl_records(fake_raw)[:-1]
            missing_path = Path(tmp) / "missing_record.jsonl"
            missing_path.write_text(
                "".join(json.dumps(row) + "\n" for row in missing_rows),
                encoding="utf-8",
            )
            expect_raw_error(missing_path)

            duplicated = mutate_raw(
                "duplicated.jsonl",
                lambda rows: rows.append(dict(rows[-1])),
            )
            expect_raw_error(duplicated)

            bad_index = mutate_raw(
                "bad_index.jsonl",
                lambda rows: rows.__setitem__(
                    0, {**rows[0], "sample_index": 9}
                ),
            )
            expect_raw_error(bad_index)

            reordered_rows = load_raw_jsonl_records(fake_raw)
            reordered_rows[0], reordered_rows[1] = reordered_rows[1], reordered_rows[0]
            reordered = Path(tmp) / "reordered.jsonl"
            reordered.write_text(
                "".join(json.dumps(row) + "\n" for row in reordered_rows),
                encoding="utf-8",
            )
            expect_raw_error(reordered)

            unexpected_phase = mutate_raw(
                "bad_phase.jsonl",
                lambda rows: rows.__setitem__(
                    0, {**rows[0], "phase": "probe"}
                ),
            )
            expect_raw_error(unexpected_phase)

            malformed_path = Path(tmp) / "malformed.jsonl"
            good_lines = fake_raw.read_text(encoding="utf-8").splitlines(True)
            good_lines[0] = "{not valid json\n"
            malformed_path.write_text("".join(good_lines), encoding="utf-8")
            expect_raw_error(malformed_path)

            duplicate_attempt = mutate_raw(
                "dup_attempt.jsonl",
                lambda rows: rows.__setitem__(
                    1, {**rows[1], "attempt_number": 1, "sample_index": 1}
                ),
            )
            expect_raw_error(duplicate_attempt)

            bad_attempt = mutate_raw(
                "bad_attempt.jsonl",
                lambda rows: rows.__setitem__(
                    0, {**rows[0], "attempt_number": 0}
                ),
            )
            expect_raw_error(bad_attempt)

            other_model, other_method = config.COMBINATIONS[1]
            uneven = mutate_raw(
                "uneven.jsonl",
                lambda rows: rows.__setitem__(
                    9,
                    {
                        **rows[9],
                        "model": other_model,
                        "method": other_method,
                    },
                ),
            )
            expect_raw_error(uneven)

            unexpected_combo = mutate_raw(
                "bad_combo.jsonl",
                lambda rows: rows.__setitem__(
                    0, {**rows[0], "model": "svm", "method": "shap"}
                ),
            )
            expect_raw_error(unexpected_combo)

            recent_end = utc_now()
            recent_summary = {
                "experiment_id": "E2",
                "session_number": 1,
                "execution_status": "completed",
                "completed_session": True,
                "counts": {"warmup_completed": 40, "measured_completed": 400},
                "metadata": {
                    "raw_results_path": str(fake_raw),
                    "session_end_timestamp_utc": recent_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                },
            }
            recent_path = Path(tmp) / "recent.json"
            recent_path.write_text(json.dumps(recent_summary), encoding="utf-8")
            try:
                validate_session_1_for_session_2(recent_path, now=utc_now())
                raise AssertionError("too-recent session 1 should fail")
            except Session1ValidationError:
                pass

            missing_path = Path(tmp) / "missing.json"
            try:
                validate_session_1_for_session_2(missing_path)
                raise AssertionError("missing summary should fail")
            except Session1ValidationError:
                pass

            incomplete = dict(recent_summary)
            incomplete["execution_status"] = "interrupted"
            incomplete["completed_session"] = False
            inc_path = Path(tmp) / "incomplete.json"
            inc_path.write_text(json.dumps(incomplete), encoding="utf-8")
            try:
                validate_session_1_for_session_2(inc_path)
                raise AssertionError("incomplete session 1 should fail")
            except Session1ValidationError:
                pass

            valid_end = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
            valid_summary = {
                "experiment_id": "E2",
                "session_number": 1,
                "execution_status": "completed",
                "completed_session": True,
                "counts": {"warmup_completed": 40, "measured_completed": 400},
                "metadata": {
                    "raw_results_path": str(fake_raw),
                    "session_end_timestamp_utc": valid_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                },
            }
            valid_path = Path(tmp) / "valid_s1.json"
            valid_path.write_text(json.dumps(valid_summary), encoding="utf-8")
            now = datetime(2026, 1, 1, 12, 31, 0, tzinfo=timezone.utc)
            check = validate_session_1_for_session_2(valid_path, now=now)
            assert check["gap_seconds"] == 1860.0

            too_soon = datetime(2026, 1, 1, 12, 10, 0, tzinfo=timezone.utc)
            try:
                validate_session_1_for_session_2(valid_path, now=too_soon)
                raise AssertionError("10-minute gap should fail")
            except Session1ValidationError:
                pass
        finally:
            config.RAW_RESULTS_DIR = occupied_raw_dir
            config.SUMMARY_RESULTS_DIR = occupied_sum_dir

    require_evidence_files(evidence_files)
    for path, digest in evidence_hashes.items():
        assert file_sha256(Path(path)) == digest, path

    print("offline-check: combinations and dataset slices OK")
    print(
        "offline-check: session 1 order "
        + ",".join(combination_label(m, h) for m, h in order_1)
    )
    print(
        "offline-check: session 2 order "
        + ",".join(combination_label(m, h) for m, h in order_2)
    )
    print(f"offline-check: random seed {config.E2_RANDOMIZATION_SEED}")
    print("offline-check: metrics exclude warmup; HTTP 200 JSON parse stays in latency n")
    print("offline-check: P95 linear method OK")
    print("offline-check: session-1 structural raw validation OK")
    print("offline-check: session-2 30-minute validation OK")
    print("offline-check: interruption and no-overwrite OK")
    print("offline-check: E1/E4 evidence unchanged")
    print("offline-check: no live HTTP client constructed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
