"""E4: Invalid-input handling against POST /explain.

Talks only to the running service over HTTP. Does not import app, SHAP, or LIME.
Does not retrain models. Does not assume HTTP 400 for every invalid request.

This module performs no HTTP I/O on import. Full execution requires
--execute --confirm E4.

Protocol: EVALUATION_GUIDELINES.md
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
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

EXPERIMENT_ID = "E4"
CONFIRM_TOKEN = "E4"
MAX_STORED_BODY_CHARS = 50_000
EVALUATION_DEPENDENCY_PACKAGES = ("httpx", "joblib", "numpy")
RESULT_FILENAME_PREFIX = "e4_invalid_inputs"
MALFORMED_JSON_BYTES = b'{"model": "random_forest", "method": "shap", "input_data": ['
JSON_CONTENT_TYPE = "application/json"

# Documented types from EVALUATION_GUIDELINES.md / API contract. Observed
# status/type may differ (e.g. V5: enum rejection as HTTP 422 ValidationError).
TEST_CASES = (
    {
        "id": "E4.1",
        "name": "unsupported_model",
        "documented_error_type": "InvalidModelError",
    },
    {
        "id": "E4.2",
        "name": "unsupported_method",
        "documented_error_type": "InvalidMethodError",
    },
    {
        "id": "E4.3",
        "name": "incorrect_feature_count",
        "documented_error_type": "ValidationError",
    },
    {
        "id": "E4.4",
        "name": "nonnumeric_feature",
        "documented_error_type": "ValidationError",
    },
    {
        "id": "E4.5",
        "name": "missing_input_data",
        "documented_error_type": "ValidationError",
    },
    {
        "id": "E4.6",
        "name": "malformed_json",
        "documented_error_type": "ValidationError",
    },
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    return utc_now().strftime("%Y-%m-%dT%H:%M:%SZ")


def stamp_for_filename(moment: datetime | None = None) -> str:
    moment = moment or utc_now()
    return moment.strftime("%Y%m%d_%H%M%S")


def result_paths_for_stamp(stamp: str) -> tuple[Path, Path]:
    raw_path = config.RAW_RESULTS_DIR / f"{RESULT_FILENAME_PREFIX}_{stamp}.jsonl"
    summary_path = config.SUMMARY_RESULTS_DIR / f"{RESULT_FILENAME_PREFIX}_{stamp}.json"
    return raw_path, summary_path


def _run_stamp_candidates(preferred_stamp: str) -> Iterator[str]:
    yield preferred_stamp
    yield f"{preferred_stamp}_{utc_now().strftime('%f')}"
    suffix = 1
    while suffix <= 10_000:
        yield f"{preferred_stamp}_{suffix}"
        suffix += 1


def allocate_run_output_paths(preferred_stamp: str | None = None) -> tuple[str, Path, Path]:
    preferred_stamp = preferred_stamp or stamp_for_filename()
    config.RAW_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    config.SUMMARY_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for stamp in _run_stamp_candidates(preferred_stamp):
        raw_path, summary_path = result_paths_for_stamp(stamp)
        if raw_path.exists() or summary_path.exists():
            continue
        return stamp, raw_path, summary_path
    raise RuntimeError(
        "Unable to allocate unique E4 raw/summary filenames without overwrite."
    )


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


def explain_url() -> str:
    return f"{config.BASE_URL.rstrip('/')}{config.EXPLAIN_PATH}"


def load_valid_input_vector() -> list[float]:
    """Valid 30-feature vector from frozen X_test[0]. Artifact is not modified."""
    payload = joblib.load(config.TRAINING_DATA_PATH)
    x_test = np.asarray(payload["X_test"])
    if x_test.ndim != 2 or x_test.shape[1] != config.FEATURE_COUNT:
        raise ValueError(f"Unexpected X_test shape: {x_test.shape}.")
    if x_test.shape[0] < 1:
        raise ValueError("X_test is empty.")
    values = [float(v) for v in np.asarray(x_test[0]).tolist()]
    if len(values) != config.FEATURE_COUNT:
        raise ValueError("Baseline input_data does not have 30 features.")
    return values


def valid_explain_payload(input_data: list[float]) -> dict[str, Any]:
    return {
        "model": config.MODEL_RANDOM_FOREST,
        "method": config.METHOD_SHAP,
        "input_data": list(input_data),
    }


def build_request(case: dict[str, str], input_data: list[float]) -> dict[str, Any]:
    """Change only the field required for this invalid-input category."""
    name = case["name"]
    if name == "malformed_json":
        return {
            "mode": "malformed",
            "content": MALFORMED_JSON_BYTES,
            "headers": {"Content-Type": JSON_CONTENT_TYPE},
            "request_payload": None,
            "request_body_representation": {
                "raw_utf8": MALFORMED_JSON_BYTES.decode("utf-8"),
                "content_type": JSON_CONTENT_TYPE,
                "sent_as": "bytes",
            },
        }

    payload = valid_explain_payload(input_data)
    if name == "unsupported_model":
        payload["model"] = "unsupported_model"
    elif name == "unsupported_method":
        payload["method"] = "unsupported_method"
    elif name == "incorrect_feature_count":
        payload["input_data"] = input_data[: config.FEATURE_COUNT - 1]
    elif name == "nonnumeric_feature":
        mutated = list(input_data)
        mutated[-1] = "not_a_number"
        payload["input_data"] = mutated
    elif name == "missing_input_data":
        del payload["input_data"]
    else:
        raise ValueError(f"Unknown E4 case: {name}")

    return {
        "mode": "json_object",
        "payload": payload,
        "request_payload": payload,
        "request_body_representation": payload,
    }


def _clip_text(text: str) -> str:
    if len(text) <= MAX_STORED_BODY_CHARS:
        return text
    return text[:MAX_STORED_BODY_CHARS] + "...[truncated]"


def parse_response_body(text: str) -> tuple[Any | None, str | None]:
    try:
        return json.loads(text), None
    except json.JSONDecodeError as exc:
        return None, str(exc)


def looks_like_explanation(body: Any) -> bool:
    if not isinstance(body, dict):
        return False
    return (
        "prediction" in body
        and "feature_importance" in body
        and "model" in body
        and "method" in body
    )


def extract_error_envelope(body: Any) -> dict[str, Any]:
    if not isinstance(body, dict) or not isinstance(body.get("error"), dict):
        return {"present": False, "type": None, "message": None}
    error = body["error"]
    return {
        "present": True,
        "type": error.get("type"),
        "message": error.get("message"),
    }


def infer_layer(http_status: int | None, observed_type: str | None) -> str | None:
    if http_status is None:
        return None
    if http_status == 422:
        return "pydantic"
    if http_status == 400 and observed_type in {
        "InvalidModelError",
        "InvalidMethodError",
        "ValidationError",
    }:
        return "service"
    return "unknown"


def classify_attempt_outcome(
    *,
    transport_error: str | None,
    http_status: int | None,
    parsed_body: Any,
) -> dict[str, Any]:
    """Map transport/HTTP results to explicit E4 outcome categories."""
    if transport_error is not None:
        return {
            "rejected": False,
            "unexpected_success": False,
            "execution_outcome": "transport_failure",
            "observed_error_category": "transport_failure",
        }
    if http_status is None:
        return {
            "rejected": False,
            "unexpected_success": False,
            "execution_outcome": "unexpected_http_status",
            "observed_error_category": "unexpected_http_status",
        }
    if 400 <= http_status <= 499:
        return {
            "rejected": True,
            "unexpected_success": False,
            "execution_outcome": "http_client_rejection",
            "observed_error_category": "http_client_rejection",
        }
    if 500 <= http_status <= 599:
        return {
            "rejected": False,
            "unexpected_success": False,
            "execution_outcome": "server_error",
            "observed_error_category": "server_error",
        }
    if http_status == 200:
        if looks_like_explanation(parsed_body):
            return {
                "rejected": False,
                "unexpected_success": True,
                "execution_outcome": "unexpected_success",
                "observed_error_category": "unexpected_success",
            }
        return {
            "rejected": False,
            "unexpected_success": False,
            "execution_outcome": "unexpected_response",
            "observed_error_category": "unexpected_response",
        }
    return {
        "rejected": False,
        "unexpected_success": False,
        "execution_outcome": "unexpected_http_status",
        "observed_error_category": "unexpected_http_status",
    }


def post_explain(client: Any, request_spec: dict[str, Any]) -> Any:
    """Single POST /explain. Callers must not retry."""
    url = explain_url()
    if request_spec["mode"] == "malformed":
        return client.post(
            url,
            content=request_spec["content"],
            headers=request_spec["headers"],
        )
    return client.post(url, json=request_spec["payload"])


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


def execute_one_attempt(
    *,
    client: Any,
    case: dict[str, str],
    repetition: int,
    input_data: list[float],
) -> dict[str, Any]:
    request_spec = build_request(case, input_data)
    http_status: int | None = None
    response_text = ""
    parsed_body: Any | None = None
    json_error: str | None = None
    transport_error: str | None = None

    try:
        response = post_explain(client, request_spec)
        http_status = int(response.status_code)
        response_text = response.text or ""
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        transport_error = f"{type(exc).__name__}: {exc}"

    if transport_error is None:
        parsed_body, json_error = parse_response_body(response_text)

    stored_body: Any
    if parsed_body is not None:
        stored_body = parsed_body
    elif response_text:
        stored_body = _clip_text(response_text)
    else:
        stored_body = None

    envelope = extract_error_envelope(parsed_body)
    classification = classify_attempt_outcome(
        transport_error=transport_error,
        http_status=http_status,
        parsed_body=parsed_body,
    )
    rejected = classification["rejected"]
    unexpected_success = classification["unexpected_success"]
    execution_outcome = classification["execution_outcome"]
    observed_error_category = classification["observed_error_category"]

    documented_type = case["documented_error_type"]
    observed_type = envelope["type"]
    documented_type_match = bool(
        rejected and observed_type is not None and observed_type == documented_type
    )

    return {
        "experiment_id": EXPERIMENT_ID,
        "timestamp_utc": utc_now_iso(),
        "test_case_id": case["id"],
        "test_case_name": case["name"],
        "repetition": repetition,
        "request_payload": request_spec["request_payload"],
        "request_body_representation": request_spec["request_body_representation"],
        "malformed_json": request_spec["mode"] == "malformed",
        "http_status": http_status,
        "response_body": stored_body,
        "response_json_parse_error": json_error,
        "transport_error": transport_error,
        "rejected": rejected,
        "unexpected_success": unexpected_success,
        "error_envelope_present": envelope["present"],
        "observed_error_type": observed_type,
        "observed_error_message": envelope["message"],
        "documented_error_type": documented_type,
        "documented_type_match": documented_type_match,
        "inferred_layer": infer_layer(http_status, observed_type),
        "observed_error_category": observed_error_category,
        "execution_outcome": execution_outcome,
    }


def run_e4(
    client: Any,
    input_data: list[float],
    raw_writer: RawJsonlWriter,
) -> tuple[list[dict[str, Any]], bool]:
    records: list[dict[str, Any]] = []
    interrupted = False
    try:
        for case in TEST_CASES:
            for repetition in range(1, config.E4_REPETITIONS_PER_CATEGORY + 1):
                record = execute_one_attempt(
                    client=client,
                    case=case,
                    repetition=repetition,
                    input_data=input_data,
                )
                records.append(record)
                raw_writer.append(record)
    except KeyboardInterrupt:
        interrupted = True
    return records, interrupted


def _status_distribution(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in records:
        key = "none" if row["http_status"] is None else str(row["http_status"])
        counts[key] = counts.get(key, 0) + 1
    return counts


def _by_test_case(records: list[dict[str, Any]]) -> dict[str, Any]:
    breakdown: dict[str, Any] = {}
    for case in TEST_CASES:
        subset = [row for row in records if row["test_case_id"] == case["id"]]
        statuses = _status_distribution(subset)
        observed_types = sorted(
            {row["observed_error_type"] for row in subset if row["observed_error_type"]}
        )
        breakdown[case["id"]] = {
            "test_case_name": case["name"],
            "documented_error_type": case["documented_error_type"],
            "planned_attempts": config.E4_REPETITIONS_PER_CATEGORY,
            "attempts": len(subset),
            "http_4xx_rejections": _count_outcome(subset, "http_client_rejection"),
            "http_5xx_server_errors": _count_outcome(subset, "server_error"),
            "rejected": sum(1 for row in subset if row["rejected"]),
            "unexpected_success": _count_outcome(subset, "unexpected_success"),
            "unexpected_http_200_responses": _count_outcome(
                subset, "unexpected_response"
            ),
            "unexpected_http_statuses": _count_outcome(
                subset, "unexpected_http_status"
            ),
            "transport_errors": _count_outcome(subset, "transport_failure"),
            "http_status_distribution": statuses,
            "observed_error_types": observed_types,
            "documented_type_matches": sum(
                1 for row in subset if row["documented_type_match"]
            ),
            "inferred_layers": sorted(
                {row["inferred_layer"] for row in subset if row["inferred_layer"]}
            ),
        }
    return breakdown


def _count_outcome(records: list[dict[str, Any]], outcome: str) -> int:
    return sum(1 for row in records if row["execution_outcome"] == outcome)


def build_summary(
    records: list[dict[str, Any]],
    metadata: dict[str, Any],
    *,
    interrupted: bool = False,
) -> dict[str, Any]:
    completed = len(records)
    planned = config.E4_TOTAL_REQUESTS
    http_4xx_rejections = _count_outcome(records, "http_client_rejection")
    http_5xx_server_errors = _count_outcome(records, "server_error")
    unexpected_success = _count_outcome(records, "unexpected_success")
    unexpected_http_200 = _count_outcome(records, "unexpected_response")
    unexpected_http_statuses = _count_outcome(records, "unexpected_http_status")
    transport_errors = _count_outcome(records, "transport_failure")
    rejected = sum(1 for row in records if row["rejected"])

    if interrupted or completed < planned:
        execution_status = "interrupted" if interrupted else "incomplete"
        overall_outcome = "incomplete"
    else:
        execution_status = "completed"
        overall_outcome = "completed"

    return {
        "experiment_id": EXPERIMENT_ID,
        "execution_status": execution_status,
        "overall_outcome": overall_outcome,
        "completed_experiment": execution_status == "completed" and completed == planned,
        "success_criterion_note": (
            "E4 records observed rejection behavior. Only HTTP 4xx counts as "
            "rejected. HTTP 5xx is a server error, not a client-input rejection. "
            "HTTP 200 with an explanation is unexpected success; HTTP 200 without "
            "an explanation is unexpected_response."
        ),
        "protocol_expectations_note": (
            "documented_error_type values come from EVALUATION_GUIDELINES.md / "
            "the written error envelope. Deviations from those types are recorded, "
            "not silently passed."
        ),
        "counts": {
            "planned_attempts": planned,
            "attempts_completed": completed,
            "rejected": rejected,
            "http_4xx_rejections": http_4xx_rejections,
            "http_5xx_server_errors": http_5xx_server_errors,
            "unexpected_success": unexpected_success,
            "unexpected_http_200_responses": unexpected_http_200,
            "unexpected_http_statuses": unexpected_http_statuses,
            "transport_error_count": transport_errors,
        },
        "http_status_distribution": _status_distribution(records),
        "by_test_case": _by_test_case(records),
        "metadata": metadata,
    }


def write_summary(summary: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(summary, indent=2, default=str) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(payload)
    return path


def build_run_metadata(*, run_stamp: str) -> dict[str, Any]:
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
        "baseline_input": "artifacts/training_data.pkl X_test[0]",
        "test_cases": [
            {"id": case["id"], "name": case["name"]} for case in TEST_CASES
        ],
        "expected_total_requests": config.E4_TOTAL_REQUESTS,
        "repetitions_per_category": config.E4_REPETITIONS_PER_CATEGORY,
    }


def run_offline_checks() -> int:
    import hashlib
    import tempfile
    from collections import Counter

    def file_sha256(path: Path) -> str | None:
        if not path.exists():
            return None
        digest = hashlib.sha256()
        digest.update(path.read_bytes())
        return digest.hexdigest()

    e1_raw = config.RAW_RESULTS_DIR / "e1_contract_20261102_204631.jsonl"
    e1_summary = config.SUMMARY_RESULTS_DIR / "e1_contract_20261102_204631.json"
    e1_raw_hash = file_sha256(e1_raw)
    e1_summary_hash = file_sha256(e1_summary)

    input_data = load_valid_input_vector()
    assert len(input_data) == config.FEATURE_COUNT
    assert config.E4_TOTAL_REQUESTS == 18
    assert len(TEST_CASES) * config.E4_REPETITIONS_PER_CATEGORY == 18

    malformed = build_request(TEST_CASES[5], input_data)
    assert malformed["mode"] == "malformed"
    assert malformed["content"] == MALFORMED_JSON_BYTES
    assert "json" not in malformed
    assert not _is_valid_json_bytes(malformed["content"])

    short = build_request(TEST_CASES[2], input_data)
    assert len(short["payload"]["input_data"]) == 29
    missing = build_request(TEST_CASES[4], input_data)
    assert "input_data" not in missing["payload"]

    class FakeResp:
        def __init__(self, code: int, body: str):
            self.status_code = code
            self.text = body

    class RecordingClient:
        def __init__(self, interrupt_after: int | None = None, fail_transport: bool = False):
            self.calls: list[dict[str, Any]] = []
            self.interrupt_after = interrupt_after
            self.fail_transport = fail_transport

        def post(self, url: str, **kwargs: Any) -> FakeResp:
            self.calls.append({"url": url, **kwargs})
            if self.interrupt_after is not None and len(self.calls) > self.interrupt_after:
                raise KeyboardInterrupt
            if self.fail_transport:
                raise TimeoutError("mocked timeout")
            if "json" in kwargs and "content" in kwargs:
                raise AssertionError("malformed JSON must not also use json=")
            if "content" in kwargs:
                assert kwargs.get("headers", {}).get("Content-Type") == JSON_CONTENT_TYPE
                assert kwargs["content"] == MALFORMED_JSON_BYTES
                try:
                    json.loads(kwargs["content"])
                    raise AssertionError("malformed content was valid JSON")
                except (json.JSONDecodeError, TypeError, UnicodeDecodeError):
                    pass
            body = {
                "error": {
                    "type": "ValidationError",
                    "message": "Invalid request body.",
                }
            }
            return FakeResp(422, json.dumps(body))

    class StatusClient:
        def __init__(
            self,
            status: int | None = None,
            body: str = "",
            transport: bool = False,
        ):
            self.status = status
            self.body = body
            self.transport = transport
            self.calls = 0

        def post(self, url: str, **kwargs: Any) -> FakeResp:
            self.calls += 1
            if self.transport:
                raise TimeoutError("mocked timeout")
            assert self.status is not None
            return FakeResp(self.status, self.body)

    validation_envelope = json.dumps(
        {"error": {"type": "ValidationError", "message": "Invalid request body."}}
    )
    explanation_body = json.dumps(
        {
            "model": config.MODEL_RANDOM_FOREST,
            "method": config.METHOD_SHAP,
            "prediction": 0,
            "probability": 0.5,
            "feature_importance": {"mean radius": 0.1},
        }
    )
    case = TEST_CASES[0]

    rec_422 = execute_one_attempt(
        client=StatusClient(422, validation_envelope),
        case=case,
        repetition=1,
        input_data=input_data,
    )
    assert rec_422["rejected"] is True
    assert rec_422["unexpected_success"] is False
    assert rec_422["execution_outcome"] == "http_client_rejection"
    assert rec_422["observed_error_type"] == "ValidationError"
    assert rec_422["error_envelope_present"] is True

    rec_500 = execute_one_attempt(
        client=StatusClient(500, json.dumps({"error": {"type": "InternalServerError"}})),
        case=case,
        repetition=1,
        input_data=input_data,
    )
    assert rec_500["rejected"] is False
    assert rec_500["execution_outcome"] == "server_error"
    assert rec_500["unexpected_success"] is False

    rec_200_ok = execute_one_attempt(
        client=StatusClient(200, explanation_body),
        case=case,
        repetition=1,
        input_data=input_data,
    )
    assert rec_200_ok["rejected"] is False
    assert rec_200_ok["unexpected_success"] is True
    assert rec_200_ok["execution_outcome"] == "unexpected_success"

    rec_200_bad = execute_one_attempt(
        client=StatusClient(200, json.dumps({"unexpected": True})),
        case=case,
        repetition=1,
        input_data=input_data,
    )
    assert rec_200_bad["rejected"] is False
    assert rec_200_bad["unexpected_success"] is False
    assert rec_200_bad["execution_outcome"] == "unexpected_response"

    rec_timeout = execute_one_attempt(
        client=StatusClient(transport=True),
        case=case,
        repetition=1,
        input_data=input_data,
    )
    assert rec_timeout["rejected"] is False
    assert rec_timeout["execution_outcome"] == "transport_failure"
    assert rec_timeout["transport_error"] is not None

    rec_other = execute_one_attempt(
        client=StatusClient(204, ""),
        case=case,
        repetition=1,
        input_data=input_data,
    )
    assert rec_other["rejected"] is False
    assert rec_other["execution_outcome"] == "unexpected_http_status"

    mixed_summary = build_summary(
        [rec_422, rec_500, rec_200_ok, rec_200_bad, rec_timeout, rec_other],
        {"n": 6},
        interrupted=False,
    )
    assert mixed_summary["counts"]["http_4xx_rejections"] == 1
    assert mixed_summary["counts"]["http_5xx_server_errors"] == 1
    assert mixed_summary["counts"]["unexpected_success"] == 1
    assert mixed_summary["counts"]["unexpected_http_200_responses"] == 1
    assert mixed_summary["counts"]["unexpected_http_statuses"] == 1
    assert mixed_summary["counts"]["transport_error_count"] == 1
    assert mixed_summary["counts"]["rejected"] == 1

    with tempfile.TemporaryDirectory() as tmp:
        raw_path = Path(tmp) / "e4_offline.jsonl"
        client = RecordingClient()
        with RawJsonlWriter(raw_path) as writer:
            records, interrupted = run_e4(client, input_data, writer)
        assert interrupted is False
        assert len(records) == 18
        assert len(client.calls) == 18
        names = Counter(row["test_case_name"] for row in records)
        assert names == Counter({case["name"]: 3 for case in TEST_CASES})
        malformed_calls = [call for call in client.calls if "content" in call]
        json_calls = [call for call in client.calls if "json" in call]
        assert len(malformed_calls) == 3
        assert len(json_calls) == 15
        for call in malformed_calls:
            assert "json" not in call
            assert call["content"] == MALFORMED_JSON_BYTES
        summary = build_summary(records, {"n": 18}, interrupted=False)
        assert summary["execution_status"] == "completed"
        assert summary["overall_outcome"] == "completed"
        assert summary["completed_experiment"] is True
        assert summary["counts"]["attempts_completed"] == 18
        assert summary["counts"]["rejected"] == 18
        assert summary["counts"]["http_4xx_rejections"] == 18
        assert summary["counts"]["http_5xx_server_errors"] == 0
        assert summary["counts"]["unexpected_success"] == 0
        assert summary["counts"]["unexpected_http_200_responses"] == 0
        assert summary["counts"]["unexpected_http_statuses"] == 0
        assert summary["counts"]["transport_error_count"] == 0
        assert summary["http_status_distribution"] == {"422": 18}
        assert all(row["execution_outcome"] == "http_client_rejection" for row in records)

        transport_client = RecordingClient(fail_transport=True)
        transport_path = Path(tmp) / "e4_transport.jsonl"
        with RawJsonlWriter(transport_path) as writer:
            transport_records, interrupted = run_e4(
                transport_client, input_data, writer
            )
        assert interrupted is False
        assert len(transport_records) == 18
        assert len(transport_client.calls) == 18
        assert all(row["transport_error"] for row in transport_records)
        assert all(row["rejected"] is False for row in transport_records)
        assert all(
            row["execution_outcome"] == "transport_failure" for row in transport_records
        )
        transport_summary = build_summary(
            transport_records, {"n": 18}, interrupted=False
        )
        assert transport_summary["counts"]["transport_error_count"] == 18
        assert transport_summary["counts"]["http_4xx_rejections"] == 0
        assert transport_summary["counts"]["rejected"] == 0
        assert transport_summary["counts"]["unexpected_success"] == 0

        interrupt_client = RecordingClient(interrupt_after=5)
        partial_path = Path(tmp) / "e4_partial.jsonl"
        with RawJsonlWriter(partial_path) as writer:
            partial_records, interrupted = run_e4(
                interrupt_client, input_data, writer
            )
        assert interrupted is True
        assert len(partial_records) == 5
        assert len(interrupt_client.calls) == 6
        assert len(partial_path.read_text(encoding="utf-8").splitlines()) == 5
        partial_summary = build_summary(
            partial_records, {"n": 5}, interrupted=True
        )
        assert partial_summary["execution_status"] == "interrupted"
        assert partial_summary["overall_outcome"] == "incomplete"
        assert partial_summary["completed_experiment"] is False
        assert partial_summary["counts"]["planned_attempts"] == 18
        assert partial_summary["counts"]["attempts_completed"] == 5

        saved_raw_dir = config.RAW_RESULTS_DIR
        saved_summary_dir = config.SUMMARY_RESULTS_DIR
        try:
            config.RAW_RESULTS_DIR = Path(tmp) / "raw_results"
            config.SUMMARY_RESULTS_DIR = Path(tmp) / "summary_results"
            config.RAW_RESULTS_DIR.mkdir()
            config.SUMMARY_RESULTS_DIR.mkdir()
            occupied_stamp = "20260101_120000"
            occupied_raw, occupied_summary = result_paths_for_stamp(occupied_stamp)
            occupied_raw.write_text("EXISTING_RAW\n", encoding="utf-8")
            occupied_summary.write_text("EXISTING_SUMMARY\n", encoding="utf-8")
            unique_stamp, unique_raw, unique_summary = allocate_run_output_paths(
                occupied_stamp
            )
            assert unique_stamp != occupied_stamp
            try:
                with RawJsonlWriter(occupied_raw):
                    raise AssertionError("overwrote raw")
            except FileExistsError:
                pass
            try:
                write_summary({"must_not_write": True}, occupied_summary)
                raise AssertionError("overwrote summary")
            except FileExistsError:
                pass
            assert occupied_raw.read_text(encoding="utf-8") == "EXISTING_RAW\n"
            assert occupied_summary.read_text(encoding="utf-8") == "EXISTING_SUMMARY\n"
            write_summary({"probe": True}, unique_summary)
            assert unique_raw.name.endswith(f"{unique_stamp}.jsonl")
            assert unique_summary.name.endswith(f"{unique_stamp}.json")
        finally:
            config.RAW_RESULTS_DIR = saved_raw_dir
            config.SUMMARY_RESULTS_DIR = saved_summary_dir

    assert file_sha256(e1_raw) == e1_raw_hash
    assert file_sha256(e1_summary) == e1_summary_hash

    print("offline-check: baseline vector and case construction OK")
    print("offline-check: 18 mocked attempts, 3 per category")
    print("offline-check: malformed JSON sent as raw content")
    print("offline-check: outcome classification (4xx/5xx/200/transport) OK")
    print("offline-check: transport failures recorded without retry")
    print("offline-check: interruption preserves JSONL and incomplete summary")
    print("offline-check: filename collision / no-overwrite OK")
    print("offline-check: E1 result files unchanged")
    print("offline-check: no live HTTP client constructed")
    return 0


def _is_valid_json_bytes(raw: bytes) -> bool:
    try:
        json.loads(raw)
        return True
    except json.JSONDecodeError:
        return False


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "E4 invalid-input evaluation: 6 categories × 3 repetitions "
            f"= {config.E4_TOTAL_REQUESTS} sequential POST /explain requests."
        )
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run the full E4 HTTP experiment (requires --confirm E4).",
    )
    parser.add_argument(
        "--confirm",
        metavar="TOKEN",
        help=f"Safety token. Must be '{CONFIRM_TOKEN}' together with --execute.",
    )
    parser.add_argument(
        "--offline-check",
        action="store_true",
        help="Run mocked case/IO checks. No HTTP.",
    )
    return parser.parse_args(argv)


def execute_experiment() -> int:
    import httpx

    input_data = load_valid_input_vector()
    run_stamp, raw_path, summary_path = allocate_run_output_paths(stamp_for_filename())
    metadata = build_run_metadata(run_stamp=run_stamp)
    timeout = httpx.Timeout(config.HTTP_TIMEOUT_SECONDS)
    with RawJsonlWriter(raw_path) as writer:
        with httpx.Client(timeout=timeout) as client:
            records, interrupted = run_e4(client, input_data, writer)
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
    if interrupted or not summary["completed_experiment"]:
        return 130 if interrupted else 1
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.offline_check:
        return run_offline_checks()
    if args.execute:
        if args.confirm != CONFIRM_TOKEN:
            print(
                "Refusing to run E4: pass --execute --confirm E4 "
                "to send 18 sequential invalid-input requests to the public "
                "evaluation URL.",
                file=sys.stderr,
            )
            return 2
        return execute_experiment()
    print(
        "No action taken. Use --help, --offline-check, "
        "or --execute --confirm E4 to run the full experiment."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
