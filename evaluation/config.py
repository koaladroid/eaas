"""Centralized constants for EaaS empirical evaluation (E1–E4).

Independent of production modules. Do not import app, SHAP, or LIME here.
Protocol source: EVALUATION_GUIDELINES.md
"""

from pathlib import Path

# Repository root (parent of evaluation/).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVALUATION_DIR = Path(__file__).resolve().parent

# --- Deployment (public evaluation target) ---
BASE_URL = "https://eaas-production-df33.up.railway.app"
HEALTH_PATH = "/health"
PREDICT_PATH = "/predict"
EXPLAIN_PATH = "/explain"

# --- Models, methods, and explicit combinations ---
MODEL_RANDOM_FOREST = "random_forest"
MODEL_LOGISTIC_REGRESSION = "logistic_regression"
SUPPORTED_MODELS = (MODEL_RANDOM_FOREST, MODEL_LOGISTIC_REGRESSION)

METHOD_SHAP = "shap"
METHOD_LIME = "lime"
SUPPORTED_METHODS = (METHOD_SHAP, METHOD_LIME)

COMBINATIONS = (
    (MODEL_RANDOM_FOREST, METHOD_SHAP),
    (MODEL_RANDOM_FOREST, METHOD_LIME),
    (MODEL_LOGISTIC_REGRESSION, METHOD_SHAP),
    (MODEL_LOGISTIC_REGRESSION, METHOD_LIME),
)

# --- Dataset (existing artifacts; do not retrain) ---
TRAINING_DATA_PATH = PROJECT_ROOT / "artifacts" / "training_data.pkl"
FEATURE_COUNT = 30
X_TEST_N_AVAILABLE = 114

# Slice bounds are half-open [start, end), matching X_test[start:end].
E1_SLICE_START = 0
E1_SLICE_END = 30
WARMUP_SLICE_START = 0
WARMUP_SLICE_END = 10
MEASURED_SLICE_START = 10
MEASURED_SLICE_END = 110

# --- E1: contract consistency ---
E1_REQUESTS_PER_COMBINATION = E1_SLICE_END - E1_SLICE_START
E1_TOTAL_REQUESTS = len(COMBINATIONS) * E1_REQUESTS_PER_COMBINATION

# --- E2: sequential runtime ---
E2_N_SESSIONS = 2
E2_WARMUP_PER_COMBINATION = WARMUP_SLICE_END - WARMUP_SLICE_START
E2_MEASURED_ATTEMPTS_PER_COMBINATION_PER_SESSION = (
    MEASURED_SLICE_END - MEASURED_SLICE_START
)
E2_MEASURED_TOTAL_ATTEMPTS = (
    E2_N_SESSIONS
    * len(COMBINATIONS)
    * E2_MEASURED_ATTEMPTS_PER_COMBINATION_PER_SESSION
)
HTTP_TIMEOUT_SECONDS = 120
# Minimum gap between the two E2 sessions.
E2_MIN_SESSION_SEPARATION_SECONDS = 30 * 60
# Seed for randomizing combination order between sessions (sample order stays fixed).
E2_RANDOMIZATION_SEED = 42
E2_SEQUENTIAL_ONLY = True
E2_RETRY_FAILED_REQUESTS = False  # failed/timeout attempts still count toward 100
E2_P95_PERCENTILE_METHOD = "linear"  # np.percentile(..., method="linear")

# --- E4: invalid-input handling ---
E4_CATEGORIES = (
    "unsupported_model",
    "unsupported_method",
    "incorrect_feature_count",
    "nonnumeric_feature",
    "missing_input_data",
    "malformed_json",
)
E4_REPETITIONS_PER_CATEGORY = 3
E4_TOTAL_REQUESTS = len(E4_CATEGORIES) * E4_REPETITIONS_PER_CATEGORY

# --- Output locations ---
RESULTS_DIR = EVALUATION_DIR / "results"
RAW_RESULTS_DIR = RESULTS_DIR / "raw"
SUMMARY_RESULTS_DIR = RESULTS_DIR / "summaries"
FIGURES_DIR = RESULTS_DIR / "figures"
ARCHITECTURE_AUDIT_DIR = EVALUATION_DIR / "architecture-audit"
FIXTURES_DIR = EVALUATION_DIR / "fixtures"

assert E1_TOTAL_REQUESTS == 120
assert E2_MEASURED_TOTAL_ATTEMPTS == 800
assert E4_TOTAL_REQUESTS == 18
assert MEASURED_SLICE_END <= X_TEST_N_AVAILABLE
