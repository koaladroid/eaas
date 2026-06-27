from pathlib import Path

import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


def main():
    data = load_breast_cancer()
    X = data.data
    y = data.target
    feature_names = data.feature_names.tolist()
    target_names = data.target_names.tolist()

    print("Dataset loaded: Breast Cancer (569 samples, 30 features)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    random_forest = RandomForestClassifier(random_state=42)
    random_forest.fit(X_train, y_train)

    logistic_regression = LogisticRegression(solver="liblinear", random_state=42)
    logistic_regression.fit(X_train, y_train)

    print("Models trained: RandomForestClassifier, LogisticRegression")

    rf_score = random_forest.score(X_test, y_test)
    lr_score = logistic_regression.score(X_test, y_test)

    print(f"Random Forest Accuracy: {rf_score:.4f}")
    print(f"Logistic Regression Accuracy: {lr_score:.4f}")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(random_forest, ARTIFACTS_DIR / "random_forest.pkl")
    joblib.dump(logistic_regression, ARTIFACTS_DIR / "logistic_regression.pkl")
    joblib.dump(
        {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "feature_names": feature_names,
            "target_names": target_names,
        },
        ARTIFACTS_DIR / "training_data.pkl",
    )

    print(f"Artifacts saved to {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
