# Train a wine-cultivar classifier inside a Docker container.
#
# Dataset: UCI Wine (bundled with scikit-learn, so the image needs no network
# access at run time) - 178 samples, 13 chemical-analysis features, 3 classes.
# Model:   StandardScaler + GradientBoostingClassifier in a single Pipeline, so
#          the saved artifact applies the same scaling at prediction time.
import json
import os
from datetime import datetime, timezone

import joblib
from sklearn.datasets import load_wine
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Mount a host directory here (docker run -v ...) to keep the outputs after
# the container exits.
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/app/output")
RANDOM_STATE = int(os.environ.get("RANDOM_STATE", "42"))

if __name__ == '__main__':
    wine = load_wine()
    X, y = wine.data, wine.target

    # Stratify so all three cultivars appear in the test set in proportion.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.1, max_depth=3,
            random_state=RANDOM_STATE)),
    ])

    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "dataset": "UCI Wine (sklearn.datasets.load_wine)",
        "model": "StandardScaler + GradientBoostingClassifier",
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "cv_accuracy_mean": round(float(cv_scores.mean()), 4),
        "cv_accuracy_std": round(float(cv_scores.std()), 4),
        "test_accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "test_f1_macro": round(float(f1_score(y_test, y_pred, average="macro")), 4),
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Store the feature/class names with the model so predict.py can label
    # its output without reloading the dataset.
    joblib.dump({"model": model,
                 "feature_names": list(wine.feature_names),
                 "target_names": list(wine.target_names)},
                os.path.join(OUTPUT_DIR, "wine_model.joblib"))
    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    print(classification_report(y_test, y_pred, target_names=wine.target_names))
    print(json.dumps(metrics, indent=4))
    print(f"Model and metrics saved to {OUTPUT_DIR}")
    print("The model training was successful")
