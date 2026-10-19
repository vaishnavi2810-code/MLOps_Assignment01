# Load the model saved by train.py and classify wine samples.
#
# Usage (inside the container):
#   python predict.py                  # predicts the built-in sample
#   python predict.py 13.2 1.78 ...    # 13 feature values, in dataset order
import os
import sys

import joblib

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/app/output")

# First row of the Wine dataset (a class_0 wine).
SAMPLE = [14.23, 1.71, 2.43, 15.6, 127.0, 2.80, 3.06, 0.28, 2.29, 5.64,
          1.04, 3.92, 1065.0]

if __name__ == '__main__':
    model_path = os.path.join(OUTPUT_DIR, "wine_model.joblib")
    if not os.path.exists(model_path):
        sys.exit(f"No model at {model_path}. Run train.py first.")
    bundle = joblib.load(model_path)

    features = [float(v) for v in sys.argv[1:]] or SAMPLE
    n_expected = len(bundle["feature_names"])
    if len(features) != n_expected:
        sys.exit(f"Expected {n_expected} feature values "
                 f"({', '.join(bundle['feature_names'])}), got {len(features)}.")

    model = bundle["model"]
    pred = int(model.predict([features])[0])
    proba = model.predict_proba([features])[0]

    print(f"Predicted cultivar: {bundle['target_names'][pred]}")
    for name, p in zip(bundle["target_names"], proba):
        print(f"  {name}: {p:.3f}")
