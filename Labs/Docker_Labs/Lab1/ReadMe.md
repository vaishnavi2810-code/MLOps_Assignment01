# Docker Lab 1: Wine Classifier in a Container

Train, evaluate, and serve predictions from a scikit-learn model entirely inside a Docker container.

## What changed from the original lab

| | Original | This version |
| --- | --- | --- |
| Dataset | Iris (150 samples, 4 features) | **UCI Wine**: 178 samples, 13 chemical-analysis features, 3 cultivars |
| Model | RandomForestClassifier | **StandardScaler + GradientBoostingClassifier** `Pipeline` |
| Evaluation | none | 5-fold CV accuracy, test accuracy, macro F1, per-class report → `metrics.json` |
| Split | 80/20 | 75/25, **stratified** by class |
| Output | `iris_model.pkl` lost when the container exits | `wine_model.joblib` + `metrics.json` in `/app/output`, kept by mounting a volume |
| Inference | none | `predict.py` loads the saved model and classifies a sample |
| Base image | `python:3.10` (~2.06 GB) | `python:3.12-slim` (~561 MB) |
| Dockerfile | single layer, runs as root | dependency layer cached separately, runs as non-root `appuser`, `.dockerignore` |
| Dependencies | unpinned | pinned (`scikit-learn==1.7.2`, `joblib==1.6.0`, `numpy==2.2.6`) |

The Wine dataset ships inside scikit-learn, so the container needs no network access at run time.

## Files

```
Lab1/
├── Dockerfile          # image definition
├── .dockerignore       # keeps outputs/tarballs out of the build context
├── ReadMe.md
└── src/
    ├── requirements.txt
    ├── train.py        # train + evaluate, writes model and metrics to $OUTPUT_DIR
    └── predict.py      # loads the saved model and predicts
```

## Run it

From `Labs/Docker_Labs/Lab1`:

```bash
# 1. Build the image
docker build -t wine-lab1:v1 .

# 2. Train. Mounting ./output keeps the model and metrics on the host.
docker run --rm -v "$PWD/output:/app/output" wine-lab1:v1

# 3. Predict with the saved model (built-in sample, a class_0 wine)
docker run --rm -v "$PWD/output:/app/output" wine-lab1:v1 python predict.py

# 4. Predict your own sample: 13 values in dataset feature order
docker run --rm -v "$PWD/output:/app/output" wine-lab1:v1 \
  python predict.py 12.37 0.94 1.36 10.6 88 1.98 0.57 0.28 0.42 1.95 1.05 1.82 520

# 5. Export the image as a tarball and load it elsewhere
docker save wine-lab1:v1 > wine_lab1.tar
docker load < wine_lab1.tar
```

On Linux, add `--user "$(id -u):$(id -g)"` to `docker run` if the container can't write to the mounted `output/` directory.

### Configuration

| Env var | Default | Purpose |
| --- | --- | --- |
| `OUTPUT_DIR` | `/app/output` | where `train.py` writes and `predict.py` reads |
| `RANDOM_STATE` | `42` | seed for the split and the model (`-e RANDOM_STATE=7`) |

## Sample output

```
              precision    recall  f1-score   support

     class_0       0.94      1.00      0.97        15
     class_1       0.94      0.94      0.94        18
     class_2       1.00      0.92      0.96        12

    accuracy                           0.96        45
```

`metrics.json`:

```json
{
    "cv_accuracy_mean": 0.9023,
    "test_accuracy": 0.9556,
    "test_f1_macro": 0.9562
}
```

`output/` and `*.tar` are gitignored.
