# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Scope

The working directory is `Labs/Github_Labs/Lab2` inside the DADS 7305 / IE 7305 MLOps teaching repo (upstream: `raminmohammadi/MLOps`). The repo is a monorepo of independent labs under `Labs/<Topic>_Labs/<LabN>/`; each lab has its own `requirements.txt` and is otherwise self-contained. There is no repo-wide build, lint, or test entry point.

Lab2 teaches CI-driven model retraining, evaluation, and timestamp-based versioning with GitHub Actions.

## Commands

```bash
# Install (from this lab directory)
pip install -r requirements.txt

# Run the pipeline the way CI does — ALWAYS from the repo root, not from Lab2
cd ../../..
timestamp=$(date '+%Y%m%d%H%M%S')
python Labs/Github_Labs/Lab2/src/train_model.py    --timestamp "$timestamp"
python Labs/Github_Labs/Lab2/src/evaluate_model.py --timestamp "$timestamp"
```

Both scripts require `--timestamp` and write their outputs into the **current working directory**, not into the lab directory:

- `train_model.py` → `model_<timestamp>_dt_model.joblib`, plus `data/{data,target}.pickle` and `mlruns/`
- `evaluate_model.py` → `<timestamp>_metrics.json`

The `os.makedirs('models/')` call in `train_model.py` is vestigial — the `.joblib` lands in the CWD, not in `models/`. The workflow `mv`s both artifacts into `Labs/Github_Labs/Lab2/models/` and `.../metrics/` afterwards. **If you change an output filename in a script, you must change the matching `mv` and `git add` lines in both workflow copies.**

`data/{data,target}.pickle` is the real train→evaluate handoff: the two scripts run as separate processes, so `train_model.py` pickles the held-out test split and `evaluate_model.py` loads it back. Don't delete those writes — evaluation has no data without them.

## The two-copy workflow pattern (important)

Every GitHub lab keeps workflow YAML in **two** places, and they are not the same file:

- `Labs/Github_Labs/Lab2/workflows/*.yml` — teaching reference, written for a standalone fork where the lab is the repo root (relative paths like `./src/train_model.py`, older `actions/checkout@v2`). **GitHub never runs these.**
- `.github/workflows/github_lab2_*.yml` — the versions that actually run, rewritten with monorepo paths (`Labs/Github_Labs/Lab2/src/...`), `checkout@v4`/`setup-python@v4`, Python 3.9.

Editing only the lab-local copy changes nothing in CI. When changing workflow behavior, update both and keep the path rewriting consistent.

Lab2's live workflows:

| File | Trigger | Purpose |
| --- | --- | --- |
| `.github/workflows/github_lab2_model_calibration_on_push.yml` | push to `main` | retrain → evaluate → move artifacts → commit & push back to `main` |
| `.github/workflows/github_lab2_model_calibration.yml` | cron `0 0 * * *` | same pipeline, nightly |

Note: `.github/workflows/github_lab2_unittest_action.yml` is misnamed — it installs and runs **Lab1's** unittests, not Lab2's. `Labs/Github_Labs/Lab2/test/` contains only an empty `__init__.py`; Lab2 has no tests.

Both Lab2 workflows commit generated models/metrics back to `main`, which requires two things that are easy to lose:

- `permissions: contents: write` on the job. `GITHUB_TOKEN` is read-only by default in most repos, and without this the final `git push` fails with a 403 after every other step has passed.
- Authorship via `${{ github.actor }}` rather than a hardcoded identity, so commits are attributed to whoever triggered the run. (Upstream hardcodes `raminmohammadi`.)

Pushes made with `GITHUB_TOKEN` do not retrigger workflows, so the on-push workflow committing to `main` does not loop.

## Dataset

The lab was moved off synthetic `make_classification` data onto **Breast Cancer Wisconsin** (`sklearn.datasets.load_breast_cancer`): 569 rows, 30 numeric features, binary target, bundled inside scikit-learn so CI needs no download.

Two constraints to preserve when touching the data:

- `f1_score(...)` is called with no `average=` argument, so it defaults to `binary`. A multi-class dataset crashes both scripts.
- `train_test_split` is called **without** `random_state`, deliberately. Each run trains on a different 80%, so every retrain produces a distinct model and a slightly different F1 — which is what gives the timestamped `models/` directory something to version. Pinning the seed makes every run byte-identical and the workflow's "No changes to commit." branch fires forever.

## Known rough edges

Inherited from upstream teaching code; fix only if the task asks for it.

- `mlflow.create_experiment` is keyed to `%y%m%d_%H%M%S`, so two runs in the same second collide with `MlflowException: Experiment already exists`.
- `train_model.py` sets `MLFLOW_ALLOW_FILE_STORE=true` because current MLflow versions reject the `./mlruns` file store. Acceptable here only because `mlruns/` is gitignored and nothing ever reads the tracking data back; switch to `sqlite:///mlflow.db` if runs ever need to survive.
- `requirements.txt` pins no versions, while the workflows pin `python-version: 3.9`. Local and CI resolve to different library versions.
- `src/test.ipynb` is a scratch notebook from an earlier RCV1-based version of the lab (`fetch_rcv1`, `DecisionTreeClassifier`) and matches neither the current scripts nor the current dataset.
