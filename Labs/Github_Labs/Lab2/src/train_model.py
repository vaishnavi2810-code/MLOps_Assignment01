import mlflow, datetime, os, pickle
from joblib import dump
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import sys
from sklearn.ensemble import RandomForestClassifier
import argparse

sys.path.insert(0, os.path.abspath('..'))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True, help="Timestamp from GitHub Actions")
    args = parser.parse_args()

    # Access the timestamp
    timestamp = args.timestamp

    # Use the timestamp in your script
    print(f"Timestamp received from GitHub Actions: {timestamp}")

    # Breast Cancer Wisconsin: 569 samples, 30 numeric features, binary target.
    # Ships inside scikit-learn, so CI needs no download.
    X, y = load_breast_cancer(return_X_y=True)

    # No random_state: each retrain sees a different 80% split, so every run
    # produces a genuinely different model to version.
    X_train, X_test, y_train, y_test = train_test_split(
                                            X, y,
                                            test_size=0.2,
                                            stratify=y,
                                        )

    # evaluate_model.py runs as a separate process, so the held-out set is
    # handed over on disk rather than in memory.
    os.makedirs('data/', exist_ok=True)
    with open('data/data.pickle', 'wb') as data:
        pickle.dump(X_test, data)

    with open('data/target.pickle', 'wb') as data:
        pickle.dump(y_test, data)

    # Recent MLflow versions refuse the ./mlruns file store unless opted in.
    # Fine here: mlruns/ is gitignored and thrown away with the CI runner.
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri("./mlruns")
    dataset_name = "Breast Cancer Wisconsin"
    current_time = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    experiment_name = f"{dataset_name}_{current_time}"
    experiment_id = mlflow.create_experiment(f"{experiment_name}")

    with mlflow.start_run(experiment_id=experiment_id,
                        run_name= f"{dataset_name}"):

        params = {
                    "dataset_name": dataset_name,
                    "number of training datapoint": X_train.shape[0],
                    "number of test datapoint": X_test.shape[0],
                    "number of dimensions": X_train.shape[1]}

        mlflow.log_params(params)


        forest = RandomForestClassifier(random_state=0)
        forest.fit(X_train, y_train)

        # Scored on the held-out split, not on the training rows.
        y_predict = forest.predict(X_test)
        mlflow.log_metrics({'Accuracy': accuracy_score(y_test, y_predict),
                            'F1 Score': f1_score(y_test, y_predict)})

        if not os.path.exists('models/'):
            # then create it.
            os.makedirs("models/")

        # After retraining the model
        model_version = f'model_{timestamp}'  # Use a timestamp as the version
        model_filename = f'{model_version}_dt_model.joblib'
        dump(forest, model_filename)

