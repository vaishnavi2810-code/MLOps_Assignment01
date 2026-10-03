import pickle, os, json
from sklearn.metrics import f1_score
import joblib, sys
import argparse

sys.path.insert(0, os.path.abspath('..'))

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True, help="Timestamp from GitHub Actions")
    args = parser.parse_args()

    # Access the timestamp
    timestamp = args.timestamp
    try:
        model_version = f'model_{timestamp}_dt_model'  # Use a timestamp as the version
        model = joblib.load(f'{model_version}.joblib')
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f'Failed to load the latest model: {model_version}.joblib. '
            'Run train_model.py with the same --timestamp first.') from e

    try:
        # The held-out split written by train_model.py.
        with open('data/data.pickle', 'rb') as data:
            X_test = pickle.load(data)

        with open('data/target.pickle', 'rb') as data:
            y_test = pickle.load(data)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            'Failed to load the test data from data/. '
            'Run train_model.py with the same --timestamp first.') from e

    y_predict = model.predict(X_test)
    metrics = {"F1_Score":f1_score(y_test, y_predict)}
    print(f"Evaluated on {X_test.shape[0]} held-out samples: {metrics}")

    # Save metrics to a JSON file
    with open(f'{timestamp}_metrics.json', 'w') as metrics_file:
        json.dump(metrics, metrics_file, indent=4)


