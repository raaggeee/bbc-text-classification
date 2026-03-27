from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import yaml
import os
import pandas as pd
import mlflow
import dagshub

dagshub_token = os.getenv("DAGSHUB_PAT")
if not dagshub_token:
    raise EnvironmentError(f"Couldn't find DAGSHUB PAT")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "raaggeee"
repo_name = "bbc-text-classification.mlflow"

mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}")


def open_yaml(file_path):
    with open(file_path, "r") as f:
        yaml_file = yaml.safe_load(f)

    return yaml_file

def open_df(file_path):
    df = pd.read_csv(file_path)
    return df

def save_df(file_path, filename, df):
    os.makedirs(file_path, exist_ok=True)
    save_path = os.path.join(file_path, filename)
    df.to_csv(save_path, index=False)
    return True

def get_yaml_content(yaml_content):
    data_path = yaml_content["model_train"]["data_path"]
    return data_path


def main():
    yaml_file = open_yaml("config.yaml")
    data_path = get_yaml_content(yaml_file)

    train_df = open_df(os.path.join(data_path, "train_df.csv"))

    X_train = train_df.iloc[:, :-1]
    y_train = train_df.iloc[:, -1]

    params_grids = {
        "n_estimators": [5, 10, 20, 30, 50, 100],
        "max_depth": [10, 20, 30, 40, 50, 100]
    }

    rf = RandomForestClassifier()

    grid_search = GridSearchCV(estimator=rf, param_grid=params_grids, cv=5, n_jobs=1, verbose=2)

    with mlflow.start_run(run_name="test-1"):
        grid_search.fit(X_train, y_train)
        
        for i in range(len(grid_search.cv_results_["params"])):
            with mlflow.start_run(run_name=f"Run {i}", nested=True):
                params=grid_search.cv_results_["params"][i]
                acc=grid_search.cv_results_["mean_test_score"][i]
                mlflow.log_param(f"Parameters of {i} run:", params)
                mlflow.log_metric(f"Accuracy of {i} run:", acc)

        best_params = grid_search.best_params_
        best_acc = grid_search.best_score_

        mlflow.log_param(f"Best Parameters", best_params)
        mlflow.log_metric(f"Best Accuracy", best_acc)

        mlflow.log_input(mlflow.data.from_pandas(train_df), "Training Data")

        mlflow.log_artifact(__file__)

        mlflow.set_tag("model", "Random Forest")
        mlflow.set_tag("cv", "Grid Search CV")

        mlflow.sklearn.log_model(grid_search.best_estimator_, name="Random Forest w CV", registered_model_name="Random-Forest-Model")

if __name__ == "__main__":
    main()
