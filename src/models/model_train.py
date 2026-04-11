from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import yaml
import os
import pandas as pd
import mlflow
import dagshub
import pickle

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
    test_df = open_df(os.path.join(data_path, "test_df.csv"))

    X_train = train_df.iloc[:, :-1]
    y_train = train_df.iloc[:, -1]

    rf = RandomForestClassifier()

    mlflow.log_input(mlflow.data.from_pandas(train_df), "Training Data")

    mlflow.log_artifact(__file__)

    mlflow.set_tag("model", "Random Forest")
    mlflow.set_tag("cv", "Grid Search CV")

    mlflow.sklearn.log_model(rf.get_params, name="Random Forest w CV", registered_model_name="Random-Forest-Model")

    with open("model.pkl", "wb") as f:
        pickle.dump(rf, f)


if __name__ == "__main__":
    main()
