import yaml
import os
import pandas as pd
import mlflow
import pickle
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from dotenv import load_dotenv
load_dotenv()

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

    xgb = XGBClassifier()

    with mlflow.start_run(run_name="xgb-normal"):
        xgb.fit(X_train, y_train)

        mlflow.log_input(mlflow.data.from_pandas(train_df), "Training Data")

        mlflow.log_artifact(__file__)

        mlflow.set_tag("model", "Random Forest")
        mlflow.set_tag("cv", "Grid Search CV")
        mlflow.set_tag("description", "XGBoost Classifier without Hyperparameter Tuning")

        #model test
        X_test = test_df.iloc[:, :-1]
        y_test = test_df.iloc[:, -1]
        y_pred = xgb.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")

        mlflow.log_metric(f"Accuracy on Test", accuracy)
        mlflow.log_metric(f"Precision on Test", precision)
        mlflow.log_metric(f"Recall on Test", recall)
        mlflow.log_metric(f"F1 on Test", f1)

        confusion = confusion_matrix(y_test, y_pred)
        sns.heatmap(confusion, annot=True,fmt=".2g")
        plt.savefig("metrics/confusion_matrix.png")
        mlflow.log_artifact("metrics/confusion_matrix.png")

        with open("model.pkl", "wb") as f:
            pickle.dump(xgb, f)


if __name__ == "__main__":
    main()
