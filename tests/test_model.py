import os
import pandas as pd
import unittest
import dagshub
import mlflow
from sklearn.metrics import accuracy_score, f1_score
class TestModel(unittest.TestCase):

    @classmethod
    def init_class(cls):
        dagshub_token = os.getenv("DAGSHUB_PAT")
        if not dagshub_token:
            raise EnvironmentError("DAGSHUB_PAT not found..")
        
        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        dagshub_url = "https://dagshub.com"
        repo_owner = "raaggeee"
        repo_name = "bbc-text-classification.mlflow"

        mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}")

        cls.new_model = "Random-Forest-Model"
        cls.new_model_version = cls.get_latest_model(cls.new_model)
        cls.new_model_uri = f"models:/{cls.new_model_name}/{cls.new_model_version}"
        cls.model = mlflow.pyfunc.load_model(cls.new_model_uri)

        cls.holdout_data = pd.read_csv("data/external/test_df.csv")

    @staticmethod
    def get_latest_model(model_name, stage="Staging"):
        client = mlflow.MlflowClient()
        latest_version = client.get_latest_versions(model_name, stage=[stage])
        return latest_version[0].version

    def test_model_performance(self):
        X_test = self.holdout_data.iloc[:, :-1]
        y_test = self.holdout_data.iloc[:, -1]

        y_pred = self.model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        expected_accuracy = 0.50

        self.assertGreaterEqual(accuracy, expected_accuracy, f"Accuracy should be greater than {expected_accuracy}")

if __name__ == "__main__":
    unittest.main()

