from mlflow.tracking import MlflowClient
import mlflow
import dagshub
import os

dagshub_token = os.getenv("DAGSHUB_PAT")
if not dagshub_token:
    raise EnvironmentError(f"Couldn't find DAGSHUB PAT")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "raaggeee"
repo_name = "bbc-text-classification.mlflow"

mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}")

client = MlflowClient()

client.transition_model_version_stage(
    name="Random-Forest-Model",
    version=6,
    stage="Staging"
)