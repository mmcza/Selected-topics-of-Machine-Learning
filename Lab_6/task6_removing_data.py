import mlflow
import subprocess

# Set the MLflow server URL
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

# Run mlflow gc
subprocess.run(["mlflow", "gc"], check=True)
