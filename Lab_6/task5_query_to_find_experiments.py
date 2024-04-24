import mlflow
import pandas as pd

# Set the option to display all columns
pd.set_option('display.max_columns', None)

mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

# Get the experiment
experiment = mlflow.get_experiment_by_name("Titanic Survivors")

# Check if the experiment exists
if experiment:
    # Define the filter string
    filter_string = "metrics.accuracy > 0.5"

    # Use the filter string in the search_runs function
    runs_with_filter = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],  # use the experiment id
        filter_string=filter_string,
    )

    print(runs_with_filter)
else:
    print("Experiment not found.")