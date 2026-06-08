import mlflow

def get_latest_run_id():
    mlflow.set_tracking_uri("mlruns")
    client = mlflow.MlflowClient()
    try:
        experiment = client.get_experiment_by_name("Eksperimen_Model_Telco")
        if experiment:
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id], 
                order_by=["attributes.start_time DESC"], 
                max_results=1
            )
            if runs:
                print(runs[0].info.run_id)
                return
        print("NONE")
    except Exception:
        print("NONE")

if __name__ == "__main__":
    get_latest_run_id()
    