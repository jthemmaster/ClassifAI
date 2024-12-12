from classifai.train import train_model
import torch
import mlflow


def main():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    data = ["pizza_steak_sushi", "pizza_steak_sushi_20"]
    epochs = [10, 20]
    models = ["TinyVGG", "effnetb0", "effnetv2_s"]

    # Initialize MLflow experiment
    mlflow.set_experiment("Food Classification Experiments")

    experiment_number = 0

    for d in data:
        for e in epochs:
            for m in models:
                experiment_number += 1

                train_dir = f"data/{d}/train"
                test_dir = f"data/{d}/test"
                model_save_path = f"models/{m}_{d}_{e}_epochs.pth"

                # End any active run before starting a new one
                if mlflow.active_run():
                    mlflow.end_run()

                try:
                    with mlflow.start_run(
                        run_name=f"Experiment_{experiment_number}", nested=True
                    ):
                        # Log parameters and run training
                        mlflow.log_param("data", d)
                        mlflow.log_param("epochs", e)
                        mlflow.log_param("model", m)
                        mlflow.log_param("train_dir", train_dir)
                        mlflow.log_param("test_dir", test_dir)
                        mlflow.log_param("device", device)

                        train_model(
                            train_dir=train_dir,
                            test_dir=test_dir,
                            model_save_path=model_save_path,
                            epochs=e,
                            device=device,
                            model_name=m,
                            experiment_name=d,
                            extra=f"{e}_epochs",
                        )
                        mlflow.log_artifact(model_save_path)
                finally:
                    mlflow.end_run()

    print("[INFO] Experiment tracking completed!")


if __name__ == "__main__":
    main()
