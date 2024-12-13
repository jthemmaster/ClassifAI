"""
Contains training and test Pytorch model
"""

from typing import Dict, List, Tuple

import mlflow
import mlflow.pytorch
import torch
import torch.utils.tensorboard
from tqdm.auto import tqdm


def train_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Trains a PyTorch model for one epoch.

    Passes a target PyTorch model through a DataLoader for one epoch,
    calculating the loss and accuracy of the model.

    Args:
      model: A PyTorch model to be trained.
      dataloader: A DataLoader instance for the model to be trained on.
      loss_fn: A PyTorch loss function to calculate loss.
      optimizer: A PyTorch optimizer to help minimize the loss function.
      device: A target device to compute on (e.g. "cuda" or "mps" or "cpu").

    Returns:
      A tuple of the average loss and accuracy for the epoch.
      (loss, accuracy)
    """
    model.train()
    train_loss, train_acc = 0, 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    return train_loss, train_acc


def test_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    test_loss, test_acc = 0, 0
    model.eval()

    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            test_loss += loss.item()
            y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            test_acc += (y_pred_class == y).sum().item() / len(y_pred)
        test_loss /= len(dataloader)
        test_acc /= len(dataloader)
    return test_loss, test_acc


def train(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: torch.device,
    writer: torch.utils.tensorboard.SummaryWriter,
) -> Dict[str, List]:
    """Trains and tests a PyTorch model.

    Passes a target PyTorch models through train_step() and test_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.

    Args:
      model: A PyTorch model to be trained and tested.
      train_dataloader: A DataLoader instance for the model to be trained on.
      test_dataloader: A DataLoader instance for the model to be tested on.
      optimizer: A PyTorch optimizer to help minimize the loss function.
      loss_fn: A PyTorch loss function to calculate loss on both datasets.
      epochs: An integer indicating how many epochs to train for.
      device: A target device to compute on (e.g. "cuda" or "mps" or "cpu").

    Returns:
      A dictionary of training and testing loss as well as training and
      testing accuracy metrics. Each metric has a value in a list for
      each epoch.
      In the form: {train_loss: [...],
                    train_acc: [...],
                    test_loss: [...],
                    test_acc: [...]}
      For example if training for epochs=2:
                   {train_loss: [2.0616, 1.0537],
                    train_acc: [0.3945, 0.3945],
                    test_loss: [1.2641, 1.5706],
                    test_acc: [0.3400, 0.2973]}
    """

    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
    }

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
        )

        test_loss, test_acc = test_step(
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            device=device,
        )
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
        if writer:
            writer.add_scalars(
                main_tag="Loss",
                tag_scalar_dict={
                    "train_loss": train_loss,
                    "test_loss": test_loss,
                },
                global_step=epoch,
            )
            writer.add_scalars(
                main_tag="Accuracy",
                tag_scalar_dict={
                    "train_acc": train_acc,
                    "test_acc": test_acc,
                },
                global_step=epoch,
            )
            writer.close()
        else:
            pass
    return results


def train_with_mlflow(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: torch.device,
) -> Dict[str, List]:
    """Trains and tests a PyTorch model with MLflow experiment tracking."""

    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
    }

    # End any lingering active run
    if mlflow.active_run():
        print(
            f"Ending active run before starting a new one: "
            f"{mlflow.active_run().info.run_id}"
        )
        mlflow.end_run()

    try:
        # Start an MLflow run
        with mlflow.start_run(nested=True):
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("optimizer", type(optimizer).__name__)
            mlflow.log_param("loss_fn", type(loss_fn).__name__)

            for epoch in tqdm(range(epochs)):
                train_loss, train_acc = train_step(
                    model=model,
                    dataloader=train_dataloader,
                    loss_fn=loss_fn,
                    optimizer=optimizer,
                    device=device,
                )

                test_loss, test_acc = test_step(
                    model=model,
                    dataloader=test_dataloader,
                    loss_fn=loss_fn,
                    device=device,
                )

                print(
                    f"Epoch: {epoch+1} | "
                    f"train_loss: {train_loss:.4f} | "
                    f"train_acc: {train_acc:.4f} | "
                    f"test_loss: {test_loss:.4f} | "
                    f"test_acc: {test_acc:.4f}"
                )

                results["train_loss"].append(train_loss)
                results["train_acc"].append(train_acc)
                results["test_loss"].append(test_loss)
                results["test_acc"].append(test_acc)

                # Log metrics to MLflow
                mlflow.log_metric("train_loss", train_loss, step=epoch)
                mlflow.log_metric("train_acc", train_acc, step=epoch)
                mlflow.log_metric("test_loss", test_loss, step=epoch)
                mlflow.log_metric("test_acc", test_acc, step=epoch)

            # Log the trained model to MLflow
            mlflow.pytorch.log_model(model, "model")

    finally:
        # Ensure the run is ended
        if mlflow.active_run():
            print(
                f"Ending run at the end of train_with_mlflow: "
                f"{mlflow.active_run().info.run_id}"
            )
            mlflow.end_run()

    return results
