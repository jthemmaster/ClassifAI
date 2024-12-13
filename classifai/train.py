import argparse
import os

import torch
import torch.multiprocessing as mp

from classifai.transform import get_transforms

from . import data_setup, engine, model_builder, utils


def train_model(
    train_dir,
    test_dir,
    model_save_path,
    epochs=5,
    batch_size=32,
    lr=0.001,
    device=None,
    model_name: str = "TinyVGG",
    experiment_name: str = "",
    extra: str = "",
):
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    data_transform = get_transforms(mode="normal")

    train_dataloader, test_dataloader, class_names = (
        data_setup.create_dataloaders(
            train_dir=train_dir,
            test_dir=test_dir,
            transform=data_transform,
            batch_size=batch_size,
        )
    )

    model = model_builder.get_model(model_name=model_name, device=device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # writer = utils.create_writer(
    #     experiment_name=experiment_name, model_name=model_name, extra=extra
    # )
    engine.train_with_mlflow(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=epochs,
        device=device,
        # writer=writer,
    )

    utils.save_model(
        model=model,
        target_dir=os.path.dirname(model_save_path),
        model_name=os.path.basename(model_save_path),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train a PyTorch image classification model."
    )
    parser.add_argument(
        "--train_dir",
        type=str,
        required=True,
        help="Path to training data directory.",
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        required=True,
        help="Path to testing data directory.",
    )
    parser.add_argument(
        "--model_save_path",
        type=str,
        required=True,
        help="Path to save the trained model.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of epochs for training.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--hidden_units",
        type=int,
        default=10,
        help="Number of hidden units in the model.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate for optimizer.",
    )
    args = parser.parse_args()

    train_model(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        model_save_path=args.model_save_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_units=args.hidden_units,
        lr=args.lr,
    )


if __name__ == "__main__":
    mp.set_start_method("fork", force=True)
    torch.set_num_threads(1)
    main()
