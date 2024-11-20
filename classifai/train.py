import argparse
import os
import torch
from . import data_setup, engine, model_builder, utils
import torch.multiprocessing as mp
from classifai.transform import get_transforms


def train_model(
    train_dir,
    test_dir,
    model_save_path,
    epochs=5,
    batch_size=32,
    hidden_units=10,
    lr=0.001,
    device=None,
):
    if device is None:
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

    # data_transform = transforms.Compose(
    #     [transforms.Resize((64, 64)), transforms.ToTensor()]
    # )
    data_transform = get_transforms(mode="normal")

    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        transform=data_transform,
        batch_size=batch_size,
    )

    model = model_builder.TinyVGG(
        input_shape=3, hidden_units=hidden_units, output_shape=len(class_names)
    ).to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    engine.train(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=epochs,
        device=device,
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
        "--train_dir", type=str, required=True, help="Path to training data directory."
    )
    parser.add_argument(
        "--test_dir", type=str, required=True, help="Path to testing data directory."
    )
    parser.add_argument(
        "--model_save_path",
        type=str,
        required=True,
        help="Path to save the trained model.",
    )
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of epochs for training."
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training."
    )
    parser.add_argument(
        "--hidden_units",
        type=int,
        default=10,
        help="Number of hidden units in the model.",
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="Learning rate for optimizer."
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
