from classifai.train import train_model
import argparse
import torch


def main():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")
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
        device=device,
    )


if __name__ == "__main__":
    main()
