"""
Script for testing code
"""

from get_data import download_data
from torchvision import transforms
from classifai import data_setup, model_builder, utils
from pathlib import Path
import torch.multiprocessing as mp


if __name__ == "__main__":
    mp.set_start_method("fork", force=True)

    path_to_data = download_data(
        url="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
        destination="pizza_steak_sushi.zip",
    )

    train_dir = Path(path_to_data / "train")
    test_dir = Path(path_to_data / "test")

    simple_transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ]
    )

    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        transform=simple_transform,
        batch_size=32,
        # num_workers=6,
    )

    model = model_builder.TinyVGG(
        input_shape=3, hidden_units=10, output_shape=len(class_names)
    )
