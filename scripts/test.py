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
        url="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi_20_percent.zip",
        destination="pizza_steak_sushi_20.zip",
    )

    train_dir = Path(path_to_data / "train")
    test_dir = Path(path_to_data / "test")
