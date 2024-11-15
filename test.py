"""
Script for testing code
"""

from get_data import download_data
from torchvision import transforms
from classifai import data_setup
from pathlib import Path

path_to_data = download_data(
    url="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
    destination="pizza_steak_sushi.zip",
)

train_dir = Path(path_to_data / "train")
test_dir = Path(path_to_data / "test")

transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])

train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir, test_dir=test_dir, transform=transform, batch_size=32
)

print(f"Class Names: {class_names}")
