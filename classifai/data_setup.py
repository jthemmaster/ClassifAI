"""
Contains functionality to create Dataloaders for the dataset
"""

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

NUM_WORKERS = 0  # os.cpu_count()


def create_dataloaders(
    train_dir: str,
    test_dir: str,
    transform: transforms.Compose,
    batch_size: int,
    num_workers: int = NUM_WORKERS,
):
    """
    Creates training and testing Dataloaders

    Takes training and testing directory paths and turns them into
    PyTorch Datasets and Dataloaders

    Args:
        train_dir: Path to training directory
        test_dir: Path to testing directory
        transform: torchvision transforms performed on training and
        testing data
        batch size: Batch size of Dataloader
        num_workers: Number of worker per DataLoader

    Returns:
        A tuple of (train_dataloader, test_dataloader, class_names)
    """

    # Using ImageFolder for transform:
    # https://pytorch.org/vision/main/generated/torchvision.datasets.ImageFolder.html

    train_data = datasets.ImageFolder(train_dir, transform=transform)
    test_data = datasets.ImageFolder(test_dir, transform=transform)
    # Turn them into dataloaders

    train_dataloader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        # pin_memory=True,
    )

    test_dataloader = DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        # pin_memory=True,
    )

    class_names = train_data.classes
    return train_dataloader, test_dataloader, class_names
