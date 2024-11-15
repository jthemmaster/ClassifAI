"""
Script to get torchvision data
"""

import os
import zipfile
from pathlib import Path
import requests

def download_data(url: str, destination: str, remove_source: bool = True) -> Path:
    """
    Downloads data from a URL and returns the path to the downloaded file
    
    Args:
        url: str: URL to download the data from
        destination: str: Destination to save the data
        remove_source: bool: Whether to remove the source file after downloading

    Returns:
        pathlib.Path: Path to the downloaded file

    Example usage:
    download_data("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip", destination="pizza_steak_sushi.zip")
    """

    data_path = Path("data")
    data_path.mkdir(exist_ok=True, parents=True)
    file_path = data_path / destination
    extracted_dir = data_path / destination.replace(".zip", "")

    if extracted_dir.exists():
        print(f"Folder {extracted_dir} already exists, skipping download...")
        return extracted_dir
    
    print(f"Downloading: {url} ...")
    response = requests.get(url)
    
    with open(file_path, "wb") as fd:
        fd.write(response.content)
    
    print(f"Unzipping to data to {extracted_dir}")
    with zipfile.ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall(extracted_dir)

    if remove_source:
        file_path.unlink()
    
    return extracted_dir

    
    

