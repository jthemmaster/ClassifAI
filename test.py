"""
Script for testing code
"""

from get_data import download_data

download_data(
    url="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
    destination="pizza_steak_sushi.zip",
)
