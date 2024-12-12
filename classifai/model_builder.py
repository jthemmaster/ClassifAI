"""
Contains PyTorch model
"""

import torch
from torch import nn
import torchvision


class TinyVGG(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=input_shape,
                out_channels=hidden_units,
                stride=1,
                padding=1,
                kernel_size=3,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                stride=1,
                padding=1,
                kernel_size=3,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                stride=1,
                padding=1,
                kernel_size=3,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                stride=1,
                padding=1,
                kernel_size=3,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units * 256, out_features=output_shape),
        )

    def forward(self, x):
        return self.classifier(self.block2(self.block1(x)))


def get_model(
    model_name: str = "TinyVGG",
    output_shape: int = 3,
    hidden_units: int = 128,
    device: torch.device = "cpu",
) -> torch.nn.Module:
    if model_name == "TinyVGG":
        model = TinyVGG(
            input_shape=3, hidden_units=hidden_units, output_shape=output_shape
        ).to(device)

    elif model_name == "effnetb0":
        weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
        model = torchvision.models.efficientnet_b0(weights=weights).to(device)
        # Freeze parameters
        for params in model.features.parameters():
            params.requires_grad = False
        model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.2, inplace=True),
            torch.nn.Linear(in_features=1280, out_features=3),
        ).to(device)

    elif model_name == "effnetv2_s":
        weights = torchvision.models.EfficientNet_V2_S_Weights.DEFAULT
        model = torchvision.models.efficientnet_v2_s(weights=weights).to(device)
        # Freeze parameters
        for params in model.features.parameters():
            params.requires_grad = False
        model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.2, inplace=True),
            torch.nn.Linear(in_features=1280, out_features=3),
        ).to(device)
    else:
        print(f"Model: {model_name} is not implemented")
        exit()
    return model
