"""
Given a custom image, predict the classification of that image
"""

import argparse

import torch
from PIL import Image

from classifai import model_builder
from classifai.transform import get_transforms
from classifai.utils import plot_one_image


def load_pytorch_model(
    model_path: str,
    device: torch.device,
    model_name: str = "effnetb0",
) -> torch.nn.Module:
    model = model_builder.get_model(model_name=model_name, device=device)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    print(f"Loaded model from {model_path}")
    return model


def raw_image_to_ml_input(image_path: str) -> torch.FloatTensor:
    data_transform = get_transforms(mode="normal")
    image = Image.open(image_path)
    image = data_transform(image)
    return image


def one_forward_pass(
    model: torch.nn.Module, image_path: str, device: torch.device
) -> torch.FloatTensor:
    image_tensor = raw_image_to_ml_input(image_path)
    image_tensor = image_tensor.to(device).unsqueeze(dim=0)
    model.eval()
    with torch.inference_mode():
        pred_logits = model(image_tensor)
        percentage = torch.softmax(pred_logits, dim=1).squeeze()
        pred = torch.argmax(torch.softmax(pred_logits, dim=1), dim=1)
        percentage = percentage[pred.item()]
    class_names = ["pizza", "steak", "sushi"]

    return class_names[pred], percentage


def main():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {device}")
    parser = argparse.ArgumentParser(
        description="Predict the label of a custom image with a PyTorch Model."
    )

    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path to custom image",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to Pytorch Model",
    )

    args = parser.parse_args()

    model = load_pytorch_model(args.model_path, device)

    pred_class, percentage = one_forward_pass(
        model=model, image_path=args.image_path, device=device
    )
    plot_one_image(
        raw_image_to_ml_input(args.image_path),
        image_class=pred_class,
        percentage=percentage,
    )


if __name__ == "__main__":
    main()
