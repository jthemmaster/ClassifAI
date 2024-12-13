from torchvision import transforms


def get_transforms(mode="normal"):
    if mode == "normal":
        return transforms.Compose(
            [transforms.Resize((64, 64)), transforms.ToTensor()]
        )
