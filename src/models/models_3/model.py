from .resnet_model import ResNet18Classifier


def build_model_3(n_classes: int, device: str):
    model = ResNet18Classifier(
        num_classes=n_classes,
        freeze_backbone=False,
        fine_tune=False
    )

    return model.to(device)