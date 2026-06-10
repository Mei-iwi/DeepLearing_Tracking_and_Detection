import torch.nn as nn
from .backbone import ResNet18Backbone
from .classifier_head import ClassifierHead


class ResNet18Classifier(nn.Module):
    def __init__(self, num_classes: int, freeze_backbone=False, fine_tune=False):
        super().__init__()

        self.backbone = ResNet18Backbone(
            freeze_backbone=freeze_backbone,
            fine_tune=fine_tune
        )

        self.classifier = ClassifierHead(
            in_features=512,
            num_classes=num_classes
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x