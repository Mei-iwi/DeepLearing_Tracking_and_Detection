import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class ResNet18Backbone(nn.Module):
    def __init__(self, freeze_backbone=True, fine_tune=True):
        super().__init__()

        weights = ResNet18_Weights.DEFAULT
        base = resnet18(weights=weights)

        base.fc = nn.Identity()
        self.base = base

        if freeze_backbone:
            for param in self.base.parameters():
                param.requires_grad = False

        if fine_tune:
            for param in self.base.layer4.parameters():
                param.requires_grad = True

    def forward(self, x):
        return self.base(x)