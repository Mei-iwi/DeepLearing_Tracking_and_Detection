import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


class Model4MobileNetV2(nn.Module):
    def __init__(self, num_classes, freeze_features=True, fine_tune=True):
        super().__init__()

        # Load MobileNetV2 pretrained trên ImageNet
        weights = MobileNet_V2_Weights.DEFAULT
        base_model = mobilenet_v2(weights=weights)

        # Feature extractor (backbone)
        self.features = base_model.features

        # FREEZE / FINE-TUNE
        if freeze_features:
            # Freeze toàn bộ backbone
            for param in self.features.parameters():
                param.requires_grad = False

            # Nếu bật fine-tune → mở block cuối
            if fine_tune:
                for param in self.features[-2:].parameters():
                    param.requires_grad = True

        # CLASSIFIER MỚI
        in_features = base_model.classifier[1].in_features

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, num_classes) # type: ignore
        )

    def forward(self, x):
        # Input: [B, 3, 224, 224]

        # Backbone
        # → [B, 1280, 7, 7]
        x = self.features(x)

        # Global Average Pooling
        # → [B, 1280, 1, 1]
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))

        # Flatten
        # → [B, 1280]
        x = torch.flatten(x, 1)

        # Classifier
        # → [B, num_classes]
        x = self.classifier(x)

        return x


# FACTORY FUNCTION (QUAN TRỌNG)
def build_model_4(n_classes, device):
    model = Model4MobileNetV2(
        num_classes=n_classes,
        freeze_features=True,
        fine_tune=True   # bật fine-tune
    )
    return model.to(device)
if __name__ == "__main__":
    import torch

    # Giả lập số class
    num_classes = 10

    # Tạo model
    model = Model4MobileNetV2(num_classes=num_classes)

    # Chuyển sang eval
    model.eval()

    # Tạo input giả
    x = torch.rand(1, 3, 224, 224)

    # Forward thử
    with torch.no_grad():
        y = model(x)

    print("Input shape :", x.shape)
    print("Output shape:", y.shape)