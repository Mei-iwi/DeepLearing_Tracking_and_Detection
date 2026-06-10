import torch.nn as nn


class ClassifierHead(nn.Module):
    def __init__(self, in_features: int, num_classes: int):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.fc(x)