import torch.nn as nn

class ClassifierHead(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 108)
        self.fc_out = nn.Linear(128, n_classes)
    def forward_classifir_head(self, x):
        pass