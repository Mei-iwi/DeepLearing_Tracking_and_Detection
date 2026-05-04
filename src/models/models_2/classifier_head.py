import torch.nn as nn

'''
    Vai trò:
        - Phân loại đặc trưng đã trích xuất
'''


class ClassifierHead(nn.Module):
    def __init__(self, n_classes):
        super(ClassifierHead, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(128 * 16 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        return self.classifier(x)