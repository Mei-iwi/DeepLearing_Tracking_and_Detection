import torch.nn as nn

'''
    Vai trò:
        - Trích xuất đặc trưng từ ảnh đầu vào
        - Gồm 3 Conv Block + MaxPool
'''


class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

    def forward(self, x):
        return self.features(x)