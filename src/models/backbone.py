import torch.nn as nn

'''
    Vai trò:
        - Trích xuất đặc trưng từ ảnh đầu vào
        - Gồm 4 stage Conv + ReLU + MaxPool
        - Đầu ra cuối cùng là tensor [N, 256, 14, 14] nếu input là [N, 2, 224, 224]
'''

class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Stage 1 [N, 3, 224, 224] -> [N, 32, 112, 112]
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Statge 2: [N, 32, 112, 112] -> [N, 64, 56, 56]
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Stage 3: [N, 64, 56, 56] -> [N, 128, 28, 28]
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Stage 3: [N, 128, 28, 28] -> [N, 256, 14, 14]
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward_state1(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        return x

    def forward_state2(self, x):
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        return x

    def forward_state3(self, x):
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        return x

    def forward_state4(self, x):
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.pool4(x)
        return x

    def forward(self, x):
        x = self.forward_state1(x)
        x = self.forward_state2(x)
        x = self.forward_state3(x)
        x = self.forward_state4(x)
        return x