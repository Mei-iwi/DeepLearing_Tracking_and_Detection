import torch.nn as nn

'''
    Vai trò:
        - Chuyển feature map thành vector
        - Chuẩn bị đầu vào cho classifier
'''


class HeadPrep(nn.Module):
    def __init__(self):
        super(HeadPrep, self).__init__()
        self.flatten = nn.Flatten()

    def forward(self, x):
        return self.flatten(x)