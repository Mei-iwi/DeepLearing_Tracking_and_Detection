import torch.nn as nn

'''
    Vai trò
        - Nhận tensor đặc trưng từ backbone
        - Flattern tensor [N, 256, 14, 14] thành [N, 50176]
        - Dùng Linear + Relu để đưa về vector [N, 512]
'''

class HeadPrep(nn.Module):
    def __init__(self):
<<<<<<< HEAD
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256*14*14, 512)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
=======
        super(HeadPrep, self).__init__()

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(256 * 14 * 14, 512) 
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)


    def forward(self, x):
        x = self.flatten(x)
        x = self.fc(x)
        x = self.relu(x)
        x = self.dropout(x)
>>>>>>> origin/main
        return x