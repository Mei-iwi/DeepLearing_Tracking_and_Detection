import torch.nn as nn

class HeadPrep(nn.Module):
    def __init__(self):
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
        return x