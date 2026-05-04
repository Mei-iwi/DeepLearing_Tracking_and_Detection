import torch.nn as nn
from src.models.models_2.backbone import Backbone
from src.models.models_2.headprep import HeadPrep
from src.models.models_2.classifier_head import ClassifierHead

'''
    Ghép toàn bộ model
'''


class CNNModel(nn.Module):
    def __init__(self, n_classes):
        super(CNNModel, self).__init__()

        self.backbone = Backbone()
        self.headprep = HeadPrep()
        self.classifier = ClassifierHead(n_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.headprep(x)
        x = self.classifier(x)
        return x