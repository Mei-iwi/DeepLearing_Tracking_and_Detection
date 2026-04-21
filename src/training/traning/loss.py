import torch.nn as nn
def build_criterion():
    return nn.CrossEntropyLoss()