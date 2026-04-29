from .backbone import Backbone
from .headprep import HeadPrep
from .cnn_model import CurrentCNN


def build_model_1(n_classes: int, device: str):
    model = CurrentCNN(
        backbone=Backbone(),
        headprep=HeadPrep(),
        n_classes=n_classes
    ).to(device)
    return model