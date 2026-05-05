from src.models.models_2.cnn_model import CNNModel

'''
    Hàm build Model 2 cho factory gọi
'''


def build_model_2(n_classes: int, device: str):
    model = CNNModel(n_classes=n_classes)
    return model.to(device)