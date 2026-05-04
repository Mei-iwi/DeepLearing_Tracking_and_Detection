from src.models.models_1.model import build_model_1
from src.models.models_2.model import build_model_2
# from src.models.models_3.model import build_model_3
# from src.models.models_4.model import build_model_4


def build_model(model_name: str, n_classes: int, device: str):
    if model_name == "model_1":
        return build_model_1(n_classes=n_classes, device=device)
    elif model_name == "model_2":
        return build_model_2(n_classes=n_classes, device=device)
        pass
    elif model_name == "model_3":
        # return build_model_3(n_classes=n_classes, device=device)
        pass
    elif model_name == "model_4":
        # return build_model_4(n_classes=n_classes, device=device)
        pass
    else:
        raise ValueError(f"Không hỗ trợ model_name={model_name}")