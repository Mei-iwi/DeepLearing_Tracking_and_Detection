import torch.optim as optim
def build_optimizer(model, lr=1e-3):
    return optim.Adam(model.parameters(), lr=lr)