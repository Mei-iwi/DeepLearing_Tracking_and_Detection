import torch.optim as optim

'''
    Vai trò:
        - Tạo optimizer Adam cho toàn bộ tham số của model
'''

def build_optimizer(model, lr=1e-3):
    return optim.Adam(model.parameters(), lr=lr)