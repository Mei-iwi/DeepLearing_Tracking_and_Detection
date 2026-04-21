import torch.nn as nn

'''
    Vai trò
        - Trả về hàm mất mát cho bài toán phân loại nhiều lớp
        - Đầu vào model là logits [N, n_classes]
        - Nhãn là tensor số nguyên tử [N]
'''

def build_criterion():
    return nn.CrossEntropyLoss()