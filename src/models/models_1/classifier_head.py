import torch.nn as nn
import torch.nn.functional as F

'''
Vai trò:
    - Giảm overfiting: dùng Dropout
    - Nén biểu diễn trung gian: dung Linear giảm/biến đổi số chiều đặc trưng
    - Trả logits => tính xác xuất dùng hàm softmax
'''
class ClassifierHead(nn.Module):
    def __init__(self, n_classes: int):
        # Gọi hàm khởi tạo của Module: quản lý các layer bên trong (bắt buộc)
        super().__init__()
        # Tắt ngẫu nhiên 50% số neuron trong lúc train
        self.dropout = nn.Dropout(0.3)
        # Giảm số chiều từ 512 xuống còn 128
        self.fc2 = nn.Linear(512, 128)
        # Giảm số chiều đầu ra từ 128 xuống còn n_classes
        self.fc_out = nn.Linear(128, n_classes)
    # Lan truyền xuôi
    def forward(self, x):
        # Hàm kích hoạt
        x = F.relu(self.fc2(x))
        # Ngẫu nhiên tắt một số neuron trên x
        x = self.dropout(x)

        # Dữ liệu đầu ra
        x = self.fc_out(x)
        return x

