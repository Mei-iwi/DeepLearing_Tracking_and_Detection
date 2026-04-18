import torch.nn as nn
from .classifier_head import ClassifierHead

'''
    Vai trò: Ghép các thành phần mô hình lại với nhau
        - backbone: Trích xuất đặc trưng
        - headpred: Xử lý trung gian trước classifier
        - classifier: Phân loại cuối cùng
'''

class CurrentCNN(nn.Module):
    def __init__(self, backbone, headprep, n_classes: int):
        # Gọi hàm khởi tạo của nn.Modulte (bắt buộc)
        super().__init__()
        # Gán backbone -> là thuộc tính nổi bật của object
        self.backbone = backbone
        # Lưu khối xử lý trung gian trước classifier -> flattern, linear, relu
        self.headprep = headprep
        # Sinh ra đầu ra có số chiều bằng số lớp
        self.classifier = ClassifierHead(n_classes)
    # Hàm forward định nghĩa hàm dữ liệu qua model -> x dữ liệu đầu vào (batch ảnh)
    def forward(self, x):
        # Cho ảnh qua backbone -> kết quả là tensor đặc trưng [N, 3, 224, 224] -> [N, 256, 14, 14]
        x = self.backbone(x)
        # Tiếp tục cho tensor đặc trưng qua khối chuẩn bị cho classifier (flattern, linear, relu, ...): [N, 256, 14, 14] -> [N, 512]
        x = self.headprep(x)
        # Đưa vector đặc trưng vào head phân loại -> đầu ra là logit [N, 512] -> [N, n_classes]
        x = self.classifier(x)
        return x
    def forward_from_stage4(self, x):
        pass
    
'''
    Ảnh đầu vào -> backbone -> headprep -> classifier -> logits đầu ra
    Ví dụ:
    Qua backbone
        - Input vào backbone:
            [1, 3, 224, 224]
        - Sau Conv1 + Pool1:
            [1, 32, 112, 112]
        - Sau Conv2 + Pool2:
            [1, 64, 56, 56]
        - Sau Conv3 + Pool3:
            [1, 128, 28, 28]
        - Sau Conv4 + Pool4:
            [1, 256, 14, 14]
    Qua headpred
       -  Đầu vào headprep:
            [1, 256, 14, 14]
        - Sau Flatten:
            [1, 50176]
        -  Sau Linear(50176, 512) + ReLU:
            [1, 512]
    Qua classifier
        - Đầu vào classifier:
            [1, 512]
        - Sau Dropout:
            [1, 512]
        - Sau Linear(512,128) + ReLU:
            [1, 128]
        - Sau Linear(128,3):
            [1, 3]
    Giả sử
     logits = [
                [2.1, 0.3, 1.0],
                [0.5, 3.2, 0.7],
                [1.4, 1.1, 2.8],
                [4.0, 0.2, 0.1]
             ]
    preds = logit.argmax(dim=1) = [0, 1, 2, 0]
'''