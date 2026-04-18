import torch 

'''
    Vai trò: đánh giá mô hình trên tập loader
'''

def evaluate_model(model, loader, device):
    # Chuyển sang chế độ đánh giá
    model.eval()

    # Lưu nhãn dự đoán
    all_preds = []

    # Lưu nhãn thật
    all_labels = []

    # Không tính đạo hàm khi đánh giá
    with torch.no_grad():
        # Duyệt qua các ảnh và labels trong loader
        for images, labels in loader:
            # Đưa ảnh và nhãn sang cùng thiết bị để tính toán
            images = images.to(device)
            labels = labels.to(device)

            # Truyền batch ảnh vào mạng
            logits = model(images)

            #Lấy lớp có điểm cao nhất
            preds = logits.argmax(dim=1)

            '''
            preds.cpu() Chuyển kết quả từ GPU -> CPU
            .tolist() Đổi tensor về list python
            extend() Nối thêm các phần tử vào list tổng
            '''
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    return all_labels, all_preds
            
def evaluate(model, loader, criterion, device):
    pass

def evaluate_testset(model, test_loader, criterion, device):
    pass

'''
    logit = model(images):
        images: là một batch ảnh -> ví dụ: [N, 3, 224, 224]
        logits =
                [
                    [1.2, 0.3, 2.5], -> batch 0
                    [0.1, 3.7, 1.4] -> batch 1
                ]
    Ý nghĩa 
        - Mỗi hàng là kết quả 1 ảnh
        - Mỗi cột là điếm số của một lớp
    pred = logit.argmax(dim=1)  
        argmax(dim=1): Tìm vị trí có giá trị lớn nhất trong từng hàng
                [
                    [1.2, 0.3, 2.5],   # lớp 2 lớn nhất
                    [0.1, 3.7, 1.4]    # lớp 1 lớn nhất
                ]
        ==> pred = [2, 1]
    dim=1 theo cột: lớp
    dim=0 theo hàng: batch
'''

if __name__ == "__main__":
    logit = torch.tensor(
        [
            [2.1, 0.5, 1.7], # 2.1 -> 0
            [0.2, 3.4, 1.1], # 3.4 -> 1
            [0.9, 0.4, 2.8], # 2.8 -> 2
        #   2.1   3.4  2.8
        #   0       1   2
        ]
    )
    preds = logit.argmax(dim=1)
    print(preds)
    preds = logit.argmax(dim=0)
    print(preds)