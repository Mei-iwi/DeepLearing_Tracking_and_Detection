import torch
from PIL import Image
from torchvision import transforms

'''
    Vai trò: dự đoán cho một ảnh
'''

def predict_image(model, image_path, device, class_names):
    '''
        Tạo pipline tiền xử lý ảnh
            - transforms.Compost([]): gép nhiều bước thành một pipline
            - Resize: Đưa ảnh về kích thước 224x224
            - ToTenseor: Chuyển ảnh từ PIL sang tensor pytorch
            - Normalize: Chuẩn hóa giá trị ảnh theo mean và std của ImageNet
    '''
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Mở ảnh theo đường dẫn và chuyển ảnh về RGB (3 kênh màu Red Green Blue)
    image = Image.open(image_path).convert("RGB")

    '''
        Tiền xử lý ảnh theo chiều batch 
            - tf(image): Áp dụng toàn bộ pipine lên ảnh [3, 224, 224]
            - unsqueeze: Thêm 1 chiều đầu ảnh để tạo batch size = 1 từ [3, 224, 223] -> [1, 3, 224, 224]
            - to(device): Chuyển sang cpu hoặc gpu
    '''
    x = tf(image).unsqueeze(0).to(device)  #type:ignore

    # Chuyển sang đánh giá mô hình -> đưa về dạng suy luận
    model.eval()
    
    # Tắt gradient
    with torch.no_grad():

        # Cho ảnh qua mô hình
        logits = model(x)

        # Chuyển logits thành xác suất
        probs = torch.softmax(logits, dim=1)

        # Lấy chỉ số dự đoán lớn nhất theo lớp
        pred_idx = probs.argmax(dim=1).item()

    return class_names[pred_idx], probs.squeeze().cpu().tolist()

def predict_logits(model, image_tensor, device):
    pass

def predict_proba(logits):
    pass

'''
    0.485, 0.456, 0.406 là trung bình của 3 kênh R, G, B
    0.229, 0.224, 0.225 là độ lệch chuẩn của 3 kênh R, G, B
'''

if __name__ == "__main__":
    print("test")