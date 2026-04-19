from torchvision import transforms

'''
    Vai trò:
        - Khai báo các phép biến đổi ản trước khi đưa vào mô hình
        - Tách riêng transform cho train và val/test để dễ quản lý
        - Train có augmentation nhẹ để tăng đa dạng dữ liệu.
        - Val/Test chỉ chuẩn hóa đầu vào, không augmentation ngẫu nhiên
'''

def get_train_transforms():
    '''
        Trả về pipeline transform danh cho tập train
        - Quy trình:
            + Resize ảnh về 224x224 -> đồng nhất kích thước đầu vào
            + Lật ngang ngẫu nhiên để tăng đa dạng dữ liệu
            + Chuyển ảnh PIL sang tensor Pytorch
    '''
    return transforms.Compose([
        transforms.Resize((224, 224)),          # đưa về 224x224
        transforms.RandomHorizontalFlip(),      # lật ảnh (tăng data)
        transforms.ToTensor(),                  # chuyển sang tensor
    ])

def get_val_test_transforms():
    '''
        Trả về pipe line transform cho validation và test
        - Khác với train:
            + Không dùng augumentation ngẫu nhiên
            + Chỉ resize và đổi sang tensor để đánh giá ổn đinh
    '''
    return transforms.Compose([
        transforms.Resize((224, 224)),          # chỉ resize
        transforms.ToTensor(),
    ])


if __name__ == "__main__":
    train_tf = get_train_transforms()
    val_tf = get_val_test_transforms()
    
    print("Train transformation: ")
    print(train_tf)
    print("-"*50)
    print("Val/Test transform")
    print(val_tf)