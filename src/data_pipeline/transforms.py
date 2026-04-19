from torchvision import transforms

def get_train_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),          # đưa về 224x224
        transforms.RandomHorizontalFlip(),      # lật ảnh (tăng data)
        transforms.ToTensor(),                  # chuyển sang tensor
    ])

def get_val_test_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),          # chỉ resize
        transforms.ToTensor(),
    ])