import os
from torchvision import datasets
from torch.utils.data import DataLoader
from .transforms import get_train_transforms, get_val_test_transforms

# Đọc dữ liệu từ folder
def get_datasets(data_dir):
    train_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, "images", "train"),
        transform=get_train_transforms()
    )

    val_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, "images", "val"),
        transform=get_val_test_transforms()
    )

    test_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, "images", "test"),
        transform=get_val_test_transforms()
    )

    return train_dataset, val_dataset, test_dataset

# Chia dữ liệu thành batch để train
def get_dataloaders(data_dir, batch_size, num_workers=2, pin_memory=True):
    train_dataset, val_dataset, test_dataset = get_datasets(data_dir)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,   # train thì shuffle
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # val/test không shuffle
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader, test_loader

# Lấy tên class
def get_class_names(train_dataset):
    return train_dataset.classes