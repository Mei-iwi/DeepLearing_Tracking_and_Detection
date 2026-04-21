import os
from pathlib import Path
import yaml
from torch.utils.data import DataLoader

from .transforms import get_train_transforms, get_val_test_transforms

'''
    Vai trò:
        - Đọc file dataset.yaml để biết đường dẫn train/val/test và tên class
        - Tạo Dataset cho train, val, test, bằng YoloSingleDataset
        - Bọc Dataset thành DataLoader để phục vụ huấn luyện
        - Giữ nguyên pipeline classifier hiện tại: mỗi sample trả về (image, label)
'''


'''
    Đọc file dataset.yaml trong thư mục dữ liệu
        Parameters: data_dir: str | Path -> Thư mục gốc của package hoặc dataset
        Returns: dict -> Nội dung đã parse từ YAML
'''
def _load_yaml(data_dir):
    # Ghép đường dẫn tới dataset.yaml
    yaml_path = Path(data_dir) / "dataset.yaml"
    # Đường dẫn không tồn tại
    if not yaml_path.exists():
        raise FileNotFoundError(f'Không tìm thấy {yaml_path}')
    # Mở file và parse YAML thành dict
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

'''
    Chuyển trường 'names' trong dataset.yaml về list trên class
    Hỗ trợ 2 kiểu thường gặp
        - dict: {0: persion, 1: car, ...}
        - list: ["Person", "Car", ...]
    Trả về danh sách tên class theo đúng thứ tự id
'''
def _parse_names(name_obj):
    # Nếu names là dict, sắp xếp theo key số
    if isinstance(name_obj, dict):
        
        return [name_obj[k] for k in sorted(name_obj, key=lambda x: int(x))]
    # Nếu names đã là list, dừng
    if isinstance(name_obj, list):
        return name_obj
    # Các trường hợp còn lại không hợp lệ
    raise ValueError("Trường 'names' trong dataset.yaml không hợp lệ" )

'''
    Xác định thư mục ảnh tương ứng với split
    Cơ chế
        - Nếu dataset.yaml có key 'train' / 'val' / 'test' thì dùng đường dẫn đó
        - Nếu không có thì fallback về:
            data_dir / images / split
    Tham số: 
        data_dir: str | Path
        split: str : 'train', 'val' hoặc 'test'
    Trả về đường dẫn thư mục ảnh của split đó
'''
def _resolve_images_dir(data_dir, split):

    # Đọc cấu hình dataset
    cfg = _load_yaml(data_dir)
    
    # Nếu split không có trong YAML, dùng cấu trúc mặc định
    if split not in cfg:
        return Path(data_dir) / "images" / split
    
    # Lấy đường dẫn split từ YAML
    p = Path(cfg[split])
    # Nếu là đường dẫn tương đối thì nối với data_dit
    if not p.is_absolute():
        p = Path(data_dir) / p
    return p


'''
    Suy ra thư mục nhãn từ thư mục ảnh
         Ví dụ:
            - image_dir = .../images/train
            - label_dir = .../labels/train
'''
def resolve_label_dir_from_image_dir(image_dir: Path):
    return image_dir.parent.parent / 'labels' / image_dir.name


''' 
    Tạo 3 dataset: train, val, test
    Tham số: 
        data_dir: str | Path -> Thư mục package hoặc dataset
        strict_single_label: bool 
            - True: chỉ giữ ảnh có đúng 1 class duy nhất
            - False: lấy class xuất hiện nhiều nhất trong ảnh
    Trả về 1 bô (train_dataset, val_dataset, test_dataset)
'''
# Đọc dữ liệu từ folder
def get_datasets(data_dir, strict_single_label=True):
    # Import tại chỗ để tránh vòng lặp import
    from .YoloSingleLabelDataset import YoloSingleLabelDataset

    # Tạo dataset cho train với transfomr train
    train_dataset = YoloSingleLabelDataset(
        data_dir=data_dir,
        split="train",
        transform=get_train_transforms(),
        strict_single_label=strict_single_label   
    )

    # Tạo dataset cho val với transform không augentation
    val_dataset = YoloSingleLabelDataset(
        data_dir=data_dir,
        split="val",
        transform=get_val_test_transforms(),
        strict_single_label=strict_single_label
    )

    # Tạo dataset cho bộ test
    test_dataset = YoloSingleLabelDataset(
        data_dir=data_dir,
        split="test",
        transform=get_val_test_transforms(),
        strict_single_label=strict_single_label
    )

    return train_dataset, val_dataset, test_dataset


'''
    Tạo DataLoader cho train, val, test
    Tham số 
            data_dir : str | Path
            batch_size : int
            num_workers : int
            pin_memory : bool
            strict_single_label : bool
            strict_single_lable : bool | None
                Alias cũ để tránh vỡ code cũ nếu đã gọi nhầm tên tham số.
    Trả về bô (train_loader, val_loader, test_loader)
'''
# Chia dữ liệu thành batch để train
def get_dataloaders(data_dir, batch_size, num_workers=2, pin_memory=True, strict_single_label=True):
    # Nếu code cũ truyền strict_single_label thì ưu tiên giá trị đó
    if strict_single_label is not None:
        strict_single_label = strict_single_label
    # Tạo 3 dataset trước
    train_dataset, val_dataset, test_dataset = get_datasets(data_dir=data_dir,strict_single_label=strict_single_label)

    # Train loader: shuffle=True để xáo trộn dữ liệu khi học
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,   # train thì shuffle
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    # Validation loader: không shuffle để đánh giá ổn định
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # val/test không shuffle
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    # Test loader: không shuffle
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader, test_loader

# Lấy danh sách tên class từ dataset train
def get_class_names(train_dataset):
    return train_dataset.classes


if __name__ == "__main__":
    # Ví dụ chạy:
    # Chạy bằng:
    #   python -m src.data_pipeline.dataset
    #
    # Nhớ thay data_dir thành package thật trên máy bạn.
    data_dir = r"D:\StudyMaterials\HK6\DeepLearning\Group\data\dataset_openimages_yolo_packages\packages\pkg_001"

    try:
        train_loader, val_loader, test_loader = get_dataloaders(
            data_dir=data_dir,
            batch_size=4,
            num_workers=0,
            pin_memory=False,
            strict_single_label=True
        )

        print("Tạo DataLoader thành công.")
        print(f"Số batch train: {len(train_loader)}")
        print(f"Số batch val  : {len(val_loader)}")
        print(f"Số batch test : {len(test_loader)}")

        # Lấy 1 batch đầu tiên để xem shape.
        images, labels = next(iter(train_loader))
        print("Train batch image shape:", images.shape)
        print("Train batch label shape:", labels.shape)

    except Exception as e:
        print("Lỗi khi test dataset.py:", e)

# python -m pip install pyyaml

# python -m data_pipeline.dataset