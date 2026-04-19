import os
from pathlib import Path

'''
    Vai trò:
        - Kiểm tra nhanh dữ liệu trước khi train.
        - Kiểm tra phân bố class.
        - Kiểm tra shape của một batch từ DataLoader.
        - Kiểm tra xem thư mục ảnh / nhãn có tồn tại và có đủ cặp ảnh-nhãn hay không.
'''

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

"""
     Đếm số lượng sample của từng class trong dataset.
"""
def check_class_distribution(dataset):
    # Dictionary lưu số lượng mẫu theo từng label.
    class_count = {}

    # Duyệt toàn bộ sample trong dataset.
    for _, label in dataset:
        # Nếu label chưa có trong dict thì khởi tạo.
        if label not in class_count:
            class_count[label] = 0
        # Tăng số lượng của label đó lên 1.
        class_count[label] += 1

    # In kết quả ra màn hình
    print("Class distribution:")
    for k, v in class_count.items():
        print(f"Class {k}: {v} samples")

"""
    In shape của batch đầu tiên để kiểm tra dữ liệu vào model.
"""
def check_batch_shape(loader):
    # Chỉ lấy batch đầu tiên rồi dừng.
    for images, labels in loader:
        print("Image batch shape:", images.shape)
        print("Label batch shape:", labels.shape)
        break

"""
    Kiểm tra dữ liệu train/val/test có đủ thư mục ảnh và nhãn hay không.
    Đồng thời kiểm tra mỗi ảnh có file nhãn tương ứng hay không.
"""
def validate_image_files(data_dir):
    # Duyệt từng split.
    for split in ["train", "val", "test"]:
        # Xác định thư mục ảnh.
        image_dir = Path(data_dir) / "images" / split
        # Xác định thư mục nhãn.
        label_dir = Path(data_dir)  / "labels" / split
        # Nếu thiếu thư mục ảnh thì báo và chuyển split tiếp theo.
        if not image_dir.exists():
            print(f"{split}: thiếu thư mục ảnh -> {image_dir}")
            continue
        # Nếu thiếu thư mục nhãn thì báo và chuyển split tiếp theo.
        if not label_dir.exists():
            print(f"{split}: thiếu thư mục nhãn -> {label_dir}")
            continue
        # Lọc danh sách file ảnh hợp lệ.
        image_files = [p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
        print(f"{split}: {len(image_files)} ảnh")
        # In số lượng ảnh.
        mising_labels = 0
         # Với mỗi ảnh, kiểm tra có file nhãn cùng tên hay không.
        for img in image_files:
            lbl = label_dir / f"{img.stem}.txt"
            if not lbl.exists():
                mising_labels += 1
        # In kết quả kiểm tra cặp ảnh/nhãn.
        if mising_labels > 0:
            print(f"{split}: thiếu {mising_labels} file nhãn")

        else:
            print(f"{split}: đủ cặp ảnh/nhãn")


if __name__ == "__main__":
    # Ví dụ chạy:
    # python -m src.data_pipeline.check_data
    data_dir = r"D:\StudyMaterials\HK6\DeepLearning\Group\data\dataset_openimages_yolo"

    try:
        validate_image_files(data_dir)
    except Exception as e:
        print("Lỗi khi kiểm tra dữ liệu:", e)