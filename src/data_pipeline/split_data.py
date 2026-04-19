import os
import random
import shutil


def split_dataset_by_ratio(raw_dir, output_dir, train_ratio, val_ratio, test_ratio, seed=42):
    # kiểm tra tổng tỉ lệ
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Tổng ratio phải = 1"

    random.seed(seed)

    # tạo folder output
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)

    # lấy danh sách ảnh
    images = [f for f in os.listdir(raw_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

    random.shuffle(images)

    total = len(images)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_files = images[:train_end]
    val_files = images[train_end:val_end]
    test_files = images[val_end:]

    # hàm copy
    def copy_files(file_list, split):
        for f in file_list:
            src = os.path.join(raw_dir, f)
            dst = os.path.join(output_dir, "images", split, f)
            shutil.copy2(src, dst)

    copy_files(train_files, "train")
    copy_files(val_files, "val")
    copy_files(test_files, "test")

    print("DONE split dataset")
    print(f"Train: {len(train_files)}")
    print(f"Val: {len(val_files)}")
    print(f"Test: {len(test_files)}")