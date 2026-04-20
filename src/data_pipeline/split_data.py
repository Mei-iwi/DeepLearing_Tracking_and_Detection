import os
import random
import shutil

'''
    Vai trò
        - Chia một bộ dữ liệu thành train / val / test theo tỉ lệ.
        - Giữ đúng cặp (ảnh, nhãn) khi copy sang thư mục mới.
        - Dùng khi bạn có dữ liệu YOLO ở dạng:
            dataset_root/
                images/
                labels/
'''


"""
    Chia dữ liệu thành 3 phần train / val / test theo tỉ lệ.

    Parameters
    ----------
    dataset_root : str
        Thư mục gốc chứa images/ và labels/.
    output_dir : str
        Thư mục đầu ra sau khi chia.
    train_ratio : float
    val_ratio : float
    test_ratio : float
    seed : int
        Seed để shuffle có thể lặp lại kết quả.
"""
def split_dataset_by_ratio(dataset_root, output_dir, train_ratio, val_ratio, test_ratio, seed=42):
    # Đảm bảo tổng 3 ratio bằng 1.0.
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Tổng ratio phải = 1"

    # Cố định seed để chia dữ liệu ổn định giữa các lần chạy.
    random.seed(seed)

    # Tạo thư mục output cho ảnh và nhãn của từng split.
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "labels", split), exist_ok=True)

    # Thư mục ảnh gốc.
    images_dir = os.path.join(dataset_root, 'images')

    # Thư mục nhãn gốc.
    labels_dir = os.path.join(dataset_root, "labels")
    
    # Danh sách đuôi file ảnh hợp lệ.
    valid_exts = (".jpg", ".png", ".jpeg", ".bmp", ".webp")

    # Danh sách cặp (ảnh, nhãn) hợp lệ.
    pairs = []

    # Duyệt toàn bộ file trong thư mục images.
    for f in os.listdir(images_dir):
        # Bỏ qua file không phải ảnh.
        if not f.lower().endswith(valid_exts):
            continue
        # Suy ra tên file nhãn tương ứng
        label_name = os.path.splitext(f)[0] + ".txt"
        label_path = os.path.join(labels_dir, label_name)
        # Chỉ giữ các cặp có đủ cả ảnh và nhãn.
        if os.path.exists(label_path):
            pairs.append((f, label_name))
    # Trộn ngẫu nhiên thứ tự cặp dữ liệu.
    random.shuffle(pairs)
    # Tính số lượng mẫu.
    total = len(pairs)
    # Mốc kết thúc train.
    train_end = int(total * train_ratio)
    # Mốc kết thúc val.
    val_end = train_end + int(total * val_ratio)
    # Cắt danh sách thành 3 phần.
    train_files = pairs[:train_end]
    val_files = pairs[train_end:val_end]
    test_files = pairs[val_end:]

    """
        Copy đồng thời ảnh và nhãn sang split tương ứng.
    """
    def copy_pairs(file_list, split):
         # Đường dẫn ảnh nguồn và đích.
        for img_name, lbl_name in file_list:
            # Đường dẫn ảnh nguồn và đích.
            img_src = os.path.join(images_dir, img_name)
            img_dst = os.path.join(output_dir, "images", split, img_name)

            # Đường dẫn nhãn nguồn và đích.
            lbl_src = os.path.join(labels_dir, lbl_name)
            lbl_dst = os.path.join(output_dir, "labels", split, lbl_name)

            # Copy ảnh.
            shutil.copy2(img_src, img_dst)

            # Copy nhãn.
            shutil.copy2(lbl_src, lbl_dst)
    # Copy dữ liệu của từng split.
    copy_pairs(train_files, "train")
    copy_pairs(val_files, "val")
    copy_pairs(test_files, "test")
    # In thông tin tổng kết.
    print("DONE split dataset")
    print(f"Train: {len(train_files)}")
    print(f"Val: {len(val_files)}")
    print(f"Test: {len(test_files)}")

if __name__ == "__main__":
    # Ví dụ chạy:
    # python src/data_pipeline/split_data.py

    # Thư mục dữ liệu gốc dạng phẳng:
    # dataset_root/
    # ├── images/
    # └── labels/
    dataset_root = r"D:\StudyMaterials\HK6\DeepLearning\Group\data\raw_yolo_flat"

    # Thư mục đầu ra sau khi chia train / val / test
    output_dir = r"D:\StudyMaterials\HK6\DeepLearning\Group\data\split_yolo_dataset"

    try:
        split_dataset_by_ratio(
            dataset_root=dataset_root,
            output_dir=output_dir,
            train_ratio=0.7,
            val_ratio=0.2,
            test_ratio=0.1,
            seed=42
        )
    except Exception as e:
        print("Lỗi khi chia dữ liệu:", e)