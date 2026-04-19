import os

# Kiểm tra phân bố dữ liệu giữa các lớp
def check_class_distribution(dataset):
    class_count = {}

    for _, label in dataset:
        if label not in class_count:
            class_count[label] = 0
        class_count[label] += 1

    print("Class distribution:")
    for k, v in class_count.items():
        print(f"Class {k}: {v} samples")

# Kiểm tra dữ liệu đầu vào của mô hình có đúng kích thước batch và số chiều hay không
def check_batch_shape(loader):
    for images, labels in loader:
        print("Image batch shape:", images.shape)
        print("Label batch shape:", labels.shape)
        break

# Kiểm tra dữ liệu đầu vào có hợp lệ
def validate_image_files(data_dir):
    for split in ["train", "val", "test"]:
        path = os.path.join(data_dir, "images", split)

        if not os.path.exists(path):
            print(f"{split} folder không tồn tại")
            continue

        files = os.listdir(path)

        if len(files) == 0:
            print(f"{split} rỗng")
        else:
            print(f"{split}: {len(files)} files")

        # check file lỗi
        for f in files:
            if not f.lower().endswith((".jpg", ".png", ".jpeg")):
                print(f"File không hợp lệ: {f}")