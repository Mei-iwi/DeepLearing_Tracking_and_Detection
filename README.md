# Hướng dẫn thiết lập đường dẫn dùng chung cho huấn luyện

## Mục đích
Chuẩn hóa đường dẫn trong repo để mọi thành viên đều dùng chung code mà không phải sửa path theo từng máy.

## Cấu trúc thư mục mong muốn
```text
repo_root/
  app.py
  shared_storage/
    dataset_openimages_yolo_packages/   <- link tới thư mục dataset trên Google Drive
      packages/
      shared_eval/
    checkpoints_shared/                 <- link tới thư mục checkpoint trên Google Drive
```

## Ý tưởng
Mỗi thành viên tự tạo 2 junction trong thư mục `shared_storage` của repo:

- `shared_storage/dataset_openimages_yolo_packages`
- `shared_storage/checkpoints_shared`

Khi đó code chỉ cần dùng **đường dẫn tương đối**, không cần hard-code đường dẫn tuyệt đối theo từng máy.

## Ví dụ `dataset.yaml`
```yaml
path: .
train: images/train
val: ../../shared_eval/images/val
test: ../../shared_eval/images/test
```

## Ví dụ lệnh chạy
```bash
python app.py --member-name cuong --package-name pkg_001
```

## Lệnh tạo junction trên Windows
Chạy trong thư mục gốc của repo:

```cmd
mkdir shared_storage
mklink /J "shared_storage\checkpoints_shared" "G:\My Drive\DeepLearning\Model1\checkpoints_shared"
mklink /J "shared_storage\dataset_openimages_yolo_packages" "G:\My Drive\DataOpenImageV7\dataset_openimages_yolo_packages"
dir /AL shared_storage
```

## Đổi lại đường dẫn tương đối trên file yaml của mỗi gói: -> Quan trọng

```
python src/help/rewrite_existing_package_yaml_relative.py 
```

## Lưu ý quan trọng
Phải tạo:

```text
shared_storage\dataset_openimages_yolo_packages
```

trỏ tới:

```text
G:\My Drive\DataOpenImageV7\dataset_openimages_yolo_packages
```

**Không trỏ thẳng vào thư mục `packages`**, vì project còn cần cả:

- `packages`
- `shared_eval`

## Kết luận
Cách làm này giúp mọi thành viên dùng cùng một cấu trúc đường dẫn, giảm lỗi và không phải sửa code theo từng máy.


## 1. Cấu trúc thư mục

```text
project/
├── .venv/
├── .vscode/
├── src/
│   ├── data_pipeline/
│   │   ├── __pycache__/
│   │   ├── check_data.py
│   │   ├── dataset.py
│   │   ├── split_data.py
│   │   ├── transforms.py
│   │   └── YoloSingleLabelDataset
│   │
│   ├── evaluation/
│   │   ├── evaluate.py
│   │   ├── metrics.py
│   │   └── predict.py
│   │
│   ├── models/
│   │   ├── __pycache__/
│   │   ├── backbone.py
│   │   ├── classifier_head.py
│   │   ├── cnn_model.py
│   │   ├── headprep.py
│   │   └── shape_test.py
│   │
│   ├── training/
│   │   ├── __pycache__/
│   │   ├── checkpoint.py
│   │   ├── engine.py
│   │   ├── loss.py
│   │   └── optimizer.py
│   │
│   └── ui/
│       ├── home.py
│       ├── image_page.py
│       └── video_page.py
│
├── .gitignore
├── app.py
├── README.md
└── requirements.txt
```

---

## 2. Giải thích chức năng từng thư mục



### `src/`
Thư mục chứa toàn bộ mã nguồn chính của dự án.

Đây là phần quan trọng nhất, nơi tổ chức các nhóm chức năng như:
- Xử lý dữ liệu
- Định nghĩa mô hình
- Huấn luyện
- Đánh giá
- Giao diện người dùng

---

## 3. Giải thích chi tiết các thư mục con trong `src`

### 3.1 `src/data_pipeline/`
Nhóm file phục vụ chuẩn bị và xử lý dữ liệu đầu vào.


#### `check_data.py`
Dùng để kiểm tra chất lượng và tính hợp lệ của dữ liệu.

Chức năng có thể bao gồm:
- Kiểm tra cấu trúc thư mục dữ liệu
- Kiểm tra số lượng ảnh và nhãn
- Phát hiện file bị thiếu
- Phát hiện nhãn sai định dạng
- Kiểm tra dữ liệu có phù hợp để huấn luyện hay không

#### `dataset.py`
Định nghĩa cách nạp dữ liệu vào mô hình.

Chức năng thường gồm:
- Đọc ảnh từ thư mục
- Đọc nhãn tương ứng
- Chuyển dữ liệu về định dạng tensor
- Trả dữ liệu cho DataLoader để train và evaluate

Đây là cầu nối giữa dữ liệu thô và mô hình.

#### `split_data.py`
Dùng để chia dữ liệu thành các tập con.

Thông thường gồm:
- Train set
- Validation set
- Test set

Mục đích:
- Phục vụ huấn luyện
- Đánh giá mô hình công bằng
- Tránh dùng chung dữ liệu train và test

#### `transforms.py`
Chứa các phép biến đổi dữ liệu.

Ví dụ:
- Resize ảnh
- Normalize
- Augmentation
- Flip, rotate, crop
- Chuyển đổi định dạng tensor

Vai trò:
- Chuẩn hóa dữ liệu đầu vào
- Tăng độ đa dạng của dữ liệu huấn luyện
- Giúp mô hình học tốt hơn

#### `YoloSingleLabelDataset`

- Đọc dữ liệu nhãn YOLO
- Tổ chức ảnh và label theo format phù hợp
- Tích hợp với pipeline huấn luyện hoặc suy luận

---

### 3.2 `src/evaluation/`
Nhóm file dùng để đánh giá mô hình và sinh kết quả dự đoán.

#### `evaluate.py`
Chạy đánh giá mô hình trên tập validation hoặc test.

Chức năng có thể gồm:
- Load mô hình đã huấn luyện
- Chạy dự đoán trên tập đánh giá
- Thu thập kết quả
- Gọi metric để tính các chỉ số hiệu năng

#### `metrics.py`
Chứa các hàm đo lường hiệu năng mô hình.

Ví dụ:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

Tách riêng file này giúp dễ bảo trì và tái sử dụng.

#### `predict.py`
Thực hiện suy luận trên dữ liệu mới.

Chức năng có thể gồm:
- Dự đoán một ảnh
- Dự đoán một tập ảnh
- Hỗ trợ dự đoán video hoặc frame
- Trả về nhãn dự đoán và độ tin cậy

---

### 3.3 `src/models/`
Thư mục định nghĩa kiến trúc mô hình.

#### `__pycache__/`
Thư mục cache tự sinh của Python.

#### `backbone.py`
Định nghĩa phần backbone của mô hình.

Vai trò:
- Trích xuất đặc trưng từ ảnh đầu vào
- Tạo feature map cho các tầng sau xử lý

Thông thường backbone là phần CNN chính của mô hình.

#### `classifier_head.py`
Định nghĩa phần head dùng cho phân loại.

Vai trò:
- Nhận đặc trưng từ backbone
- Chuyển thành đầu ra phân lớp
- Sinh logits hoặc xác suất dự đoán

#### `cnn_model.py`
Lắp ghép toàn bộ mô hình CNN hoàn chỉnh.

Chức năng:
- Kết hợp backbone và classifier head
- Định nghĩa hàm `forward()`
- Tạo kiến trúc cuối cùng dùng để train và predict

#### `headprep.py`
File hỗ trợ xử lý dữ liệu trước khi đưa vào head.

Có thể gồm:
- Flatten tensor
- Reshape tensor
- Chuẩn bị đầu ra backbone để khớp với classifier head

Đây là file hỗ trợ kết nối giữa các phần trong mô hình.

#### `shape_test.py`
Dùng để kiểm tra kích thước tensor trong pipeline mô hình.

Mục tiêu:
- Phát hiện lỗi sai chiều tensor
- Kiểm tra output backbone có khớp input head không
- Hỗ trợ debug nhanh khi xây dựng kiến trúc

---

### 3.4 `src/training/`
Nhóm file phục vụ quá trình huấn luyện mô hình.


#### `checkpoint.py`
Quản lý checkpoint trong lúc train.

Chức năng thường gồm:
- Lưu trọng số mô hình
- Tải checkpoint để train tiếp
- Lưu mô hình tốt nhất
- Khôi phục trạng thái train khi cần

#### `engine.py`
Đây thường là bộ điều phối chính của quá trình train.

Chức năng có thể gồm:
- Chạy một epoch huấn luyện
- Chạy một epoch đánh giá
- Tính loss
- Backpropagation
- Logging kết quả train/val

Có thể xem đây là file trung tâm của quá trình huấn luyện.

#### `loss.py`
Định nghĩa hàm mất mát.

Ví dụ:
- CrossEntropyLoss
- BCE
- Custom loss

Vai trò:
- Đo sai khác giữa dự đoán và nhãn thật
- Là tiêu chí để mô hình tối ưu trong quá trình học

#### `optimizer.py`
Cấu hình thuật toán tối ưu.

Ví dụ:
- SGD
- Adam
- AdamW
- Learning rate scheduler

Vai trò:
- Cập nhật trọng số mô hình sau mỗi lần tính gradient

---

### 4.5 `src/ui/`
Nhóm file giao diện người dùng.

#### `home.py`
Trang chủ của ứng dụng.

Chức năng có thể gồm:
- Hiển thị giới thiệu dự án
- Cung cấp menu điều hướng
- Mô tả các tính năng chính

#### `image_page.py`
Trang xử lý ảnh.

Chức năng có thể gồm:
- Tải ảnh lên
- Chạy dự đoán
- Hiển thị ảnh và kết quả phân loại hoặc nhận diện

#### `video_page.py`
Trang xử lý video.

Chức năng có thể gồm:
- Tải video lên
- Chạy suy luận trên từng frame
- Hiển thị kết quả dự đoán trên video

---

## 5. Giải thích các file ở thư mục gốc

### `.gitignore`
Khai báo các file và thư mục Git không theo dõi.

Thường dùng để bỏ qua:
- `.venv/`
- `__pycache__/`
- File log
- File model lớn
- Dữ liệu sinh tạm

---

### `app.py`
File khởi chạy chính của ứng dụng.

Vai trò có thể gồm:
- Điểm bắt đầu của chương trình
- Gọi các thành phần trong `ui/`
- Kết nối giao diện với phần dự đoán mô hình

Nếu dự án có giao diện demo, đây thường là file entry point để chạy ứng dụng.

---

### `README.md`
Tài liệu mô tả dự án.

Thông thường bao gồm:
- Mục tiêu dự án
- Cấu trúc thư mục
- Cách cài đặt
- Cách chạy
- Mô tả chức năng

---

### `requirements.txt`
Danh sách các thư viện cần cài để chạy dự án.

Ví dụ thường gặp:
- torch
- torchvision
- numpy
- opencv-python
- streamlit
- pandas

Cài đặt bằng lệnh:

```bash
pip install -r requirements.txt
```

---

## 6. Luồng hoạt động tổng quát của dự án

Có thể hình dung luồng xử lý của dự án theo sơ đồ sau:

```text
Dữ liệu đầu vào
   ↓
data_pipeline/
   ↓
models/
   ↓
training/
   ↓
evaluation/
   ↓
ui/ + app.py
```

Giải thích:
1. Dữ liệu được kiểm tra, chia tập và biến đổi trong `data_pipeline/`
2. Mô hình được định nghĩa trong `models/`
3. Quá trình huấn luyện diễn ra trong `training/`
4. Kết quả được đo lường trong `evaluation/`
5. Giao diện người dùng tương tác qua `ui/` và `app.py`


## Mô hình kiến trúc
<img width="2752" height="1536" alt="Models" src="https://github.com/user-attachments/assets/c9550d5b-1bf9-4b39-96b6-f8b33dad7175" />

## Quy trình nạp dữ liệu '

<img width="2752" height="1536" alt="Data_Loader" src="https://github.com/user-attachments/assets/87eb07be-9043-4f37-b4da-60053bc6eddc" />

