import torch

'''
    Vai trò:
        - train_one_epoch(...): huấn luyện mô hình trong 1 epoch
        - validate_one_epoch(...): đánh giá mô hình trên tập validation/test trong 1 epoch
'''

def train_one_epoch(model, loader, criterion, optimizer, device, progress_fn=None):
    # Chuyển mô hình sang huấn luyện
    model.train()

    '''
        Khởi tạo:
            + Tổng loss toàn bộ mẫu
            + Tổng số dự đoán đúng
            + Tổng số mẫu đi qua
    '''
    total_loss, total_correct, total_sample = 0.0, 0, 0

    # Duyệt qua từng batch trong dữ liệu train
    for batch_idx, (images, labels) in enumerate(loader, start=1):
        batch_size = images.size(0)

        # Nếu có hàm callback từ app.py thì dùng callback đó để in tiến độ
        if progress_fn is not None:
            progress_fn("TRAIN", batch_idx, len(loader), total_sample, batch_size)
        else:
            start_img_idx = total_sample + 1
            end_img_idx = total_sample + batch_size
            print(
                f"[TRAIN] Batch {batch_idx}/{len(loader)} | "
                f"Đang xử lý ảnh {start_img_idx}-{end_img_idx}"
            )

        # Chuyển dữ liệu sang cùng thiết bị với model
        images = images.to(device)
        labels = labels.to(device)

        # Xóa gradient cũ trước khi tính gradient mới
        optimizer.zero_grad()

        # Cho batch ảnh qua model để lấy đầu ra [N, n_classes]
        logits = model(images)

        # Tính hàm mất mát giữa dự đoán và nhãn
        loss = criterion(logits, labels)

        # Lan truyền ngược để tính gradient cho các tham số
        loss.backward()

        # Cập nhật trọng số model dựa trên gradient vừa tính
        optimizer.step()

        '''
            Cộng dồn tổng loss của batch vào tổng loss của model
                - loss.item() là loss trung bình của batch
                - images.size(0) là số mẫu trong batch
        '''
        total_loss += loss.item() * images.size(0)

        '''
            Tính số dự đoán đúng trong batch
                - logits.argmax(dim=1): lấy lớp có điểm cao nhất
                - so sánh với labels
                - .sum().item(): đếm số mẫu dự đoán đúng
        '''
        total_correct += (logits.argmax(dim=1) == labels).sum().item()

        # Cộng số lượng mẫu của batch vào tổng số mẫu
        total_sample += labels.size(0)

    '''
        - Trả về:
            + Loss trung bình epoch
            + Accuracy của epoch
    '''
    return total_loss / total_sample, total_correct / total_sample


def validate_one_epoch(model, loader, criterion, device, phase="VAL", progress_fn=None):
    # Chuyển mô hình sang đánh giá
    model.eval()

    '''
        Khởi tạo:
            + Tổng loss toàn bộ mẫu
            + Tổng số dự đoán đúng
            + Tổng số mẫu đi qua
    '''
    total_loss, total_correct, total_sample = 0.0, 0, 0

    # Tắt gradient trong quá trình validation
    with torch.no_grad():
        # Duyệt qua từng batch validation/test
        for batch_idx, (images, labels) in enumerate(loader, start=1):
            batch_size = images.size(0)

            # Nếu có hàm callback từ app.py thì dùng callback đó để in tiến độ
            if progress_fn is not None:
                progress_fn(phase, batch_idx, len(loader), total_sample, batch_size)
            else:
                start_img_idx = total_sample + 1
                end_img_idx = total_sample + batch_size
                print(
                    f"[{phase}] Batch {batch_idx}/{len(loader)} | "
                    f"Đang xử lý ảnh {start_img_idx}-{end_img_idx}"
                )

            # Chuyển dữ liệu sang cùng thiết bị với mô hình
            images = images.to(device)
            labels = labels.to(device)

            # Cho batch đi qua model để lấy đầu ra
            logits = model(images)

            # Tính hàm mất mát giữa dự đoán và nhãn thật
            loss = criterion(logits, labels)

            '''
                Cộng dồn tổng loss của batch vào tổng loss của model
                    - loss.item() là loss trung bình của batch
                    - images.size(0) là số mẫu trong batch
            '''
            total_loss += loss.item() * images.size(0)

            '''
                Tính số dự đoán đúng trong batch
                    - logits.argmax(dim=1): lấy lớp có điểm cao nhất
                    - so sánh với labels
                    - .sum().item(): đếm số mẫu dự đoán đúng
            '''
            total_correct += (logits.argmax(dim=1) == labels).sum().item()

            # Cộng số lượng mẫu của batch vào tổng số mẫu
            total_sample += labels.size(0)

    '''
        - Trả về:
            + Validation/Test loss trung bình
            + Validation/Test accuracy
    '''
    return total_loss / total_sample, total_correct / total_sample


'''
    Khi model.train()
       - Dropout sẽ hoạt động
       - BatchNorm sẽ cập nhật thống kê trong lúc train

    Luồng hoạt động:
        - Train
            batch ảnh
            -> model(images)
            -> tính loss
            -> backward
            -> optimizer.step()
            -> cộng dồn loss và accuracy

        - Validation / Test
            batch ảnh
            -> model(images)
            -> tính loss
            -> không backward
            -> không update trọng số
            -> cộng dồn loss và accuracy
'''