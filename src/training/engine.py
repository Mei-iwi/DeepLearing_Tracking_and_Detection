import torch

'''
    Vai trò:
        - train_one_epach(...): huấn luyện mô hình trong 1 epoch
        - validate_one_epoch(...): đánh giá mô hình trên tập validation trong 1 epoch
'''

def train_one_epoch(model, loader, criterion, optimizer, device):
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
    for images, labels in loader:
        # Chuyển dữ liệu sang cùng thiết bị với model
        images = images.to(device)
        labels = labels.to(device)

        # Xóa gradient cũng trước khi tính gradient mới
        optimizer.zero_grad()
        # Cho batch ảnh qua model để lấy đầu ra [N, n_classes]
        logits = model(images)
        # Tính hàm mất mát giữa dữ đoán và nhãn
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
            Tính số dự đoán trong batch
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
            + Accurrucy của epoch
    '''
    return total_loss / total_sample, total_correct / total_sample

def validate_one_epoch(model, loader, criterion, device):
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
        # Duyệt qua từng batch validation
        for images, labels in loader:
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
            total_loss += loss.item()  * images.size(0)
            '''
            Tính số dự đoán trong batch
                - logits.argmax(dim=1): lấy lớp có điểm cao nhất
                - so sánh với labels
                - .sum().item(): đếm số mẫu dự đoán đúng
            '''
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
             # Cộng số lượng mẫu của batch vào tổng số mẫu
            total_sample += labels.size(0)
    '''
        - Trả về:
            + Validation trung bình 
            + Validation Accurrucy
    '''
    return total_loss / total_sample, total_correct / total_sample

'''
    Khi model.train()
       -  Dropout sẽ hoạt động
       - BatchNorm sẽ cập nhật thống kê trong lúc train
    Luồng hoạt động
        - Train
            batch ảnh
            -> model(images)
            -> tính loss
            -> backward
            -> optimizer.step()
            -> cộng dồn loss và accuracy
        - Validation
            batch ảnh
            -> model(images)
            -> tính loss
            -> không backward
            -> không update trọng số
            -> cộng dồn loss và accuracy

'''