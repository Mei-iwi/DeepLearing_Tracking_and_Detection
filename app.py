import os
from datetime import datetime

import torch

from src.data_pipeline.dataset import get_dataloaders
from src.models.backbone import Backbone
from src.models.headprep import HeadPrep
from src.models.cnn_model import CurrentCNN
from src.training.loss import build_criterion
from src.training.optimizer import build_optimizer
from src.training.engine import train_one_epoch, validate_one_epoch
from src.training.checkpoint import save_checkpoint, load_checkpoint


def write_status(status_path, status_text):
    with open(status_path, "w", encoding="utf-8") as f:
        f.write(status_text)


def read_status(status_path):
    if not os.path.exists(status_path):
        return ""
    with open(status_path, "r", encoding="utf-8") as f:
        return f.read()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    member_name = input("Tên thành viên: ").strip()
    package_name = input("Package (ví dụ pkg_001): ").strip()

    packages_root = r"G:\My Drive\DataOpenImageV7\dataset_openimages_yolo_packages\packages"
    data_dir = os.path.join(packages_root, package_name)

    if not os.path.exists(data_dir):
        print(f"Không tìm thấy package: {data_dir}")
        return

    batch_size = 8
    num_workers = 0
    pin_memory = (device == "cuda")
    epochs = 5
    lr = 1e-3
    strict_single_label = True

    shared_ckpt_dir = r"G:\My Drive\DeepLearning\Model1\checkpoints_shared"
    os.makedirs(shared_ckpt_dir, exist_ok=True)

    latest_ckpt_path = os.path.join(shared_ckpt_dir, "latest_resume_model.pth")
    best_ckpt_path = os.path.join(shared_ckpt_dir, "best_global_model.pth")
    status_path = os.path.join(shared_ckpt_dir, "training_status.txt")

    # Kiểm tra trạng thái trước khi train
    old_status = read_status(status_path)
    if "status: busy" in old_status:
        print("Hiện đang có người khác train. Kiểm tra training_status.txt trước khi chạy tiếp.")
        print(old_status)
        return

    # Ghi trạng thái bận
    write_status(
        status_path,
        f"status: busy\n"
        f"current_user: {member_name}\n"
        f"current_package: {package_name}\n"
        f"start_time: {datetime.now()}\n"
    )

    try:
        train_loader, val_loader, test_loader = get_dataloaders(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            strict_single_label=strict_single_label
        )

        class_names = train_loader.dataset.classes #type:ignore
        n_classes = len(class_names)

        print("Classes:", class_names)
        print("Num classes:", n_classes)
        print("Train samples:", len(train_loader.dataset)) #type:ignore
        print("Val samples:", len(val_loader.dataset)) #type:ignore
        print("Test samples:", len(test_loader.dataset)) #type:ignore

        backbone = Backbone()
        headprep = HeadPrep()
        model = CurrentCNN(
            backbone=backbone,
            headprep=headprep,
            n_classes=n_classes
        ).to(device)

        criterion = build_criterion()
        optimizer = build_optimizer(model, lr=lr)

        start_epoch = 1
        best_val_acc = -1.0

        # Nạp checkpoint mới nhất để train tiếp
        if os.path.exists(latest_ckpt_path):
            model, optimizer, start_epoch, best_val_acc = load_checkpoint(
                model=model,
                optimizer=optimizer,
                checkpoint_path=latest_ckpt_path,
                device=device
            )
            print(f"Đã nạp checkpoint mới nhất: {latest_ckpt_path}")
            print(f"Train tiếp từ epoch: {start_epoch}")
            print(f"Best val acc hiện tại: {best_val_acc:.4f}")

        # Train thêm 'epochs' vòng kể từ checkpoint hiện tại
        end_epoch = start_epoch + epochs - 1

        for epoch in range(start_epoch, end_epoch + 1):
            train_loss, train_acc = train_one_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device
            )

            val_loss, val_acc = validate_one_epoch(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device
            )

            print(f"\nEpoch [{epoch}/{end_epoch}]")
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

            # Luôn lưu checkpoint mới nhất để người sau train tiếp
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_metric=best_val_acc,
                save_path=latest_ckpt_path
            )

            # Chỉ lưu best global nếu tốt hơn
            if val_acc > best_val_acc:
                best_val_acc = val_acc

                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    best_metric=best_val_acc,
                    save_path=latest_ckpt_path
                )

                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    best_metric=best_val_acc,
                    save_path=best_ckpt_path
                )

                print(f"-> Đã cập nhật best global checkpoint tại: {best_ckpt_path}")

        # Final test: nạp lại best global rồi test
        if os.path.exists(best_ckpt_path):
            model, optimizer, _, best_metric = load_checkpoint(
                model=model,
                optimizer=optimizer,
                checkpoint_path=best_ckpt_path,
                device=device
            )
            print(f"\nĐã nạp best global checkpoint | best_val_acc = {best_metric:.4f}")

        test_loss, test_acc = validate_one_epoch(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device
        )

        print("\n--------------------- FINAL TEST ---------------------")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Acc: {test_acc:.4f}")

        # Ghi trạng thái rảnh
        write_status(
            status_path,
            f"status: free\n"
            f"last_user: {member_name}\n"
            f"last_package: {package_name}\n"
            f"last_time: {datetime.now()}\n"
            f"latest_checkpoint: {latest_ckpt_path}\n"
            f"best_checkpoint: {best_ckpt_path}\n"
            f"best_val_acc: {best_val_acc:.4f}\n"
        )

    except Exception as e:
        # Nếu lỗi cũng nên ghi lại để người khác biết
        write_status(
            status_path,
            f"status: free\n"
            f"last_user: {member_name}\n"
            f"last_package: {package_name}\n"
            f"last_time: {datetime.now()}\n"
            f"error: {e}\n"
        )
        raise


if __name__ == "__main__":
    main()