from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path

import torch

from src.data_pipeline.dataset import get_dataloaders
from src.models.backbone import Backbone
from src.models.headprep import HeadPrep
from src.models.cnn_model import CurrentCNN
from src.training.loss import build_criterion
from src.training.optimizer import build_optimizer
from src.training.engine import train_one_epoch, validate_one_epoch
from src.training.checkpoint import save_checkpoint, load_checkpoint


def is_colab() -> bool:
    return "COLAB_GPU" in os.environ or "COLAB_RELEASE_TAG" in os.environ


def get_default_base_dir() -> Path:
    env_base = os.environ.get("DL_BASE_DIR", "").strip()
    if env_base:
        return Path(env_base)

    if is_colab():
        return Path("/content/drive/MyDrive/DeepLearningData")

    gdrive_base = Path(r"G:\My Drive")
    if gdrive_base.exists():
        return gdrive_base

    return Path.cwd()


def get_default_packages_root() -> Path:
    env_root = os.environ.get("PKG_ROOT", "").strip()
    if env_root:
        return Path(env_root)

    base = get_default_base_dir()

    if is_colab():
        return base / "dataset_openimages_yolo_packages" / "packages"

    gdrive_candidate = Path(r"G:\My Drive\DataOpenImageV7\dataset_openimages_yolo_packages\packages")
    if gdrive_candidate.exists():
        return gdrive_candidate

    return base / "dataset_openimages_yolo_packages" / "packages"


def get_default_ckpt_dir() -> Path:
    env_root = os.environ.get("CKPT_ROOT", "").strip()
    if env_root:
        return Path(env_root)

    base = get_default_base_dir()

    if is_colab():
        return base / "checkpoints_shared"

    gdrive_candidate = Path(r"G:\My Drive\DeepLearning\Model1\checkpoints_shared")
    if gdrive_candidate.exists():
        return gdrive_candidate

    return base / "checkpoints_shared"


def write_status(status_path: Path, status_text: str) -> None:
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(status_text, encoding="utf-8")


def read_status(status_path: Path) -> str:
    if not status_path.exists():
        return ""
    return status_path.read_text(encoding="utf-8")


def print_image_progress(phase, batch_idx, total_batches, processed_samples, batch_size):
    start_img_idx = processed_samples + 1
    end_img_idx = processed_samples + batch_size
    print(
        f"[{phase}] Batch {batch_idx}/{total_batches} | "
        f"Đang xử lý ảnh {start_img_idx}-{end_img_idx}"
    )


def build_model(n_classes: int, device: str):
    backbone = Backbone()
    headprep = HeadPrep()
    model = CurrentCNN(
        backbone=backbone,
        headprep=headprep,
        n_classes=n_classes
    ).to(device)
    return model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train classifier theo package, chạy được cả local và Colab"
    )

    parser.add_argument("--member-name", type=str, default="", help="Tên thành viên")
    parser.add_argument("--package-name", type=str, default="", help="Tên package, ví dụ: pkg_001")

    parser.add_argument(
        "--packages-root",
        type=str,
        default=str(get_default_packages_root()),
        help="Thư mục chứa các package",
    )
    parser.add_argument(
        "--shared-ckpt-dir",
        type=str,
        default=str(get_default_ckpt_dir()),
        help="Thư mục chứa checkpoint dùng chung",
    )

    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=-1, help="-1 = tự chọn theo môi trường")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument(
        "--strict-single-label",
        dest="strict_single_label",
        action="store_true",
        default=True,
        help="Chỉ giữ ảnh có đúng 1 class duy nhất",
    )
    parser.add_argument(
        "--no-strict-single-label",
        dest="strict_single_label",
        action="store_false",
        help="Cho phép nhiều class, lấy class xuất hiện nhiều nhất",
    )

    parser.add_argument(
        "--ignore-busy",
        action="store_true",
        help="Bỏ qua trạng thái busy trong training_status.txt",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Không nạp latest checkpoint để train tiếp",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    member_name = args.member_name.strip()
    if not member_name:
        member_name = input("Tên thành viên: ").strip()

    package_name = args.package_name.strip()
    if not package_name:
        package_name = input("Package (ví dụ pkg_001): ").strip()

    packages_root = Path(args.packages_root)
    data_dir = packages_root / package_name

    if not data_dir.exists():
        print(f"Không tìm thấy package: {data_dir}")
        return

    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    strict_single_label = args.strict_single_label

    if args.num_workers == -1:
        num_workers = 2 if device == "cuda" else 0
    else:
        num_workers = args.num_workers

    pin_memory = device == "cuda"

    shared_ckpt_dir = Path(args.shared_ckpt_dir)
    shared_ckpt_dir.mkdir(parents=True, exist_ok=True)

    latest_ckpt_path = shared_ckpt_dir / "latest_resume_model.pth"
    best_ckpt_path = shared_ckpt_dir / "best_global_model.pth"
    status_path = shared_ckpt_dir / "training_status.txt"

    old_status = read_status(status_path)
    if (not args.ignore_busy) and ("status: busy" in old_status):
        print("Hiện đang có người khác train. Kiểm tra training_status.txt trước khi chạy tiếp.")
        print(old_status)
        return

    write_status(
        status_path,
        f"status: busy\n"
        f"current_user: {member_name}\n"
        f"current_package: {package_name}\n"
        f"start_time: {datetime.now()}\n"
        f"device: {device}\n"
        f"data_dir: {data_dir}\n"
    )

    try:
        train_loader, val_loader, test_loader = get_dataloaders(
            data_dir=str(data_dir),
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            strict_single_label=strict_single_label
        )

        class_names = train_loader.dataset.classes  # type: ignore
        n_classes = len(class_names)

        print("Classes:", class_names)
        print("Num classes:", n_classes)
        print("Train samples:", len(train_loader.dataset))  # type: ignore
        print("Val samples:", len(val_loader.dataset))      # type: ignore
        print("Test samples:", len(test_loader.dataset))    # type: ignore

        model = build_model(n_classes=n_classes, device=device)
        criterion = build_criterion()
        optimizer = build_optimizer(model, lr=lr)

        start_epoch = 1
        best_val_acc = -1.0

        if latest_ckpt_path.exists() and (not args.no_resume):
            model, optimizer, start_epoch, best_val_acc = load_checkpoint(
                model=model,
                optimizer=optimizer,
                checkpoint_path=str(latest_ckpt_path),
                device=device
            )
            print(f"Đã nạp checkpoint mới nhất: {latest_ckpt_path}")
            print(f"Train tiếp từ epoch: {start_epoch}")
            print(f"Best val acc hiện tại: {best_val_acc:.4f}")

        end_epoch = start_epoch + epochs - 1

        for epoch in range(start_epoch, end_epoch + 1):
            train_loss, train_acc = train_one_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                progress_fn=print_image_progress
            )

            val_loss, val_acc = validate_one_epoch(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                phase="VAL",
                progress_fn=print_image_progress
            )

            print(f"\nEpoch [{epoch}/{end_epoch}]")
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_metric=best_val_acc,
                save_path=str(latest_ckpt_path)
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc

                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    best_metric=best_val_acc,
                    save_path=str(latest_ckpt_path)
                )

                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    best_metric=best_val_acc,
                    save_path=str(best_ckpt_path)
                )

                print(f"-> Đã cập nhật best global checkpoint tại: {best_ckpt_path}")

        if best_ckpt_path.exists():
            model, optimizer, _, best_metric = load_checkpoint(
                model=model,
                optimizer=optimizer,
                checkpoint_path=str(best_ckpt_path),
                device=device
            )
            print(f"\nĐã nạp best global checkpoint | best_val_acc = {best_metric:.4f}")

        test_loss, test_acc = validate_one_epoch(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
            phase="TEST",
            progress_fn=print_image_progress
        )

        print("\n--------------------- FINAL TEST ---------------------")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Acc: {test_acc:.4f}")

        write_status(
            status_path,
            f"status: free\n"
            f"last_user: {member_name}\n"
            f"last_package: {package_name}\n"
            f"last_time: {datetime.now()}\n"
            f"latest_checkpoint: {latest_ckpt_path}\n"
            f"best_checkpoint: {best_ckpt_path}\n"
            f"best_val_acc: {best_val_acc:.4f}\n"
            f"device: {device}\n"
            f"data_dir: {data_dir}\n"
        )

    except Exception as e:
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