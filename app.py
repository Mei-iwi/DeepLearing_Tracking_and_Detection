from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import torch

# Đảm bảo chạy được khi gọi: python app.py từ thư mục gốc project
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.data_pipeline.dataset import get_dataloaders
from src.models.common.factory import build_model
from src.training.loss import build_criterion
from src.training.optimizer import build_optimizer
from src.training.engine import train_one_epoch, validate_one_epoch
from src.training.checkpoint import save_checkpoint, load_checkpoint


def get_repo_root() -> Path:
    """
    Trả về thư mục gốc project.
    app.py đang nằm ở root nên mặc định ROOT là đủ.
    """
    here = Path(__file__).resolve().parent

    for candidate in [here, *here.parents]:
        if (candidate / "src").exists():
            return candidate

    return here


def normalize_user_path(path_str: str | Path) -> Path:
    """
    Cho phép truyền path tuyệt đối hoặc path tương đối theo repo.
    Ví dụ:
        shared_storage/checkpoints_shared
        D:/Data/checkpoints_shared
    """
    p = Path(path_str).expanduser()

    if not p.is_absolute():
        p = get_repo_root() / p

    return p.resolve()


def get_default_shared_root() -> Path:
    """
    Root dùng chung cho data và checkpoint.
    Ưu tiên biến môi trường DL_SHARED_ROOT nếu có.
    Nếu không, dùng repo/shared_storage.
    """
    env_root = os.environ.get("DL_SHARED_ROOT", "").strip()
    if env_root:
        return normalize_user_path(env_root)

    return get_repo_root() / "shared_storage"


def get_default_packages_root() -> Path:
    """
    Mặc định package nằm ở:
    shared_storage/dataset_openimages_yolo_packages/packages
    """
    env_root = os.environ.get("PKG_ROOT", "").strip()
    if env_root:
        return normalize_user_path(env_root)

    return (
        get_default_shared_root()
        / "dataset_openimages_yolo_packages"
        / "packages"
    )


def get_default_ckpt_root() -> Path:
    """
    Mặc định checkpoint nằm ở:
    shared_storage/checkpoints_shared

    Lưu ý: checkpoint cụ thể sẽ được lưu theo từng model:
    shared_storage/checkpoints_shared/model_1/latest_resume_model.pth
    shared_storage/checkpoints_shared/model_1/best_model.pth
    shared_storage/checkpoints_shared/model_1/training_status.txt
    """
    env_root = os.environ.get("CKPT_ROOT", "").strip()
    if env_root:
        return normalize_user_path(env_root)

    return get_default_shared_root() / "checkpoints_shared"


def write_status(status_path: Path, status_text: str) -> None:
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(status_text, encoding="utf-8")


def read_status(status_path: Path) -> str:
    if not status_path.exists():
        return ""
    return status_path.read_text(encoding="utf-8")


def print_image_progress(phase, batch_idx, total_batches, processed_samples, batch_size):
    """
    Giữ hàm này để tương thích với train_one_epoch / validate_one_epoch.
    Hiện tại không in từng batch để terminal đỡ rối.
    """
    _ = (phase, batch_idx, total_batches, processed_samples, batch_size)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train classifier theo package, hỗ trợ chọn model"
    )

    parser.add_argument(
        "--member-name",
        type=str,
        default="",
        help="Tên thành viên train",
    )

    parser.add_argument(
        "--package-name",
        type=str,
        default="",
        help="Tên package, ví dụ: pkg_001",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="model_1",
        choices=["model_1", "model_2", "model_3", "model_4"],
        help="Tên model cần train",
    )

    parser.add_argument(
        "--packages-root",
        type=str,
        default=str(get_default_packages_root()),
        help="Thư mục chứa các package",
    )

    parser.add_argument(
        "--shared-ckpt-root",
        type=str,
        default=str(get_default_ckpt_root()),
        help="Thư mục gốc chứa checkpoint dùng chung",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=-1,
        help="-1 = tự chọn theo device, CPU dùng 0, CUDA dùng 2",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Số epoch train thêm",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate",
    )

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
        help="Cho phép ảnh nhiều class, lấy class xuất hiện nhiều nhất",
    )

    parser.add_argument(
        "--ignore-busy",
        action="store_true",
        help="Bỏ qua trạng thái busy trong training_status.txt",
    )

    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Không load latest checkpoint để train tiếp",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    repo_root = get_repo_root()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Repo root:", repo_root)
    print("Device:", device)
    print("Model:", args.model_name)

    member_name = args.member_name.strip()
    if not member_name:
        member_name = input("Tên thành viên: ").strip()

    package_name = args.package_name.strip()
    if not package_name:
        package_name = input("Package, ví dụ pkg_001: ").strip()

    packages_root = normalize_user_path(args.packages_root)
    shared_ckpt_root = normalize_user_path(args.shared_ckpt_root)

    data_dir = packages_root / package_name

    print("Packages root:", packages_root)
    print("Checkpoint root:", shared_ckpt_root)
    print("Data dir:", data_dir)

    if not data_dir.exists():
        print(f"Không tìm thấy package: {data_dir}")
        print("Kiểm tra lại shared_storage hoặc truyền --packages-root cho đúng.")
        return

    if args.num_workers == -1:
        num_workers = 2 if device == "cuda" else 0
    else:
        num_workers = args.num_workers

    pin_memory = device == "cuda"

    # Checkpoint dùng chung theo từng model, không tách theo package.
    # Ví dụ:
    # shared_storage/checkpoints_shared/model_1/latest_resume_model.pth
    # shared_storage/checkpoints_shared/model_1/best_model.pth
    # shared_storage/checkpoints_shared/model_1/training_status.txt
    model_ckpt_dir = shared_ckpt_root / args.model_name
    model_ckpt_dir.mkdir(parents=True, exist_ok=True)

    latest_ckpt_path = model_ckpt_dir / "latest_resume_model.pth"
    best_ckpt_path = model_ckpt_dir / "best_model.pth"
    status_path = model_ckpt_dir / "training_status.txt"

    old_status = read_status(status_path)
    if (not args.ignore_busy) and ("status: busy" in old_status):
        print("Model này đang được train. Kiểm tra training_status.txt trước khi chạy tiếp.")
        print(old_status)
        return

    write_status(
        status_path,
        f"status: busy\n"
        f"current_user: {member_name}\n"
        f"current_model: {args.model_name}\n"
        f"current_package: {package_name}\n"
        f"start_time: {datetime.now()}\n"
        f"device: {device}\n"
        f"data_dir: {data_dir}\n",
    )

    try:
        train_loader, val_loader, test_loader = get_dataloaders(
            data_dir=str(data_dir),
            batch_size=args.batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            strict_single_label=args.strict_single_label,
        )

        class_names = train_loader.dataset.classes  # type: ignore[attr-defined]
        n_classes = len(class_names)

        print("Classes:", class_names)
        print("Num classes:", n_classes)
        print("Train samples:", len(train_loader.dataset))  # type: ignore[arg-type]
        print("Val samples:", len(val_loader.dataset))      # type: ignore[arg-type]
        print("Test samples:", len(test_loader.dataset))    # type: ignore[arg-type]

        # Build model thông qua factory.
        model = build_model(
            model_name=args.model_name,
            n_classes=n_classes,
            device=device,
        )

        criterion = build_criterion()
        optimizer = build_optimizer(model, lr=args.lr)

        start_epoch = 1
        best_val_acc = -1.0

        if latest_ckpt_path.exists() and (not args.no_resume):
            model, optimizer, start_epoch, best_val_acc = load_checkpoint(
                model=model,
                optimizer=optimizer,
                checkpoint_path=str(latest_ckpt_path),
                device=device,
            )

            print(f"Đã nạp checkpoint mới nhất: {latest_ckpt_path}")
            print(f"Train tiếp từ epoch: {start_epoch}")
            print(f"Best val acc hiện tại: {best_val_acc:.4f}")

        end_epoch = start_epoch + args.epochs - 1

        for epoch in range(start_epoch, end_epoch + 1):
            train_loss, train_acc = train_one_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                progress_fn=print_image_progress,
            )

            val_loss, val_acc = validate_one_epoch(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                phase="VAL",
                progress_fn=print_image_progress,
            )

            print(f"\nEpoch [{epoch}/{end_epoch}]")
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_metric=best_val_acc,
                save_path=str(latest_ckpt_path),
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc

                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    best_metric=best_val_acc,
                    save_path=str(latest_ckpt_path),
                )

                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    best_metric=best_val_acc,
                    save_path=str(best_ckpt_path),
                )

                print(f"-> Đã cập nhật best checkpoint: {best_ckpt_path}")

        if best_ckpt_path.exists():
            model, optimizer, _, best_metric = load_checkpoint(
                model=model,
                optimizer=optimizer,
                checkpoint_path=str(best_ckpt_path),
                device=device,
            )

            print(f"\nĐã nạp best checkpoint | best_val_acc = {best_metric:.4f}")

        test_loss, test_acc = validate_one_epoch(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
            phase="TEST",
            progress_fn=print_image_progress,
        )

        print("\n--------------------- FINAL TEST ---------------------")
        print(f"Model: {args.model_name}")
        print(f"Package: {package_name}")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Acc: {test_acc:.4f}")

        write_status(
            status_path,
            f"status: free\n"
            f"last_user: {member_name}\n"
            f"last_model: {args.model_name}\n"
            f"last_package: {package_name}\n"
            f"last_time: {datetime.now()}\n"
            f"latest_checkpoint: {latest_ckpt_path}\n"
            f"best_checkpoint: {best_ckpt_path}\n"
            f"best_val_acc: {best_val_acc:.4f}\n"
            f"device: {device}\n"
            f"data_dir: {data_dir}\n",
        )

    except Exception as e:
        write_status(
            status_path,
            f"status: free\n"
            f"last_user: {member_name}\n"
            f"last_model: {args.model_name}\n"
            f"last_package: {package_name}\n"
            f"last_time: {datetime.now()}\n"
            f"error: {e}\n",
        )
        raise


if __name__ == "__main__":
    main()
