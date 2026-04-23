from __future__ import annotations

import argparse
import os
from pathlib import Path


def extract_names_block(dataset_yaml: Path) -> str:
    text = dataset_yaml.read_text(encoding='utf-8')
    marker = 'names:'
    idx = text.find(marker)
    if idx == -1:
        raise RuntimeError(f"Khong doc duoc block 'names:' tu {dataset_yaml}")
    return text[idx:].rstrip() + '\n'


def extract_nc(dataset_yaml: Path) -> int | None:
    for line in dataset_yaml.read_text(encoding='utf-8').splitlines():
        if line.strip().startswith('nc:'):
            try:
                return int(line.split(':', 1)[1].strip())
            except Exception:
                return None
    return None


def to_rel_posix(from_dir: Path, to_path: Path) -> str:
    return os.path.relpath(to_path, start=from_dir).replace('\\', '/')


def rewrite_package_yaml(pkg_root: Path, shared_eval_root: Path) -> None:
    yaml_path = pkg_root / 'dataset.yaml'
    if not yaml_path.exists():
        raise FileNotFoundError(f'Khong tim thay {yaml_path}')

    names_block = extract_names_block(yaml_path)
    nc = extract_nc(yaml_path)

    val_rel = to_rel_posix(pkg_root, shared_eval_root / 'images' / 'val')
    test_rel = to_rel_posix(pkg_root, shared_eval_root / 'images' / 'test')

    lines = [
        'path: .',
        'train: images/train',
        f'val: {val_rel}',
        f'test: {test_rel}',
    ]
    if nc is not None:
        lines.append(f'nc: {nc}')
    lines.append(names_block.rstrip())

    yaml_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def write_train_commands(packages_root: Path, model: str, epochs: int, imgsz: int, batch: int) -> None:
    bat_lines = [
        '@echo off',
        'set SCRIPT_DIR=%~dp0',
    ]
    txt_lines = []

    for pkg in sorted(p for p in packages_root.iterdir() if p.is_dir() and p.name.startswith('pkg_')):
        yaml_rel_win = f'{pkg.name}\\dataset.yaml'
        project_rel_win = f'{pkg.name}\\runs'
        bat_cmd = (
            f'yolo task=detect mode=train model="{model}" data="%SCRIPT_DIR%{yaml_rel_win}" '
            f'epochs={epochs} imgsz={imgsz} batch={batch} project="%SCRIPT_DIR%{project_rel_win}" name=train'
        )
        bat_lines.append(bat_cmd)

        txt_cmd = (
            f'yolo task=detect mode=train model="{model}" data="./{pkg.name}/dataset.yaml" '
            f'epochs={epochs} imgsz={imgsz} batch={batch} project="./{pkg.name}/runs" name=train'
        )
        txt_lines.append(txt_cmd)

    (packages_root / 'train_all_packages.bat').write_text('\n'.join(bat_lines) + '\n', encoding='utf-8')
    (packages_root / 'train_all_packages.txt').write_text('\n'.join(txt_lines) + '\n', encoding='utf-8')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Rewrite existing package dataset.yaml files to relative paths'
    )
    parser.add_argument(
        '--packages-root',
        default='./shared_storage/dataset_openimages_yolo_packages/packages',
        help='Directory containing pkg_001, pkg_002, ...',
    )
    parser.add_argument(
        '--shared-eval-root',
        default='./shared_storage/dataset_openimages_yolo_packages/shared_eval',
        help='Directory containing shared_eval/images and shared_eval/labels',
    )
    parser.add_argument('--model', default='yolov8n.pt')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    packages_root = Path(args.packages_root)
    shared_eval_root = Path(args.shared_eval_root)

    if not packages_root.exists():
        raise FileNotFoundError(f'Khong tim thay packages_root: {packages_root}')
    if not shared_eval_root.exists():
        raise FileNotFoundError(f'Khong tim thay shared_eval_root: {shared_eval_root}')

    count = 0
    for pkg_root in sorted(p for p in packages_root.iterdir() if p.is_dir() and p.name.startswith('pkg_')):
        rewrite_package_yaml(pkg_root, shared_eval_root)
        count += 1

    write_train_commands(
        packages_root=packages_root,
        model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
    )

    print(f'[DONE] Rewrote {count} package yaml files under {packages_root}')
    print(f'[DONE] Updated command files in {packages_root}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
