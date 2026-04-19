from torch.utils.data import Dataset
from .dataset import _load_yaml, _parse_names, _resolve_images_dir, resolve_label_dir_from_image_dir
from collections import Counter
from PIL import Image

# Danh sách các ảnh hợp lệ
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

class YoloSingleLabelDataset(Dataset):
    '''
        Dataset tối giản để giữ nguyên pipelien classifier hiện tại.
        Mỗi ảnh trả về đúng 1 label:
            - strict_single_labels=True: chỉ giữ ảnh có đúng 1 class duy nhất
            - strict_single_label=False: lấy class xuất hiện nhiều nhất trong ảnh
    '''
    def __init__(self, data_dir, split, transform=None, strict_single_label=True):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.strict_single_label = strict_single_label

        cfg = _load_yaml(data_dir)
        self.classes = _parse_names(cfg["names"])

        self.image_dir = _resolve_images_dir(data_dir, split)
        self.label_dir = resolve_label_dir_from_image_dir(self.image_dir)

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Không tìm thấy thư mục ảnh: {self.image_dir}")
        if not self.label_dir.exists():
            raise FileNotFoundError(f"Không tìm thấy thư mục nhãn: {self.label_dir}")
        
        self.samples = []

        for image_path in sorted(self.image_dir.iterdir()):
            if not image_path.is_file() or image_path.suffix.lower() not in IMG_EXTS:
                continue
            label_path = self.label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                continue
                
            rows = []

            with open(label_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    rows.append(int(parts[0]))
            
            if not rows:
                continue

            unique_ids = sorted(set(rows))

            if self.strict_single_label:
                if len(unique_ids) != 1:
                    continue
                label_id = unique_ids[0]
            else: 
                label_id = Counter(rows).most_common(1)[0][0]
            
            self.samples.append((image_path, label_id))
        
        if not self.samples:
            raise RuntimeError(f"Không tìm thấy sample hợp lệ cho split='{split}' ở {self.image_dir}")
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label