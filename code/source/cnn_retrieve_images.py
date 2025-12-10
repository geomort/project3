
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

from tqdm import tqdm
from PIL import Image

import numpy as np
import random
import time
import json
import yaml

from pathlib import Path
from collections import defaultdict


def set_seed(seed: int = 42, deterministic: bool = True):
    """
    Docstrings made with Copilot and edited
    Set random seeds for reproducibility across Python, NumPy, and PyTorch.

    Parameters
    ----------
    seed : int, optional
        Seed value for random number generators. Default is 42.
    deterministic : bool, optional
        If True, enforce deterministic behavior in PyTorch CUDA operations
        (may reduce performance). Default is True.

    Effects
    -------
    - Sets seeds for `random`, `numpy`, `torch`, and all CUDA devices.
    - Configures PyTorch backend for deterministic execution if requested.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Makes CUDA ops deterministic (slower but reproducible)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class YOLODataset(Dataset):

    """
    Docstrings made with Copilot and edited
    Load a YOLO-style dataset with optional class subsetting, balanced sampling,
    resizing, and configurable normalization. Displays progress bars during
    data collection and normalization.

    Parameters
    ----------
    yaml_path : str or pathlib.Path
        Path to `data.yaml` in YOLO format (must include `nc`, `names`, and split paths).
    subset_classes : list[str] or None, optional
        Class names to include (case-insensitive). If None, use all classes. Default is None.
    max_images : int or None, optional
        Per-class cap to balance the dataset. If None, include all available. Default is None.
    split : str, optional
        Dataset split: `'train'`, `'val'`/`'valid'`, or `'test'`. Default is `'train'`.
    img_size : tuple[int, int], optional
        Target image size `(H, W)` for resizing prior to `ToTensor`. Default is `(224, 224)`.
    normalize : bool or {'auto', 'compute', 'load', 'ImageNet'}, optional
        Normalization mode:
        - True: same as `'ImageNet'`.
        - False: no normalization.
        - 'auto': compute mean/std from a sample; load from `norm_file` if it exists.
        - 'compute': force computation of mean/std and optionally save to `norm_file`.
        - 'load': load mean/std from `norm_file` (file must exist).
        - 'ImageNet': use ImageNet mean/std.
        Default is True.
    norm_file : str or pathlib.Path or None, optional
        JSON file to load/save normalization stats when using `'auto'`, `'compute'`, or `'load'`. Default is None.
    debug : bool, optional
        If True, print debug information (paths, normalization, sampling). Default is False.
    auto_norm_sample : int, optional
        Number of images to sample when estimating mean/std for `'auto'`/`'compute'`. Default is 500.
    crop_strategy : {'largest', 'none'}, optional
        Cropping strategy:
        - 'largest': crop around the largest bounding box for the selected class.
        - 'none': use full image.
        Default is 'largest'.
    seed : int, optional
        RNG seed for reproducible shuffling/sampling. Default is 42.

    Attributes
    ----------
    class_names : list[str]
        Lowercased class names from `data.yaml`.
    num_classes : int
        Number of classes (`nc` in `data.yaml` or len(`names`)).
    class_to_idx_full : dict[str, int]
        Mapping from original class names to indices.
    subset_class_names : list[str]
        Active class names after subsetting (lowercased).
    remap : dict[int, int]
        Mapping from original YOLO class IDs to compact indices for the subset.
    pairs : list[tuple[pathlib.Path, list[tuple[int, float, float, float, float]]]]
        Collected items as `(image_path, boxes_all)` where `boxes_all` are YOLO labels:
        `(cls, cx, cy, w, h)` in normalized coordinates.
    transform : torchvision.transforms.Compose
        Composed transforms applied to images (resize, tensor, optional normalize).
    crop_strategy : str
        Active crop strategy ('largest' or 'none').

    Returns
    -------
    YOLODataset
        Initialized dataset ready for use with a PyTorch `DataLoader`.

    Raises
    ------
    KeyError
        If the requested `split` is not present in `data.yaml`.
    ValueError
        If `subset_classes` contains unknown class names or `crop_strategy` is invalid.
    FileNotFoundError
        If `'load'` normalization is requested and `norm_file` does not exist.
    RuntimeError
        If no items are found after filtering and balancing.

    Prints
    ------
    - Debug messages (paths, normalization mode/stats, sampling details) when `debug=True`.
    - Per-class counts after balanced sampling.

    Progress
    --------
    Displays `tqdm` progress bars:
    - "Loading dataset" for uncapped collection (`max_images is None`).
    - "Balanced sampling" for per-class capped collection.
    - "Computing mean/std" when estimating normalization statistics.

    __getitem__ Output
    ------------------
    dict with keys:
    - 'image' : torch.Tensor
        Transformed image tensor.
    - 'label' : int
        Remapped class index for the selected box.
    - 'class_name' : str
        Class name corresponding to 'label'.
    - 'path' : str
        Image file path.
    - 'crop_size' : tuple[int, int]
        (width, height) of the cropped region (or original image if no crop).
    - 'orig_size' : tuple[int, int]
        (width, height) of the original image.
    """

    def __init__(self, yaml_path, subset_classes=None, max_images=None, split='train',
                 img_size=(224, 224), normalize=True, norm_file=None, debug=False, auto_norm_sample=500,
                 crop_strategy="largest", seed=42):

        yaml_path = Path(yaml_path)
        with open(yaml_path, "r") as f:
            data_cfg = yaml.safe_load(f)

        # RNG for reproducibility
        self.rng = random.Random(seed)

        self.debug = debug
        self.class_names = [str(x).lower() for x in data_cfg.get("names", [])]
        self.num_classes = int(data_cfg.get("nc", len(self.class_names)))
        self.class_to_idx_full = {name: i for i, name in enumerate(self.class_names)}

        # Resolve split key/value from YAML
        split_map = {"train": "train", "val": "val", "valid": "valid", "test": "test"}
        split_key = split_map.get(split, split)
        split_value = data_cfg.get(split_key) or data_cfg.get("valid") or data_cfg.get("val")
        if split_value is None:
            raise KeyError(f"Split '{split}' not found in data.yaml")

        # Subset handling & remapping (original YOLO IDs -> compact indices)
        if subset_classes is None:
            self.subset_class_names = self.class_names
            self.remap = {i: i for i in range(self.num_classes)}
            subset_ids = set(range(self.num_classes))
        else:
            normalized_subset = [c.lower() for c in subset_classes]
            missing = [c for c in normalized_subset if c not in self.class_to_idx_full]
            if missing:
                raise ValueError(f"Unknown subset classes: {missing}")
            self.subset_class_names = normalized_subset
            self.remap = {self.class_to_idx_full[c]: i for i, c in enumerate(normalized_subset)}
            subset_ids = set(self.remap.keys())

        # Collect balanced image/label pairs with EARLY STOP (randomized order)
        self.pairs = self._collect_pairs_balanced(
            yaml_path=yaml_path,
            split_value=split_value,
            subset_ids=subset_ids,
            max_images=max_images,
            debug=debug
        )

        if len(self.pairs) == 0:
            raise RuntimeError("YOLODataset found 0 items after filtering.")

        # Transforms
        base_transforms = [T.Resize(img_size), T.ToTensor()]

        if normalize in ('auto', 'compute', 'load'):
            norm_file = Path(norm_file) if norm_file else None

            if normalize == 'load':
                if not norm_file or not norm_file.exists():
                    raise FileNotFoundError(f"Normalization file {norm_file} not found for 'load' mode.")
                if debug:
                    print(f"[DEBUG] Loading normalization from {norm_file}")
                with open(norm_file, "r") as f:
                    norm_data = json.load(f)
                mean = torch.tensor(norm_data["mean"])
                std = torch.tensor(norm_data["std"])

            elif normalize == 'compute':
                if debug:
                    print("[DEBUG] Computing dataset mean/std (compute mode)...")
                mean, std = self._compute_mean_std(base_transforms, auto_norm_sample)
                if norm_file:
                    with open(norm_file, "w") as f:
                        json.dump({"mean": mean.tolist(), "std": std.tolist()}, f)
                    if debug:
                        print(f"[DEBUG] Saved normalization to {norm_file}")

            elif normalize == 'auto':
                if norm_file and norm_file.exists():
                    if debug:
                        print(f"[DEBUG] Loading normalization from {norm_file} (auto mode)")
                    with open(norm_file, "r") as f:
                        norm_data = json.load(f)
                    mean = torch.tensor(norm_data["mean"])
                    std = torch.tensor(norm_data["std"])
                else:
                    if debug:
                        print("[DEBUG] Computing dataset mean/std (auto mode)...")
                    mean, std = self._compute_mean_std(base_transforms, auto_norm_sample)
                    if norm_file:
                        with open(norm_file, "w") as f:
                            json.dump({"mean": mean.tolist(), "std": std.tolist()}, f)
                        if debug:
                            print(f"[DEBUG] Saved normalization to {norm_file}")

            base_transforms.append(T.Normalize(mean.tolist(), std.tolist()))

        elif normalize == 'ImageNet':
            imagenet_mean = [0.485, 0.456, 0.406]
            imagenet_std = [0.229, 0.224, 0.225]
            base_transforms.append(T.Normalize(mean=imagenet_mean, std=imagenet_std))
            if debug:
                print(f"[DEBUG] Using ImageNet normalization: mean={imagenet_mean}, std={imagenet_std}")

        # normalize=False → no normalization
        self.transform = T.Compose(base_transforms)




        self.crop_strategy = crop_strategy.lower()
        if self.crop_strategy not in ("largest", "none"):
            raise ValueError("crop_strategy must be 'largest' or 'none'")

    # ----------------------------
    # Data collection (balanced)
    # ----------------------------

    def _collect_pairs_balanced(self, yaml_path, split_value, subset_ids, max_images, debug):
        IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
        candidates = []
        for raw_path in ([split_value] if isinstance(split_value, str) else split_value):
            resolved = (yaml_path.parent / raw_path).resolve()
            images_dir = resolved
            labels_dir = resolved.parent / "labels" if "images" in resolved.name else resolved / "labels"

            if debug:
                print(f"[DEBUG] Images dir: {images_dir}")
                print(f"[DEBUG] Labels dir: {labels_dir}")

            if not images_dir.exists():
                continue

            for img_file in images_dir.rglob("*"):
                if img_file.suffix.lower() in IMG_EXTS:
                    candidates.append((img_file, labels_dir))

        self.rng.shuffle(candidates)

        # Wrap the main loop with tqdm
        pairs = []
        if max_images is None:
            for img_file, labels_dir in tqdm(candidates, desc="Loading dataset", unit="img"):
                lbl_file = labels_dir / f"{img_file.stem}.txt"
                if lbl_file.exists():
                    boxes = self._read_yolo_label_file(lbl_file)
                    if any(cls in subset_ids for cls, *_ in boxes):
                        pairs.append((img_file, boxes))
            if debug:
                print(f"[DEBUG] Cached {len(pairs)} images (no per-class cap).")
            return pairs

        # Balanced sampling with progress bar
        class_counts = {self.remap[orig]: 0 for orig in subset_ids}
        target = max_images
        for img_file, labels_dir in tqdm(candidates, desc="Balanced sampling", unit="img"):
            if all(count >= target for count in class_counts.values()):
                break
            lbl_file = labels_dir / f"{img_file.stem}.txt"
            if not lbl_file.exists():
                continue
            boxes_all = self._read_yolo_label_file(lbl_file)
            boxes_subset = [(cls, cx, cy, w, h) for (cls, cx, cy, w, h) in boxes_all if cls in subset_ids]
            if not boxes_subset:
                continue
            chosen_cls_orig = boxes_subset[0][0]
            chosen_cls = self.remap[chosen_cls_orig]
            if class_counts[chosen_cls] < target:
                pairs.append((img_file, boxes_all))
                class_counts[chosen_cls] += 1

        if debug:
            print("[DEBUG] Per-class counts after balanced sampling:")
            for subset_idx, count in class_counts.items():
                cname = self.subset_class_names[subset_idx]
                print(f"  {cname}: {count}")

        return pairs


    # ----------------------------
    # Stats (optional normalization)
    # ----------------------------

    def _compute_mean_std(self, transforms, sample_size):
        sample_size = min(sample_size, len(self.pairs))
        indices = self.rng.sample(range(len(self.pairs)), sample_size)
        mean = torch.zeros(3)
        std = torch.zeros(3)
        raw_transform = T.Compose(transforms[:2])  # Resize + ToTensor

        # Wrap loop with tqdm
        for i in tqdm(indices, desc="Computing mean/std", unit="img"):
            img_path, _ = self.pairs[i]
            img = Image.open(img_path).convert("RGB")
            img_tensor = raw_transform(img)
            mean += img_tensor.mean(dim=(1, 2))
            std += img_tensor.std(dim=(1, 2))

        mean /= sample_size
        std /= sample_size
        return mean, std


    def __len__(self):
        return len(self.pairs)

    @staticmethod
    def _yolo_norm_to_xyxy_pixels(cx, cy, w, h, W, H):
        x_center = cx * W
        y_center = cy * H
        bw = w * W
        bh = h * H
        x1 = int(x_center - bw / 2)
        y1 = int(y_center - bh / 2)
        x2 = int(x_center + bw / 2)
        y2 = int(y_center + bh / 2)
        return x1, y1, x2, y2

    def _read_yolo_label_file(self, lbl_path):
        boxes = []
        with open(lbl_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts or len(parts) < 5:
                    continue
                cls = int(parts[0])
                cx, cy, w, h = map(float, parts[1:5])
                boxes.append((cls, cx, cy, w, h))
        return boxes


    def __getitem__(self, idx):
        img_path, boxes_all = self.pairs[idx]
        img = Image.open(img_path).convert("RGB")
        W_orig, H_orig = img.size

        subset_ids = set(self.remap.keys())
        boxes_subset = [(cls, cx, cy, w, h) for (cls, cx, cy, w, h) in boxes_all if cls in subset_ids]

        if self.crop_strategy == "largest":
            selected = max(boxes_subset, key=lambda b: b[3] * b[4])
        else:
            selected = boxes_subset[0]

        cls, cx, cy, w, h = selected
        crop_w, crop_h = W_orig, H_orig
        if self.crop_strategy == "largest":
            x1, y1, x2, y2 = self._yolo_norm_to_xyxy_pixels(cx, cy, w, h, W_orig, H_orig)
            x1 = max(0, x1); y1 = max(0, y1); x2 = min(W_orig, x2); y2 = min(H_orig, y2)
            if x2 > x1 and y2 > y1:
                img = img.crop((x1, y1, x2, y2))
                crop_w, crop_h = (x2 - x1), (y2 - y1)

        img_tensor = self.transform(img)
        return {
            "image": img_tensor,
            "label": self.remap[cls],
            "class_name": self.subset_class_names[self.remap[cls]],
            "path": str(img_path),
            "crop_size": (crop_w, crop_h),   # <--- new
            "orig_size": (W_orig, H_orig)    # <--- optional
        }



def classification_collate_fn(batch):
    """
    Docstrings made with Copilot and edited
    Collate a batch of classification samples into tensors and lists for DataLoader.

    Parameters
    ----------
    batch : list of dict
        Each element is a dictionary with keys:
        - 'image' : torch.Tensor
            Transformed image tensor.
        - 'label' : int
            Class index.
        - 'path' : str
            Image file path.
        - 'class_name' : str
            Class name.

    Returns
    -------
    dict
        - 'images' : torch.Tensor
            Stacked image tensors of shape (batch_size, C, H, W).
        - 'labels' : torch.Tensor
            Tensor of class indices (dtype=torch.long).
        - 'paths' : list of str
            Original image file paths.
        - 'class_names' : list of str
            Class names for each sample.
    """
    images = torch.stack([b["image"] for b in batch], dim=0)
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    paths = [b["path"] for b in batch]
    class_names = [b["class_name"] for b in batch]

    return {
        "images": images,
        "labels": labels,
        "paths": paths,
        "class_names": class_names
    }