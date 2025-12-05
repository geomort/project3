import random
import shutil
from pathlib import Path
from PIL import Image
import yaml



#original_image_folder = Path(r"C:\Users\brumor\Onedrive - Statens Kartverk\PhD\courses\fys-stk4155\projects\project3\datasets\agropest12")


from sklearn.model_selection import train_test_split
import shutil
import random
from pathlib import Path

from sklearn.model_selection import train_test_split
import shutil
import random
from collections import Counter

import shutil
from pathlib import Path
import yaml
import random



import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, Subset
import yaml
import random
from torchvision.transforms import ToTensor

import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class YOLOClassificationDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None, visualize=False):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.visualize = visualize
        self.image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.jpg')])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        label_path = os.path.join(self.labels_dir, img_name.replace('.jpg', '.txt'))

        image = Image.open(img_path).convert("RGB")
        width, height = image.size

        boxes = []
        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    cls_id = int(parts[0])
                    x_center, y_center, w, h = map(float, parts[1:])
                    # Convert YOLO normalized coords to absolute
                    x_center *= width
                    y_center *= height
                    w *= width
                    h *= height
                    x_min = max(0, x_center - w / 2)
                    y_min = max(0, y_center - h / 2)
                    x_max = min(width, x_center + w / 2)
                    y_max = min(height, y_center + h / 2)
                    boxes.append((x_min, y_min, x_max, y_max))
                    labels.append(cls_id)

        # Pick largest box
        if boxes:
            areas = [(x2 - x1) * (y2 - y1) for (x1, y1, x2, y2) in boxes]
            largest_idx = areas.index(max(areas))
            x_min, y_min, x_max, y_max = boxes[largest_idx]
            cropped_image = image.crop((x_min, y_min, x_max, y_max))
            cls_id = labels[largest_idx]

            # Visualization if enabled
            if self.visualize:
                self.show_image_with_bbox(image, (x_min, y_min, x_max, y_max), cropped_image)
        else:
            cropped_image = image
            cls_id = -1  # Handle missing label case

        # Apply transform or fallback to ToTensor
        if self.transform:
            cropped_image = self.transform(cropped_image)
        else:
            cropped_image = ToTensor()(cropped_image)

        return cropped_image, cls_id


    @staticmethod
    def show_image_with_bbox(image, bbox, cropped_image=None):
        # Use column layout: 2 rows, 1 column if cropped_image exists
        fig, axes = plt.subplots(2 if cropped_image else 1, 1, figsize=(6, 8))

        # Original image with bounding box
        ax = axes[0] if cropped_image else axes
        ax.imshow(image)
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1],
                                linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        ax.set_title("Original with Bounding Box")
        ax.axis("off")

        # Cropped image below
        if cropped_image:
            axes[1].imshow(cropped_image)
            axes[1].set_title("Cropped Image")
            axes[1].axis("off")

        plt.tight_layout()
        plt.show()



def load_split_dataset(base_dir, split, selected_classes=None, n_images_per_class=None, transform=None, seed=42):
    """
    Load a dataset split (train/valid/test) and optionally filter classes and limit images per class.
    """
    random.seed(seed)
    images_dir = base_dir / split / 'images'
    labels_dir = base_dir / split / 'labels'

    dataset = YOLOClassificationDataset(str(images_dir), str(labels_dir), transform=transform)

    if selected_classes is None and n_images_per_class is None:
        return dataset

    # Group indices by class
    class_to_indices = {cls_id: [] for cls_id in selected_classes}
    for idx, img_name in enumerate(dataset.image_files):
        label_path = os.path.join(labels_dir, img_name.replace('.jpg', '.txt'))
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
                if lines:
                    cls_id = int(lines[0].split()[0])
                    if cls_id in selected_classes:
                        class_to_indices[cls_id].append(idx)

    # Collect filtered indices
    selected_indices = []
    for cls_id, indices in class_to_indices.items():
        random.shuffle(indices)
        if n_images_per_class is not None:
            indices = indices[:n_images_per_class]
        selected_indices.extend(indices)

    return Subset(dataset, selected_indices)






def load_dataset_from_yaml(yaml_path, split="train", n_classes=None, n_images_per_class=None, seed=42, transform=None):
    random.seed(seed)

    # Load YAML config
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    base_dir = Path(yaml_path).parent
    images_dir = base_dir / split / 'images'
    labels_dir = base_dir / split / 'labels'

    class_names = [name.lower() for name in config['names']]

    # Load full dataset
    dataset = YOLOClassificationDataset(str(images_dir), str(labels_dir), transform=transform, visualize=False)
    dataset.class_names = class_names  # attach for reference

    # If no filtering requested, return full dataset
    if n_classes is None and n_images_per_class is None:
        return dataset

    # Build index mapping by class
    class_to_indices = {i: [] for i in range(len(class_names))}
    for idx, img_name in enumerate(dataset.image_files):
        label_path = os.path.join(labels_dir, img_name.replace('.jpg', '.txt'))
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
                if lines:
                    cls_id = int(lines[0].split()[0])  # first class for indexing
                    class_to_indices[cls_id].append(idx)

    # Select classes
    selected_classes = list(range(len(class_names)))
    if n_classes:
        selected_classes = random.sample(selected_classes, n_classes)

    # Collect indices
    selected_indices = []
    for cls_id in selected_classes:
        indices = class_to_indices[cls_id]
        if n_images_per_class is not None:  # Only limit if explicitly set
            if len(indices) > n_images_per_class:
                indices = random.sample(indices, n_images_per_class)
        # If n_images_per_class is None → keep all images
        selected_indices.extend(indices)

    return Subset(dataset, selected_indices)




def check_images_size_equal_dataset(dataset):
    sizes = {}
    for i in range(len(dataset)):
        img, _ = dataset[i]
        size = img.size() if hasattr(img, "size") else img.shape  # Tensor shape
        sizes.setdefault(tuple(size), []).append(i)

    if len(sizes) == 1:
        print(f"All images have the same size: {list(sizes.keys())[0]}")
    else:
        print("Images have different sizes:")
        for size, indices in sizes.items():
            print(f"Size {size}: {len(indices)} samples")





def inspect_dataset(yaml_file, seed_value=42):
    random.seed(seed_value)

    # Load YAML
    if not yaml_file:
        raise FileNotFoundError("No YAML file found in agropest12 folder")

    with open(yaml_file, 'r') as f:
        config = yaml.safe_load(f)

    # Class names in lowercase
    names = [name.lower() for name in config['names']]

    target_classes = names

    images_original = yaml_file.parent 

    # Known variations or typos
    class_variations = {
        "caterpillars": ["caterpillar", "catterpillar", "catterpillars", "caterpillars"],
        "beetles": ["beetle", "beetles"],
        "ants": ["ant", "ants"],
        "bees": ["bee", "bees"],
        "earwigs": ["earwigs", "earwig"],
        "grasshoppers": ["grasshopper", "grasshoppers"],
        "moths": ["moths", "moth"],
        "slugs": ["slug", "slugs"],
        "snails": ["snail", "snails"],
        "wasps": ["wasp", "wasps"],
        "weevils": ["weevil", "weevils"]
    }

    # Collect all images by class
    all_images_by_class = {cls: [] for cls in target_classes}
    for img_file in (images_original / "train" / "images").glob("*.jpg"):
        filename_lower = img_file.stem.lower()
        for class_name in sorted(target_classes, key=len, reverse=True):
            variations = sorted(class_variations.get(class_name, [class_name]), key=len, reverse=True)
            if any(var in filename_lower for var in variations):
                all_images_by_class[class_name].append(img_file)
                break

    # Print summary only
    print("\nNumber of images in train dataset")
    for cls in target_classes:
        print(f"{cls}: {len(all_images_by_class[cls])}")






























"""

def prepare_subset(classes_reduced_number=None, max_images_per_class=None, images_original=original_image_folder, seed_value=42):
    if classes_reduced_number is None:
        dataset_path = images_original
        print('Using full dataset')
    else:
        random.seed(seed_value)

        # Load YAML
        yaml_file = next(images_original.glob("*.yaml"), None) or next(images_original.glob("*.yml"), None)
        if not yaml_file:
            raise FileNotFoundError("No YAML file found in agropest12 folder")

        with open(yaml_file, 'r') as f:
            config = yaml.safe_load(f)

        # Class names in lowercase
        names = [name.lower() for name in config['names']]

        # Randomly select classes
        classes_reduced = random.sample(names, classes_reduced_number)
        target_classes = [c for c in names if c in classes_reduced]

        # Known variations or typos
        class_variations = {
            "caterpillars": ["caterpillar", "catterpillar", "catterpillars", "caterpillars"],
            "beetles": ["beetle", "beetles"],
            "ants": ["ant", "ants"],
            "bees": ["bee", "bees"],
            "earwigs": ["earwigs", "earwig"],
            "grasshoppers": ["grasshopper", "grasshoppers"],
            "moths": ["moths", "moth"],
            "slugs": ["slug", "slugs"],
            "snails": ["snail", "snails"],
            "wasps": ["wasp", "wasps"],
            "weevils": ["weevil", "weevils"]
        }

        # Prepare destination folder
        n_classes = len(target_classes)
        n_images = max_images_per_class if max_images_per_class else "all"
        dest_folder = images_original.parent / f"agropest-n_images_{n_images}-n_classes_{n_classes}"
        dest_folder.mkdir(parents=True, exist_ok=True)

        print(f"Creating subset in: {dest_folder}")

        # Collect all images by class from original train folder
        all_images_by_class = {cls: [] for cls in target_classes}
        for img_file in (images_original / "train" / "images").glob("*.jpg"):
            filename_lower = img_file.stem.lower()
            for class_name in sorted(target_classes, key=len, reverse=True):
                variations = sorted(class_variations.get(class_name, [class_name]), key=len, reverse=True)
                if any(var in filename_lower for var in variations):
                    all_images_by_class[class_name].append(img_file)
                    break

        # Apply max_images_per_class if specified
        for cls in all_images_by_class:
            if max_images_per_class and len(all_images_by_class[cls]) > max_images_per_class:
                all_images_by_class[cls] = random.sample(all_images_by_class[cls], max_images_per_class)

        # Split into train/valid/test (60/20/20)
        for split_name in ["train", "valid", "test"]:
            (dest_folder / split_name / "images").mkdir(parents=True, exist_ok=True)
            (dest_folder / split_name / "labels").mkdir(parents=True, exist_ok=True)

        for cls, img_list in all_images_by_class.items():
            random.shuffle(img_list)
            n_total = len(img_list)
            n_train = int(n_total * 0.6)
            n_valid = int(n_total * 0.2)
            train_imgs = img_list[:n_train]
            valid_imgs = img_list[n_train:n_train+n_valid]
            test_imgs = img_list[n_train+n_valid:]

            # Copy images and labels
            for split_name, imgs in zip(["train", "valid", "test"], [train_imgs, valid_imgs, test_imgs]):
                img_dest = dest_folder / split_name / "images"
                lbl_dest = dest_folder / split_name / "labels"
                for img in imgs:
                    shutil.copy(img, img_dest)
                    label_file = img.with_suffix('.txt').parent.parent / "labels" / img.name.replace('.jpg', '.txt')
                    if label_file.exists():
                        shutil.copy(label_file, lbl_dest)

        # Print summary
        print("\nSUMMARY:")
        for cls in target_classes:
            print(f"{cls}: {len(all_images_by_class[cls])} images (split into train/valid/test)")

        dataset_path = dest_folder
        print("\nSubset creation complete!")

    return dataset_path














def test(classes_reduced_dataset=None, max_images_per_class=None, images_original=original_image_folder, seed_value=42):
    random.seed(seed_value)

    # Load YAML
    yaml_file = next(images_original.glob("*.yaml"), None) or next(images_original.glob("*.yml"), None)
    if not yaml_file:
        raise FileNotFoundError("No YAML file found in agropest12 folder")

    with open(yaml_file, 'r') as f:
        config = yaml.safe_load(f)

    # Class names in lowercase
    names = [name.lower() for name in config['names']]
    #print("Classes:", names)
    print(names)
    # If user provided a reduced class list, filter it
    if classes_reduced:
        classes_reduced = [c.lower() for c in classes_reduced]
        target_classes = [c for c in names if c in classes_reduced]
    else:
        target_classes = names

    # Paths for splits
    splits = {
        "train": images_original / "train" / "images",
        "valid": images_original / "valid" / "images",
        "test": images_original / "test" / "images"
    }

    # Known variations or typos
    class_variations = {
        "caterpillars": ["caterpillar", "catterpillar", "catterpillars", "caterpillars"],
        "beetles": ["beetle", "beetles"],
        "ants": ["ant", "ants"],
        "bees": ["bee", "bees"],
        "earwigs": ["earwigs", "earwig"],
        "grasshoppers": ["grasshopper", "grasshoppers"],
        "moths": ["moths", "moth"],
        "slugs": ["slug", "slugs"],
        "snails": ["snail", "snails"],
        "wasps": ["wasp", "wasps"],
        "weevils": ["weevil", "weevils"]
    }

    for split_name, split_path in splits.items():
        if not split_path.exists():
            print(f"{split_name}: No images folder found.")
            continue

        # Collect images by class
        images_by_class = {cls: [] for cls in target_classes}

        for img_file in split_path.glob("*.jpg"):
            filename_lower = img_file.stem.lower()
            for class_name in sorted(target_classes, key=len, reverse=True):
                variations = sorted(class_variations.get(class_name, [class_name]), key=len, reverse=True)
                if any(var in filename_lower for var in variations):
                    images_by_class[class_name].append(img_file)
                    break

        # Apply max_images_per_class if specified
        if max_images_per_class:
            for cls in images_by_class:
                if len(images_by_class[cls]) > max_images_per_class:
                    images_by_class[cls] = random.sample(images_by_class[cls], max_images_per_class)

        # Print summary
        print(f"\n{split_name.upper()} SUMMARY:")
        for cls in target_classes:
            print(f"  {cls}: {len(images_by_class[cls])} images")

        # Optional: return or copy selected images
        return images_by_class








def extract_reduced_dataset(number_images_for_train, classes_reduced, images_original=original_image_folder, seed_value=42):
    
    Creates a reduced dataset with train/valid/test splits:
    - First split: train vs test
    - Second split: train -> train + valid
    
    random.seed(seed_value)

    reduced_image_folder = images_original.parent / f"{images_original.stem}_reduced_{number_images_for_train}_images"

    if not reduced_image_folder.exists():
        for split in ["train", "valid", "test"]:
            for class_name in classes_reduced:
                (reduced_image_folder / split / 'images' / class_name).mkdir(parents=True, exist_ok=True)

        for class_name in classes_reduced:
            print(f"\nProcessing class: {class_name}")

            # Collect all images for this class from original train folder
            source_folder = images_original / "train" / "images"
            all_images = [
                img for img in source_folder.glob("*")
                if img.is_file() and img.name.startswith(class_name) and img.suffix.lower() in [".jpg", ".jpeg", ".png"]
            ]

            if not all_images:
                print(f"No images found for class '{class_name}'")
                continue

            # Limit to requested number for training
            selected_images = random.sample(all_images, min(number_images_for_train, len(all_images)))

            # First split: train vs test (80/20)
            train_imgs, test_imgs = train_test_split(
                selected_images,
                test_size=0.2,
                random_state=seed_value
            )

            # Second split: train -> train + valid (20% of train for valid)
            train_imgs, valid_imgs = train_test_split(
                train_imgs,
                test_size=0.2,
                random_state=seed_value
            )

            # Copy train images
            for img in train_imgs:
                shutil.copy2(img, reduced_image_folder / "train" / "images" / class_name / img.name)

            # Copy valid images
            for img in valid_imgs:
                shutil.copy2(img, reduced_image_folder / "valid" / "images" / class_name / img.name)

            # Copy test images
            for img in test_imgs:
                shutil.copy2(img, reduced_image_folder / "test" / "images" / class_name / img.name)

            print(f"Copied {len(train_imgs)} train, {len(valid_imgs)} valid, {len(test_imgs)} test images for '{class_name}'")
    else:
        print(f"Folder already exists: {reduced_image_folder.name}")
        reduced_folders = [p for p in images_original.parent.iterdir() if p.is_dir() and "reduced" in p.name]
        print('Available datasets:')
        for folder in reduced_folders:
            print(folder.name)

    return reduced_image_folder



"""