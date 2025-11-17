import random
import shutil
from pathlib import Path
from PIL import Image



original_image_folder = Path(r"C:\Users\brumor\Onedrive - Statens Kartverk\PhD\courses\fys-stk4155\projects\project3\datasets\agropest12")


from sklearn.model_selection import train_test_split
import shutil
import random
from pathlib import Path

from sklearn.model_selection import train_test_split
import shutil
import random
from pathlib import Path

def extract_reduced_dataset(number_images_for_train, classes_reduced, images_original=original_image_folder, seed_value=42):
    """
    Creates a reduced dataset with train/valid/test splits:
    - First split: train vs test
    - Second split: train -> train + valid
    """
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


def check_images_size_equal(dataset):
    # Check if all images have same size
    sizes = {}

    for img_path in dataset.rglob("*"):
        if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            with Image.open(img_path) as img:
                size = img.size  # (width, height)
                sizes.setdefault(size, []).append(str(img_path))

    if len(sizes) == 1:
        print(f"All images have the same size: {list(sizes.keys())[0]}")
    else:
        print("Images have different sizes:")
        for size, paths in sizes.items():
            print(f"Size {size}: {len(paths)} images")