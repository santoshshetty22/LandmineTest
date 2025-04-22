import os
import shutil
from PIL import Image
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Configuration
DATA_ROOT = r"C:/Users/MarkoLillemets/Desktop/Landmine_Test"
TERRAINS = ["Concrete", "GrassField", "WetGravel"]
OUTPUT_DIR = os.path.join(DATA_ROOT, "Combined")
IMG_SIZE = (640, 640)
VAL_SPLIT = 0.2

# Utility to create folders
def make_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Merge images and masks into a single folder with terrain prefix
def collect_and_split_data():
    all_images = []
    all_masks = []
    all_labels = []

    for terrain in TERRAINS:
        terrain_path = os.path.join(DATA_ROOT, terrain)
        img_dir = os.path.join(terrain_path, "images")
        mask_dir = os.path.join(terrain_path, "SegmentationClass")

        with open(os.path.join(terrain_path, "ImageSets", "Segmentation", "default.txt")) as f:
            file_list = f.read().splitlines()

        for fname in file_list:
            image_path = os.path.join(img_dir, fname + ".jpg")
            mask_path = os.path.join(mask_dir, fname + ".png")
            if os.path.exists(image_path) and os.path.exists(mask_path):
                all_images.append(image_path)
                all_masks.append(mask_path)
                all_labels.append(terrain)

    # Split
    train_imgs, val_imgs, train_masks, val_masks = train_test_split(
        all_images, all_masks, test_size=VAL_SPLIT, random_state=42
    )

    # Create directories
    for split in ["train", "val"]:
        for subfolder in ["images", "masks"]:
            make_dirs(os.path.join(OUTPUT_DIR, split, subfolder))

    # Save
    def save_pairs(img_list, mask_list, split):
        for img_path, mask_path in zip(img_list, mask_list):
            base_name = os.path.basename(img_path)
            mask_name = os.path.basename(mask_path)

            img = Image.open(img_path).convert("RGB").resize(IMG_SIZE)
            mask = Image.open(mask_path).resize(IMG_SIZE)

            img.save(os.path.join(OUTPUT_DIR, split, "images", base_name))
            mask.save(os.path.join(OUTPUT_DIR, split, "masks", mask_name))

    save_pairs(train_imgs, train_masks, "train")
    save_pairs(val_imgs, val_masks, "val")

    print("Dataset preparation completed. Train/Val split with resized images/masks saved in:", OUTPUT_DIR)

# Dataset class for DeepLabV3
class LandmineDatasetDeeplab(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_paths = sorted([os.path.join(image_dir, fname) for fname in os.listdir(image_dir)])
        self.mask_paths = sorted([os.path.join(mask_dir, fname) for fname in os.listdir(mask_dir)])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path))

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        if isinstance(mask, torch.Tensor):
            mask = mask.squeeze()  # from [1, H, W] to [H, W]
            mask = mask.long()

        return image, mask

def get_transforms():
    return A.Compose([
        A.Resize(640, 640),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

if __name__ == '__main__':
    collect_and_split_data()
