import os
from torch.utils.data import DataLoader
from landmine_dataset_deeplab import LandmineDatasetDeeplab, get_transforms

# Configuration
TRAIN_IMAGE_DIR = "C:/Users/MarkoLillemets/Desktop/Landmine_Test/Combined/train/images"
TRAIN_MASK_DIR = "C:/Users/MarkoLillemets/Desktop/Landmine_Test/Combined/train/masks"
VAL_IMAGE_DIR = "C:/Users/MarkoLillemets/Desktop/Landmine_Test/Combined/val/images"
VAL_MASK_DIR = "C:/Users/MarkoLillemets/Desktop/Landmine_Test/Combined/val/masks"
BATCH_SIZE = 2  # Reduced from 4 to 2 to help with memory usage

# Use Albumentations transforms
transform = get_transforms()

# Load training and validation datasets
train_dataset = LandmineDatasetDeeplab(
    image_dir=TRAIN_IMAGE_DIR,
    mask_dir=TRAIN_MASK_DIR,
    transform=transform
)

val_dataset = LandmineDatasetDeeplab(
    image_dir=VAL_IMAGE_DIR,
    mask_dir=VAL_MASK_DIR,
    transform=transform
)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Train samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
