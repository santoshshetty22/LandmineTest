import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from dataloader import train_loader, val_loader

# Configuration
NUM_CLASSES = 8  # Update according to your label_map.txt
EPOCHS = 30
LR = 1e-4
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Model
weights = DeepLabV3_ResNet50_Weights.DEFAULT
model = deeplabv3_resnet50(weights=weights)
model.classifier[4] = nn.Conv2d(256, NUM_CLASSES, kernel_size=(1, 1), stride=(1, 1))
model.to(device)

# Loss, optimizer, and scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# Metric calculations
def compute_metrics(outputs, masks):
    preds = torch.argmax(outputs, dim=1)
    correct = (preds == masks).float()
    accuracy = correct.sum() / correct.numel()

    intersection = ((preds == masks) & (masks > 0)).float().sum()
    union = ((preds > 0) | (masks > 0)).float().sum()
    iou = intersection / (union + 1e-6)

    return accuracy.item(), iou.item()

# Training and validation
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_accuracy = 0.0
    running_iou = 0.0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.long().to(device)

        optimizer.zero_grad()
        outputs = model(images)['out']
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        acc, iou = compute_metrics(outputs, masks)
        running_loss += loss.item() * images.size(0)
        running_accuracy += acc * images.size(0)
        running_iou += iou * images.size(0)

    n = len(loader.dataset)
    return running_loss / n, running_accuracy / n, running_iou / n

def validate(model, loader, criterion, device):
    model.eval()
    val_loss = 0.0
    val_accuracy = 0.0
    val_iou = 0.0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)

            # Make sure mask shape is [B, H, W]
            if masks.ndim == 4 and masks.shape[1] == 1:
                masks = masks.squeeze(1)

            masks = masks.long().to(device)

            outputs = model(images)['out']
            loss = criterion(outputs, masks)

            acc, iou = compute_metrics(outputs, masks)
            val_loss += loss.item() * images.size(0)
            val_accuracy += acc * images.size(0)
            val_iou += iou * images.size(0)

    n = len(loader.dataset)
    return val_loss / n, val_accuracy / n, val_iou / n

# Main training loop
best_val_loss = float('inf')
for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc, train_iou = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc, val_iou = validate(model, val_loader, criterion, device)
    print(f"Epoch {epoch}/{EPOCHS}")
    print(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, IoU: {train_iou:.4f}")
    print(f"  Val   Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, IoU: {val_iou:.4f}")

    scheduler.step()

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        checkpoint_path = os.path.join(CHECKPOINT_DIR, "deeplabv3_best.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved best model to {checkpoint_path}")

# Save final model
final_path = os.path.join(CHECKPOINT_DIR, "deeplabv3_final.pth")
torch.save(model.state_dict(), final_path)
print(f"Training complete. Final model saved to {final_path}")
