import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.utils.data import random_split
from torch.utils.data import WeightedRandomSampler
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

# ==== GPU CHECK ====
if not torch.cuda.is_available():
    print("FATAL: No GPU detected! This code requires a CUDA-enabled GPU.")
    exit(1)

DEVICE = torch.device("cuda")
print(f"Using GPU: {torch.cuda.get_device_name(0)}")

# ==== DIRECTORIES ====
USER = os.getenv("USER", "user")
SCRATCH_DIR = f"/home/users/ntu/atulya00/scratch"
OUTPUT_DIR = f"/home/users/ntu/atulya00/AIDM"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==== CONFIGURATION ====
BATCH_SIZE = 64
NUM_EPOCHS = 100
LR = 1e-4
IMG_SIZE = 299
NUM_CLASSES = 2

# ==== DATASETS ====
train_dir = os.path.join(SCRATCH_DIR, "train/preprocessed")
val_dir = os.path.join(SCRATCH_DIR, "val")

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_data = datasets.ImageFolder(train_dir, transform=transform)
val_data = datasets.ImageFolder(val_dir, transform=transform)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

print(f"Training samples: {len(train_data)}, Validation samples: {len(val_data)}")

# ==== MODEL SETUP ====
model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
for param in model.parameters():
    param.requires_grad = False

in_features = model.fc.in_features
model.fc = nn.Linear(in_features, NUM_CLASSES)
model = model.to(DEVICE)

# ==== TRAINING CONFIGURATION ====
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

# ==== TRAINING ====
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []
learning_rates = []

best_val_acc = 0.0
best_model_path = os.path.join(OUTPUT_DIR, "best_inceptionV3.pth")

start_time = time.time()
print("Starting training...")

for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}]")
    current_lr = optimizer.param_groups[0]['lr']
    learning_rates.append(current_lr)
    print(f"Learning Rate: {current_lr:.6f}")

    # ---- Training ----
    model.train()
    model.aux_logits = True
    running_loss, correct, total = 0.0, 0, 0
    loop = tqdm(train_loader, desc="Training", leave=False)

    for imgs, labels in loop:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)

        # Handle InceptionV3 dual outputs
        if hasattr(outputs, "aux_logits"):
            loss1 = criterion(outputs.logits, labels)
            loss2 = criterion(outputs.aux_logits, labels)
            loss = loss1 + 0.4 * loss2
            logits = outputs.logits
        else:
            loss = criterion(outputs, labels)
            logits = outputs

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        _, preds = torch.max(logits, 1)
        correct += torch.sum(preds == labels).item()
        total += labels.size(0)
        loop.set_postfix(loss=loss.item())


    train_loss = running_loss / total
    train_acc = correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    # ---- Validation ----
    model.eval()
    model.aux_logits = False
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            if hasattr(outputs, "logits"):
                outputs = outputs.logits
            loss = criterion(outputs, labels)

            val_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            val_correct += torch.sum(preds == labels).item()
            val_total += labels.size(0)

    val_loss /= val_total
    val_acc = val_correct / val_total
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)
    print("Current LR:", scheduler.get_last_lr())
    scheduler.step(val_loss)

    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), best_model_path)
        print(f"Saved best model (Val Acc: {best_val_acc:.4f})")

# ==== TRAINING COMPLETE ====
total_time = (time.time() - start_time) / 60
print(f"\nTraining Complete in {total_time:.1f} minutes")
print(f"Best Validation Accuracy: {best_val_acc:.4f}")

# ==== PLOTTING ====
epochs = range(1, len(train_losses) + 1)
plt.figure(figsize=(16, 10))

# ---- (1) Training and Validation Loss vs Epoch ----
plt.subplot(2, 2, 1)
plt.plot(epochs, train_losses, label="Training Loss", color='blue', linewidth=2)
plt.plot(epochs, val_losses, label="Validation Loss", color='orange', linewidth=2)
plt.title("Training vs Validation Loss", fontsize=14, fontweight='bold')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, alpha=0.3)

# ---- (2) Training and Validation Accuracy vs Epoch ----
plt.subplot(2, 2, 2)
plt.plot(epochs, train_accuracies, label="Training Accuracy", color='green', linewidth=2)
plt.plot(epochs, val_accuracies, label="Validation Accuracy", color='red', linewidth=2)
plt.title("Training vs Validation Accuracy", fontsize=14, fontweight='bold')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True, alpha=0.3)

# ---- (3) Learning Rate vs Epoch ----
plt.subplot(2, 2, 3)
plt.plot(epochs, learning_rates, label="Learning Rate", color='purple', linewidth=2)
plt.yscale("log")
plt.title("Learning Rate Schedule", fontsize=14, fontweight='bold')
plt.xlabel("Epoch")
plt.ylabel("Learning Rate (log scale)")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, "inception_v3_training_results.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"Training performance plots saved to: {plot_path}")