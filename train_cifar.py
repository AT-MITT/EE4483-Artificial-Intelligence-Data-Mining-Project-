print("Script starts")


# ==============================================================
# MULTI-CATEGORY CLASSIFICATION ON CIFAR-10 DATASET (VGG19)
# ==============================================================

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torchvision import datasets, transforms, models
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
USER = os.getenv("USER", "karthik3")
SCRATCH_DIR = f"/home/users/ntu/karthik3/scratch"
OUTPUT_DIR = f"/home/users/ntu/karthik3/scratch/AIDM_Project"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==== CONFIGURATION ====
NUM_EPOCHS = 100
LR = 1e-4
IMG_SIZE = 224   
NUM_CLASSES = 2

# ==== CIFAR-10 CONFIGURATION ====
CIFAR_BATCH_SIZE = 128
CIFAR_NUM_CLASSES = 10

# ==== DATA AUGMENTATION AND NORMALIZATION ====
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ==== LOAD CIFAR-10 DATA ====
cifar_full_train = datasets.CIFAR10(root="./data", train=True, download=True, transform=train_transform)

# Split 80-20 train-validation
train_size = int(0.8 * len(cifar_full_train))
val_size = len(cifar_full_train) - train_size
cifar_train, cifar_val = random_split(cifar_full_train, [train_size, val_size])

train_loader_cifar = DataLoader(cifar_train, batch_size=CIFAR_BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader_cifar = DataLoader(cifar_val, batch_size=CIFAR_BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

print(f"\nCIFAR-10 Dataset Loaded:")
print(f"Training samples: {len(cifar_train)}, Validation samples: {len(cifar_val)}")

# ==============================================================
# MODEL SETUP (VGG19 with Dropout) - FIXED
# ==============================================================
model_cifar = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)

# Freeze feature extractor
for param in model_cifar.features.parameters():
    param.requires_grad = False

# FIX: Correct classifier modification for VGG19
# VGG19 original classifier: [Linear(25088, 4096), ReLU, Dropout, Linear(4096, 4096), ReLU, Dropout, Linear(4096, 1000)]
in_features = model_cifar.classifier[0].in_features  # This should be 25088

# Replace just the final layer, keep the rest of the classifier intact
model_cifar.classifier[6] = nn.Linear(4096, CIFAR_NUM_CLASSES)

# Alternative: If you want to modify the entire classifier, use this structure:
# model_cifar.classifier = nn.Sequential(
#     nn.Linear(in_features, 4096),  # 25088 -> 4096
#     nn.ReLU(inplace=True),
#     nn.Dropout(p=0.5),
#     nn.Linear(4096, 4096),
#     nn.ReLU(inplace=True),
#     nn.Dropout(p=0.5),
#     nn.Linear(4096, CIFAR_NUM_CLASSES)
# )

model_cifar = model_cifar.to(DEVICE)

print(f"Model setup complete:")
print(f"Input features to classifier: {in_features}")
print(f"Classifier structure: {model_cifar.classifier}")

# ==============================================================
# LOSS, OPTIMIZER, LR SCHEDULER
# ==============================================================
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
# Train only the classifier parameters (features are frozen)
optimizer = optim.Adam(model_cifar.classifier.parameters(), lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

# ==============================================================
# TRAINING LOOP
# ==============================================================
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []
learning_rates = []
best_val_acc = 0.0
best_model_path = os.path.join(OUTPUT_DIR, "best_cifar10_vgg19.pth")

print("\nStarting CIFAR-10 training with VGG19...")
start_time = time.time()

for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}]")
    current_lr = optimizer.param_groups[0]['lr']
    learning_rates.append(current_lr)
    print(f"Learning Rate: {current_lr:.6f}")

    # ---- Training Phase ----
    model_cifar.train()
    running_loss, correct, total = 0.0, 0, 0
    loop = tqdm(train_loader_cifar, desc="Training", leave=False)

    for imgs, labels in loop:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model_cifar(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels).item()
        total += labels.size(0)
        loop.set_postfix(loss=loss.item())

    train_loss = running_loss / total
    train_acc = correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    # ---- Validation Phase ----
    model_cifar.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader_cifar:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model_cifar(imgs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            val_correct += torch.sum(preds == labels).item()
            val_total += labels.size(0)

    val_loss /= val_total
    val_acc = val_correct / val_total
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)
    scheduler.step(val_loss)

    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model_cifar.state_dict(), best_model_path)
        print(f"Saved best CIFAR-10 model (Val Acc: {best_val_acc:.4f})")

# ==============================================================
# TRAINING COMPLETE
# ==============================================================
total_time = (time.time() - start_time) / 60
print(f"\nCIFAR-10 Training Complete in {total_time:.1f} minutes")
print(f"Best Validation Accuracy: {best_val_acc:.4f}")

# ==== PLOTTING RESULTS ====
plt.figure(figsize=(16, 10))
epochs = range(1, len(train_losses) + 1)

# (1) Loss vs Epoch
plt.subplot(2, 2, 1)
plt.plot(epochs, train_losses, label="Training Loss", color='blue')
plt.plot(epochs, val_losses, label="Validation Loss", color='orange')
plt.title("Training vs Validation Loss (VGG19)", fontsize=14, fontweight='bold')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, alpha=0.3)

# (2) Accuracy vs Epoch
plt.subplot(2, 2, 2)
plt.plot(epochs, train_accuracies, label="Training Accuracy", color='green')
plt.plot(epochs, val_accuracies, label="Validation Accuracy", color='red')
plt.title("Training vs Validation Accuracy (VGG19)", fontsize=14, fontweight='bold')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True, alpha=0.3)

# (3) Learning Rate Schedule
plt.subplot(2, 2, 3)
plt.plot(epochs, learning_rates, label="Learning Rate", color='purple')
plt.yscale("log")
plt.title("Learning Rate Schedule", fontsize=14, fontweight='bold')
plt.xlabel("Epoch")
plt.ylabel("Learning Rate (log scale)")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, "vgg19_cifar10_training_results.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"CIFAR-10 training performance plots saved to: {plot_path}")