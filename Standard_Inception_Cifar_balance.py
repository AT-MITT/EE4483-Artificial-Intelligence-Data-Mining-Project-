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
NUM_EPOCHS = 100
LR = 1e-4
IMG_SIZE = 299

# ==== CIFAR-10 CONFIGURATION ====
CIFAR_BATCH_SIZE = 128
CIFAR_NUM_CLASSES = 10
# ==== DATA AUGMENTATION AND NORMALIZATION ====
train_transform = transforms.Compose([
    transforms.AutoAugment(),
    transforms.Resize(180),
    transforms.CenterCrop(IMG_SIZE),
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

# ---- Simulate Class Imbalance (reduce samples for first 3 classes) ----
np.random.seed(42)
targets = np.array(cifar_full_train.targets)
class_counts = np.bincount(targets)
print("\nOriginal Class Distribution:", class_counts)

reduction_factor = {0: 0.2, 1: 0.2, 2: 0.2}  # Keep only 20% of samples for first 3 classes
new_indices = []
for idx, label in enumerate(targets):
    if label in reduction_factor:
        if np.random.rand() < reduction_factor[label]:
            new_indices.append(idx)
    else:
        new_indices.append(idx)

imbalanced_dataset = torch.utils.data.Subset(cifar_full_train, new_indices)
imbalanced_targets = targets[new_indices]
imbalanced_counts = np.bincount(imbalanced_targets)
print("\nImbalanced Class Distribution:", imbalanced_counts)

# Split 80/20 train-validation
train_size = int(0.8 * len(imbalanced_dataset))
val_size = len(imbalanced_dataset) - train_size

cifar_train, cifar_val = random_split(imbalanced_dataset, [train_size, val_size])

print(f"\nCIFAR-10 Dataset Loaded:")
print(f"Training samples: {len(cifar_train)}, Validation samples: {len(cifar_val)}")

# Compute class weights for sampling
class_weights = 1.0 / np.maximum(imbalanced_counts, 1)

# Map weights
train_indices = cifar_train.indices
train_subset_targets = targets[train_indices]
train_sample_weights = class_weights[train_subset_targets]

train_sampler = WeightedRandomSampler(weights=train_sample_weights, num_samples=len(train_sample_weights), replacement=True)

# DataLoaders
train_loader_balanced = DataLoader(cifar_train, batch_size=CIFAR_BATCH_SIZE, sampler=train_sampler, num_workers=4, pin_memory=True)

# Validation loader
val_loader_cifar = DataLoader(cifar_val, batch_size=CIFAR_BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# ---- Model Setup ----
model_imbalanced = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)

# ---- Freeze all layers ----
for param in model_imbalanced.parameters():
    param.requires_grad = False

# Unfreeze last Inception blocks
for name, param in model_imbalanced.named_parameters():
    if "Mixed_7b" in name or "Mixed_7c" in name or "fc" in name:
        param.requires_grad = True

# ---- Replace final fully connected (fc) layer ----
in_features = model_imbalanced.fc.in_features
model_imbalanced.fc = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(in_features, 1024),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(1024, CIFAR_NUM_CLASSES)
)

# ---- Replace auxiliary head FC layer ----
aux_in_features = model_imbalanced.AuxLogits.fc.in_features
model_imbalanced.AuxLogits.fc = nn.Linear(aux_in_features, CIFAR_NUM_CLASSES)

model_imbalanced = model_imbalanced.to(DEVICE)

# ---- Weighted loss with Label Smoothing ----
class_weights = torch.tensor(1.0 / np.maximum(class_counts, 1), dtype=torch.float32).to(DEVICE)
criterion_weighted = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
optimizer_imbalanced = optim.Adam(filter(lambda p: p.requires_grad, model_imbalanced.parameters()), lr=LR, weight_decay=1e-4)
scheduler_imbalanced = optim.lr_scheduler.ReduceLROnPlateau(optimizer_imbalanced, mode='min', patience=5, factor=0.5)


# ---- Training loop (same format) ----
print("\nStarting training on imbalanced CIFAR-10 (with rebalancing techniques)...")

train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []
learning_rates = []

best_val_acc = 0.0
best_model_path = os.path.join(OUTPUT_DIR, "best_cifar10_inceptionV3_balanced.pth")
start_time = time.time()

for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}]")
    current_lr = optimizer_imbalanced.param_groups[0]['lr']
    learning_rates.append(current_lr)
    print(f"Learning Rate: {current_lr:.6f}")

    # ---- Training ----
    model_imbalanced.train()
    model_imbalanced.aux_logits = True
    running_loss, correct, total = 0.0, 0, 0
    loop = tqdm(train_loader_balanced, desc="Training (Balanced)", leave=False)

    for imgs, labels in loop:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer_imbalanced.zero_grad()
        outputs = model_imbalanced(imgs)
        # Handle InceptionV3 dual outputs
        if hasattr(outputs, "aux_logits"):
            loss1 = criterion_weighted(outputs.logits, labels)
            loss2 = criterion_weighted(outputs.aux_logits, labels)
            loss = loss1 + 0.4 * loss2
            logits = outputs.logits
        else:
            loss = criterion_weighted(outputs, labels)
            logits = outputs
        loss.backward()
        optimizer_imbalanced.step()

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
    model_imbalanced.eval()
    model_imbalanced.aux_logits = False
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader_cifar:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model_imbalanced(imgs)
            if hasattr(outputs, "logits"):
                outputs = outputs.logits
            loss = criterion_weighted(outputs, labels)
            val_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            val_correct += torch.sum(preds == labels).item()
            val_total += labels.size(0)

    val_loss /= val_total
    val_acc = val_correct / val_total
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)
    scheduler_imbalanced.step(val_loss)

    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model_imbalanced.state_dict(), best_model_path)
        print(f"Saved best balanced CIFAR-10 model (Val Acc: {best_val_acc:.4f})")

total_time = (time.time() - start_time) / 60
print(f"\nBalanced CIFAR-10 Training Complete in {total_time:.1f} minutes")
print(f"Best Validation Accuracy: {best_val_acc:.4f}")

# ==== PLOTTING RESULTS ====
plt.figure(figsize=(16, 10))
epochs = range(1, len(train_losses) + 1)
# (1) Loss vs Epoch
plt.subplot(2, 2, 1)
plt.plot(epochs, train_losses, label="Training Loss", color='blue')
plt.plot(epochs, val_losses, label="Validation Loss", color='orange')
plt.title("Training vs Validation Loss (CIFAR-10)", fontsize=14, fontweight='bold')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, alpha=0.3)

# (2) Accuracy vs Epoch
plt.subplot(2, 2, 2)
plt.plot(epochs, train_accuracies, label="Training Accuracy", color='green')
plt.plot(epochs, val_accuracies, label="Validation Accuracy", color='red')
plt.title("Training vs Validation Accuracy (CIFAR-10)", fontsize=14, fontweight='bold')
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
plot_path = os.path.join(OUTPUT_DIR, "inceptionV3_cifar10_imbalance_training_results.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"CIFAR-10 training performance plots saved to: {plot_path}")
