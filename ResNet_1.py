"""
============================================================
Cats vs Dogs Classification using ResNet50 (PyTorch)
============================================================

Model: ResNet50 (pre-trained on ImageNet)
Input Size: 224 x 224 RGB images
Output: 2 classes (Cats, Dogs)

Loss Function: CrossEntropyLoss
Optimizer: Adam (learning rate scheduler: ReduceLROnPlateau)
Training Strategy:
  - Freeze all convolutional layers (feature extractor).
  - Train only the final fully connected classification layer.
  - Use preprocessed and pre-split datasets.
  - Save only the best model (based on validation accuracy).
  - Record and plot all metrics to check for overfitting.
============================================================
"""

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
SCRATCH_DIR = f"/home/users/ntu/po0001yi/scratch"
OUTPUT_DIR = f"/home/users/ntu/po0001yi/scratch/AIDM"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==== CONFIGURATION ====
BATCH_SIZE = 32
NUM_EPOCHS = 1
LR = 1e-4
IMG_SIZE = 224
NUM_CLASSES = 2

# ==== DATASETS ====
train_dir = '/home/users/ntu/po0001yi/scratch/preprocessed'
val_dir = '/home/users/ntu/po0001yi/scratch/datasets/val'

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
# Load pre-trained ResNet50 and replace the final fully connected layer
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Replace classifier (only final layer will be trainable)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, NUM_CLASSES)
model = model.to(DEVICE)

# ==== TRAINING CONFIGURATION ====
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=True)

# ==== TRAINING LOOP ====
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []
learning_rates = []

best_val_acc = 0.0
best_model_path = os.path.join(OUTPUT_DIR, "best_resnet50.pth")

start_time = time.time()
print("Starting training...")

for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}]")
    current_lr = optimizer.param_groups[0]['lr']
    learning_rates.append(current_lr)
    print(f"Learning Rate: {current_lr:.6f}")

    # ---- Training ----
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    loop = tqdm(train_loader, desc="Training", leave=False)

    for imgs, labels in loop:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
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

    # ---- Validation ----
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
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
plt.figure(figsize=(16, 10))

# ---- (1) Training and Validation Loss vs Epoch ----
plt.subplot(2, 2, 1)
plt.plot(range(1, NUM_EPOCHS + 1), train_losses, label="Training Loss", color='blue', linewidth=2)
plt.plot(range(1, NUM_EPOCHS + 1), val_losses, label="Validation Loss", color='orange', linewidth=2)
plt.title("Training vs Validation Loss", fontsize=14, fontweight='bold')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, alpha=0.3)

# ---- (2) Training and Validation Accuracy vs Epoch ----
plt.subplot(2, 2, 2)
plt.plot(range(1, NUM_EPOCHS + 1), train_accuracies, label="Training Accuracy", color='green', linewidth=2)
plt.plot(range(1, NUM_EPOCHS + 1), val_accuracies, label="Validation Accuracy", color='red', linewidth=2)
plt.title("Training vs Validation Accuracy", fontsize=14, fontweight='bold')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True, alpha=0.3)

# ---- (3) Learning Rate vs Epoch ----
plt.subplot(2, 2, 3)
plt.plot(range(1, NUM_EPOCHS + 1), learning_rates, label="Learning Rate", color='purple', linewidth=2)
plt.yscale("log")
plt.title("Learning Rate Schedule", fontsize=14, fontweight='bold')
plt.xlabel("Epoch")
plt.ylabel("Learning Rate (log scale)")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, "resnet50_training_results.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"Training performance plots saved to: {plot_path}")


# ==============================================================
# MULTI-CATEGORY CLASSIFICATION ON CIFAR-10 DATASET
# ==============================================================

"""
Dataset Description:
--------------------
The CIFAR-10 dataset contains 60,000 color images of size 32x32 pixels 
belonging to 10 categories:
['airplane', 'automobile', 'bird', 'cat', 'deer', 
 'dog', 'frog', 'horse', 'ship', 'truck'].

Each class has 6,000 images, split into 50,000 for training and 10,000 for testing.
We will further split the training set into 80% training and 20% validation 
to monitor model generalization.

Classification Task:
--------------------
We are adapting the ResNet50 model (pre-trained on ImageNet) for multi-class 
classification of CIFAR-10. This involves:
1. Resizing small (32x32) images to 224x224 for ResNet input.
2. Modifying the final layer to output 10 classes.
3. Using data augmentation to improve robustness.
4. Using learning rate scheduling to handle plateauing loss.

Compared to the Dogs vs Cats problem:
-------------------------------------
- Binary classification → Multi-class classification (10 categories)
- High-resolution → Low-resolution images (resized)
- Simpler dataset → More diverse visual content
- Different accuracy metrics (multi-class accuracy)
"""

# ==== CIFAR-10 CONFIGURATION ====
CIFAR_BATCH_SIZE = 128
CIFAR_NUM_CLASSES = 10

# ==== DATA AUGMENTATION AND NORMALIZATION ====
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
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
print(f"Training samples: {len(cifar_train)}, Validation samples: {len(cifar_val)},")

# ==== MODEL SETUP ====
model_cifar = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

for name, param in model_cifar.named_parameters():
    if "fc" not in name:
        param.requires_grad = False


# Modify final classifier for CIFAR-10
in_features = model_cifar.fc.in_features
model_cifar.fc = nn.Linear(in_features, CIFAR_NUM_CLASSES)
model_cifar = model_cifar.to(DEVICE)

# ==== LOSS, OPTIMIZER, LR SCHEDULER ====
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_cifar.fc.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=True)

# ==== TRAINING LOOP (same style as original code) ====
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []
learning_rates = []

best_val_acc = 0.0
best_model_path = os.path.join(OUTPUT_DIR, "best_cifar10_resnet50.pth")

print("\nStarting CIFAR-10 training...")
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

# ==== TRAINING COMPLETE ====
total_time = (time.time() - start_time) / 60
print(f"\nCIFAR-10 Training Complete in {total_time:.1f} minutes")
print(f"Best Validation Accuracy: {best_val_acc:.4f}")

# ==== PLOTTING RESULTS ====
plt.figure(figsize=(16, 10))

# (1) Loss vs Epoch
plt.subplot(2, 2, 1)
plt.plot(range(1, NUM_EPOCHS + 1), train_losses, label="Training Loss", color='blue')
plt.plot(range(1, NUM_EPOCHS + 1), val_losses, label="Validation Loss", color='orange')
plt.title("Training vs Validation Loss (CIFAR-10)", fontsize=14, fontweight='bold')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, alpha=0.3)

# (2) Accuracy vs Epoch
plt.subplot(2, 2, 2)
plt.plot(range(1, NUM_EPOCHS + 1), train_accuracies, label="Training Accuracy", color='green')
plt.plot(range(1, NUM_EPOCHS + 1), val_accuracies, label="Validation Accuracy", color='red')
plt.title("Training vs Validation Accuracy (CIFAR-10)", fontsize=14, fontweight='bold')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True, alpha=0.3)

# (3) Learning Rate Schedule
plt.subplot(2, 2, 3)
plt.plot(range(1, NUM_EPOCHS + 1), learning_rates, label="Learning Rate", color='purple')
plt.yscale("log")
plt.title("Learning Rate Schedule", fontsize=14, fontweight='bold')
plt.xlabel("Epoch")
plt.ylabel("Learning Rate (log scale)")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, "resnet50_cifar10_training_results.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"CIFAR-10 training performance plots saved to: {plot_path}")

# ==============================================================
# IMPROVING CLASSIFIER UNDER DATA IMBALANCE CONDITIONS
# ==============================================================

"""
Data Imbalance Handling:
------------------------
To simulate real-world imbalance, we reduce the number of samples in 
some CIFAR-10 classes and retrain the model using:

1. WeightedRandomSampler:
   Ensures minority classes are sampled more frequently per batch.

2. Class-Weighted CrossEntropyLoss:
   Penalizes misclassification of minority classes more strongly.

Together, these methods balance the training signal across all categories.
"""

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
print("Imbalanced Class Distribution:", imbalanced_counts)

# ---- Compute sampling weights ----
weights = 1.0 / np.maximum(imbalanced_counts, 1)
sample_weights = weights[imbalanced_targets]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

# ---- Create balanced DataLoader ----
train_loader_balanced = DataLoader(imbalanced_dataset, batch_size=CIFAR_BATCH_SIZE,
                                   sampler=sampler, num_workers=4, pin_memory=True)

# ---- Reinitialize model ----
model_imbalanced = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
for name, param in model_cifar.named_parameters():
    if "fc" not in name:
        param.requires_grad = False

in_features = model_imbalanced.fc.in_features
model_imbalanced.fc = nn.Linear(in_features, CIFAR_NUM_CLASSES)
model_imbalanced = model_imbalanced.to(DEVICE)

# ---- Weighted loss ----
class_weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)
criterion_weighted = nn.CrossEntropyLoss(weight=class_weights)
optimizer_imbalanced = optim.Adam(model_imbalanced.fc.parameters(), lr=LR)
scheduler_imbalanced = optim.lr_scheduler.ReduceLROnPlateau(optimizer_imbalanced, mode='min', patience=5, factor=0.5, verbose=True)

# ---- Training loop (same format) ----
print("\nStarting training on imbalanced CIFAR-10 (with rebalancing techniques)...")

train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []
learning_rates = []

best_val_acc = 0.0
best_model_path = os.path.join(OUTPUT_DIR, "best_cifar10_resnet50_balanced.pth")
start_time = time.time()

for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}]")
    current_lr = optimizer_imbalanced.param_groups[0]['lr']
    learning_rates.append(current_lr)
    print(f"Learning Rate: {current_lr:.6f}")

    # ---- Training ----
    model_imbalanced.train()
    running_loss, correct, total = 0.0, 0, 0
    loop = tqdm(train_loader_balanced, desc="Training (Balanced)", leave=False)

    for imgs, labels in loop:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer_imbalanced.zero_grad()
        outputs = model_imbalanced(imgs)
        loss = criterion_weighted(outputs, labels)
        loss.backward()
        optimizer_imbalanced.step()

        running_loss += loss.item() * imgs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels).item()
        total += labels.size(0)
        loop.set_postfix(loss=loss.item())

    train_loss = running_loss / total
    train_acc = correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    # ---- Validation ----
    model_imbalanced.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader_cifar:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model_imbalanced(imgs)
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

# (1) Loss vs Epoch
plt.subplot(2, 2, 1)
plt.plot(range(1, NUM_EPOCHS + 1), train_losses, label="Training Loss", color='blue')
plt.plot(range(1, NUM_EPOCHS + 1), val_losses, label="Validation Loss", color='orange')
plt.title("Training vs Validation Loss (CIFAR-10)", fontsize=14, fontweight='bold')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, alpha=0.3)

# (2) Accuracy vs Epoch
plt.subplot(2, 2, 2)
plt.plot(range(1, NUM_EPOCHS + 1), train_accuracies, label="Training Accuracy", color='green')
plt.plot(range(1, NUM_EPOCHS + 1), val_accuracies, label="Validation Accuracy", color='red')
plt.title("Training vs Validation Accuracy (CIFAR-10)", fontsize=14, fontweight='bold')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True, alpha=0.3)

# (3) Learning Rate Schedule
plt.subplot(2, 2, 3)
plt.plot(range(1, NUM_EPOCHS + 1), learning_rates, label="Learning Rate", color='purple')
plt.yscale("log")
plt.title("Learning Rate Schedule", fontsize=14, fontweight='bold')
plt.xlabel("Epoch")
plt.ylabel("Learning Rate (log scale)")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, "resnet50_cifar10_imbalance_training_results.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"CIFAR-10 training performance plots saved to: {plot_path}")

