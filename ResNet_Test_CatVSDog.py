"""
evaluate_dogs_vs_cats_test_resnet50.py
======================================
Evaluates the fine-tuned ResNet50 model on the Dogs vs Cats test dataset.

Outputs:
1. submission.csv  →  Predicted labels for each test image
2. Confusion matrix
3. Classification metrics
4. Visualization of correctly and incorrectly classified samples

Author: [Your Name]
Date: [Current Date]
"""

import os
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image

# ==== CONFIGURATION ====
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

SCRATCH_DIR = f"/home/users/ntu/po0001yi/scratch"
OUTPUT_DIR = f"/home/users/ntu/po0001yi/scratch/resnet_catvsdog_test"
MODEL_PATH = '/home/users/ntu/po0001yi/scratch/resnet_catvsdog/best_resnet50.pth'

TEST_DIR = '/home/users/ntu/po0001yi/scratch/datasets/test'

BATCH_SIZE = 32
IMG_SIZE = 224
NUM_CLASSES = 2
LABELS = {0: "cat", 1: "dog"}

# ==== DATA TRANSFORM ====
test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ==== LOAD TEST DATA ====
test_data = datasets.ImageFolder(TEST_DIR, transform=test_transform)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
print(f"Loaded test dataset with {len(test_data)} images.")

# ==== MODEL SETUP ====
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(DEVICE)

# ==== LOAD TRAINED WEIGHTS ====
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model checkpoint not found at {MODEL_PATH}")

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print(f"Loaded trained model weights from: {MODEL_PATH}")

# ==== EVALUATION ====
criterion = nn.CrossEntropyLoss()
test_loss, correct, total = 0.0, 0, 0
all_preds, all_labels = [], []

print("\nEvaluating on test dataset...")

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        test_loss += loss.item() * imgs.size(0)
        _, preds = torch.max(outputs, 1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# ==== METRICS ====
test_loss /= total
test_acc = correct / total
print(f"\n==== TEST RESULTS ====")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc*100:.2f}%")

# ==== SAVE SUBMISSION FILE ====
image_files = [os.path.basename(path) for path, _ in test_data.imgs]
submission = pd.DataFrame({
    "id": image_files,
    "label": all_preds
})
submission.to_csv(os.path.join(OUTPUT_DIR, "submission_resnet50.csv"), index=False)
print(f"Submission file saved to: {os.path.join(OUTPUT_DIR, 'submission_resnet50.csv')}")

# ==== CONFUSION MATRIX ====
cm = confusion_matrix(all_labels, all_preds)
cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(6, 5))
sns.heatmap(cm_norm, annot=True, cmap="Blues", fmt=".2f", xticklabels=LABELS.values(), yticklabels=LABELS.values())
plt.title("Dogs vs Cats - ResNet50 Confusion Matrix", fontsize=14, fontweight='bold')
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
cm_path = os.path.join(OUTPUT_DIR, "resnet50_confusion_matrix.png")
plt.savefig(cm_path, dpi=300)
plt.show()

print(f"Confusion matrix saved to: {cm_path}")

# ==== CLASSIFICATION REPORT ====
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=list(LABELS.values()), digits=3))

# ==== VISUALIZATION ====
print("\nAnalyzing incorrect predictions...")
image_paths = [path for path, _ in test_data.imgs]
all_labels, all_preds = np.array(all_labels), np.array(all_preds)
correct_indices = np.where(all_labels == all_preds)[0]
incorrect_indices = np.where(all_labels != all_preds)[0]

# Incorrect examples
fig, axes = plt.subplots(1, 5, figsize=(15, 4))
for i, idx in enumerate(np.random.choice(incorrect_indices, 5, replace=False)):
    img = Image.open(image_paths[idx])
    axes[i].imshow(img)
    axes[i].set_title(f"Pred: {LABELS[all_preds[idx]]}\nTrue: {LABELS[all_labels[idx]]}", color='red')
    axes[i].axis('off')
plt.suptitle("Incorrect Predictions (ResNet50)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
