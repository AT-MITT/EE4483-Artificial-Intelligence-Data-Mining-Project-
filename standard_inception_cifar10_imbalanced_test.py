import os
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

# ==== CONFIGURATION ====
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

OUTPUT_DIR = "/home/users/ntu/atulya00/AIDM"  # same as training script
MODEL_PATH = os.path.join(OUTPUT_DIR, "best_cifar10_inceptionV3_balanced.pth")
CIFAR_IMG_SIZE = 224
CIFAR_NUM_CLASSES = 10
BATCH_SIZE = 128

# ==== CLASS LABELS ====
CIFAR_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# ==== DATA PREPROCESSING ====
test_transform = transforms.Compose([
    transforms.Resize((CIFAR_IMG_SIZE, CIFAR_IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ==== LOAD CIFAR-10 TEST DATA ====
test_data = datasets.CIFAR10(root="./data", train=False, download=True, transform=test_transform)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
print(f"Loaded CIFAR-10 test set with {len(test_data)} images.")

# ==== MODEL SETUP ====
model_cifar = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
model_cifar.aux_logits = False

# ---- Freeze all layers ----
for param in model_cifar.parameters():
    param.requires_grad = False

# Unfreeze last Inception blocks
for name, param in model_cifar.named_parameters():
    if "Mixed_7b" in name or "Mixed_7c" in name or "fc" in name:
        param.requires_grad = True

# ---- Replace final fully connected (fc) layer ----
in_features = model_cifar.fc.in_features
model_cifar.fc = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(in_features, 1024),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(1024, CIFAR_NUM_CLASSES)
)

# ---- Replace auxiliary head FC layer ----
aux_in_features = model_cifar.AuxLogits.fc.in_features
model_cifar.AuxLogits.fc = nn.Linear(aux_in_features, CIFAR_NUM_CLASSES)

model_cifar = model_cifar.to(DEVICE)

# ==== LOAD TRAINED WEIGHTS ====
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model checkpoint not found at {MODEL_PATH}")

model_cifar.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model_cifar.eval()
print(f"Loaded trained weights from: {MODEL_PATH}")

# ==== EVALUATION ====
criterion = nn.CrossEntropyLoss()

test_loss, correct, total = 0.0, 0, 0
all_preds, all_labels = [], []

print("\nEvaluating model on CIFAR-10 test set...")
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model_cifar(imgs)
        loss = criterion(outputs, labels)
        test_loss += loss.item() * imgs.size(0)

        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# ==== OVERALL ACCURACY ====
test_loss /= total
test_acc = correct / total
print(f"\n==== TEST RESULTS ====")
print(f"Test Loss: {test_loss:.4f}")
print(f"Overall Test Accuracy: {test_acc*100:.2f}%")

# ==== CLASS-WISE ACCURACY ====
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
class_correct = [0] * CIFAR_NUM_CLASSES
class_total = [0] * CIFAR_NUM_CLASSES

for label, pred in zip(all_labels, all_preds):
    if label == pred:
        class_correct[label] += 1
    class_total[label] += 1

print("\nClass-wise Accuracy:")
for i in range(CIFAR_NUM_CLASSES):
    acc = 100 * class_correct[i] / class_total[i]
    print(f"{CIFAR_CLASSES[i]:>10s}: {acc:.2f}%")

# ==== CONFUSION MATRIX ====
cm = confusion_matrix(all_labels, all_preds)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # normalize per class

plt.figure(figsize=(10, 8))
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=CIFAR_CLASSES, yticklabels=CIFAR_CLASSES)
plt.title("CIFAR-10 Imbalanced Test Confusion Matrix (Normalized)", fontsize=14, fontweight='bold')
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()

conf_matrix_path = os.path.join(OUTPUT_DIR, "inception_cifar10_imbalanced_test_confusion_matrix.png")
plt.savefig(conf_matrix_path, dpi=300, bbox_inches='tight')
plt.show()

print(f"\nConfusion matrix saved to: {conf_matrix_path}")

# ==== DETAILED CLASSIFICATION REPORT ====
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=CIFAR_CLASSES, digits=3))