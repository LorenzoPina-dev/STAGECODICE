# train_segmentation.py
import torch
from torch.utils.data import DataLoader, random_split
import torch_directml
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision import models
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from learnHsiFooding.SegmentationDataset import SegmentationDataset
import os
from gestioneCNN.CustomAdamW import CustomAdamW
import albumentations as A
from albumentations.pytorch import ToTensorV2
from gestioneCNN.ManageCNN import ManageCNN
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import gc
import numpy as np
import segmentation_models_pytorch as smp



gc.collect()

image_dir = "./immaginihsifoodCannon"
json_dir = "./label"
batch_size = 16
multispectral = True
input_channels = 4 if multispectral else 3
image_size = 224
train=True


def get_transforms():
    
    train_transform = A.Compose([
        A.Resize(image_size, image_size),
       # A.HorizontalFlip(p=1),
       # A.VerticalFlip(p=1),
        #A.Affine(scale=(0.75, 1.25), p=0.5),
        A.GaussianBlur(blur_limit=(3, 3), sigma_limit=(0.1, 2.0)),
        A.Normalize(mean=[0.5]*input_channels, std=[0.5]*input_channels),
        ToTensorV2(),
    ], additional_targets={'mask': 'mask'})

    val_transform = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.5]*input_channels, std=[0.5]*input_channels),
        ToTensorV2(),
    ], additional_targets={'mask': 'mask'})

    return train_transform, val_transform
# Configurazione
device = torch_directml.device()

print(f"Using device: {torch_directml.device_name(device.index)}")

if __name__ == "__main__":
# Dataset e DataLoader
    train_tf, val_tf = get_transforms()
    dataset = SegmentationDataset(image_dir, json_dir, transform=train_tf)
    dataset_part=int(len(dataset)*0.5)
    dataset_half, _ = random_split(dataset, [dataset_part, len(dataset)-dataset_part])
    train_size = int(0.75 * len(dataset_half))
    val_size = len(dataset_half) - train_size
    train_dataset, val_dataset = random_split(dataset_half, [train_size, val_size])
    val_dataset.dataset.transform = val_tf
    print(f"num immagini train {train_size}| test size {val_size}")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
    num_workers=13,
    persistent_workers=True,
    pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
    num_workers=13,
    persistent_workers=True,
    pin_memory=False)

    # Modello
    model = models.segmentation.fcn_resnet50(weights=None, num_classes=dataset.num_classes)
    '''model = smp.FPN(
            encoder_name="resnet18",      # Backbone
            classes=dataset.num_classes,
            activation=None,
        )
        '''
    model=model.to(device)
    for param in model.parameters():
        param.requires_grad = True

    # Ottimizzatore e Loss
    criterion = nn.CrossEntropyLoss()
    optimizer = CustomAdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    if train:
        manager = ManageCNN(device, model, train_loader, val_loader, lr=5e-4, wd=1e-4)
    # manager.visualizzaSegmentazione()
        totAcc_train, totAcc_test = manager.learn(200, verbose=True)
        manager.save('segmenter.pth')
        # Plotting
        plt.plot(totAcc_test, label="Test Accuracy")
        plt.plot(totAcc_train, label="Train Accuracy")
        plt.legend()
        plt.show()
    else:
         manager=ManageCNN.load('segmenter.pth',device,model,train_loader,val_loader)

    # Valutazione
    y_pred, y_true = manager.get_predictions(train=False)
    y_pred_np = np.array(y_pred).flatten()
    y_true_np = np.array(y_true).flatten()
    print(f"accuracy: {manager._get_accuracy(train=False)}")
    labels = np.unique(np.concatenate((y_true_np, y_pred_np)))

    # Compute the confusion matrix with explicit labels
    cm = confusion_matrix(y_true_np, y_pred_np, labels=labels, normalize='true') * 100

    # Trova le classi con accuratezza più bassa (diagonale della CM)
    accuracies = np.diag(cm)
    worst_class_indices = np.argsort(accuracies)[:20]  # ad es. le 20 peggiori

    # Filtra la CM e le label
    filtered_cm = cm[worst_class_indices][:, worst_class_indices]
    filtered_labels = [dataset.classes[i] for i in worst_class_indices]

    # Visualizza
    ConfusionMatrixDisplay(confusion_matrix=filtered_cm, display_labels=filtered_labels).plot(
            values_format=".0f", cmap="Blues", xticks_rotation=90
            )
    plt.tight_layout()
    plt.show()
    manager.show_segmentations()
    plt.show()

'''
    train_tf, val_tf = get_transforms()
    dataset = SegmentationDataset(image_dir, json_dir, transform=train_tf)
    train_size = int(0.75 * len(dataset)*0.50)
    val_size = len(dataset)*0.50 - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    val_dataset.dataset.transform = val_tf
    print(f"num immagini train {train_size}| test size {val_size}")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
    num_workers=13,
    persistent_workers=True,
    pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
    num_workers=13,
    persistent_workers=True,
    pin_memory=False)

    # Modello
    model = deeplabv3_resnet50(weights=None, num_classes=len(dataset.class_to_idx))
    model = model.to(device)

    # Ottimizzatore e Loss
    criterion = nn.CrossEntropyLoss()
    optimizer = CustomAdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    if train:
        manager = ManageCNN(device, model, train_loader, val_loader, lr=5e-4, wd=1e-4)
        totAcc_train, totAcc_test = manager.learn(200, verbose=True)
        manager.save('modello_ottimizzato.pth')
        # Plotting
        plt.plot(totAcc_test, label="Test Accuracy")
        plt.plot(totAcc_train, label="Train Accuracy")
        plt.legend()
        plt.show()
    else:
        manager=ManageCNN.load('modello_ottimizzato.pth',device,model,train_loader,val_loader)

    
    y_pred, y_true = manager.get_predictions(train=False)
    y_pred_np = np.array(y_pred).flatten()
    y_true_np = np.array(y_true).flatten()
    print(f"accuracy: {manager._get_accuracy(train=False)}")
    y_pred, y_true = manager.get_predictions(train=False)
    cm = confusion_matrix(y_true_np, y_pred_np, normalize='true') * 100
    ConfusionMatrixDisplay(cm, display_labels=dataset.classes).plot(values_format=".0f",cmap="Blues")
''''''
    train_tf, val_tf = get_transforms()
    dataset = SegmentationDataset(image_dir, json_dir, transform=train_tf)
    train_size = int(0.75 * len(dataset)*0.50)
    val_size = len(dataset)*0.50 - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    val_dataset.dataset.transform = val_tf
    print(f"num immagini train {train_size}| test size {val_size}")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
    num_workers=13,
    persistent_workers=True,
    pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
    num_workers=13,
    persistent_workers=True,
    pin_memory=False)

    # Modello
    model = deeplabv3_resnet50(weights=None, num_classes=len(dataset.class_to_idx))
    model = model.to(device)

    # Ottimizzatore e Loss
    criterion = nn.CrossEntropyLoss()
    optimizer = CustomAdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    manager = ManageCNN(device, model, train_loader, val_loader, lr=5e-4, wd=1e-4)
    totAcc_train, totAcc_test = manager.learn(200, verbose=True)
    manager.save('modello_ottimizzato.pth')
    # Plotting
    plt.plot(totAcc_test, label="Test Accuracy")
    plt.plot(totAcc_train, label="Train Accuracy")
    plt.legend()
    plt.show()

    # Training
    num_epochs = 20
    train_losses = []
    val_losses = []
    with open("classes.txt", "w") as f:
        for cls in dataset.class_to_idx:
            f.write(cls + "\n")


    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)['out']
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_losses.append(total_loss / len(train_loader))

        # Valutazione
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)['out']
                loss = criterion(outputs, masks)
                val_loss += loss.item()
        val_losses.append(val_loss / len(val_loader))

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_losses[-1]:.4f} - Val Loss: {val_losses[-1]:.4f}")

    # Salvataggio modello
    torch.save(model.state_dict(), "segmenter.pth")

    # Plot
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.legend()
    plt.title("Loss over epochs")
    plt.show()

    # Salva anche le classi
    with open("classes.txt", "w") as f:
        for cls in dataset.class_to_idx:
            f.write(cls + "\n")
            '''
'''import torch
from torch.utils.data import DataLoader, random_split
from torchvision.models.segmentation import deeplabv3_resnet50
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from learnHsiFooding.SegmentationDataset import SegmentationDataset
import os

import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transforms():
    train_transform = A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2(),
    ])

    return train_transform, val_transform


# Configurazione
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classes = ["background", "Real Apple", "Real Banana", "Real Bread", "Read Donut",]  # aggiorna secondo le classi reali
image_dir = "./immaginihsifoodCannon"
json_path = "labels.json"
batch_size = 32

# Dataset e DataLoader
train_tf, val_tf = get_transforms()
dataset = SegmentationDataset(image_dir, json_path, classes, transform=train_tf)
train_size = int(0.75 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
val_dataset.dataset.transform = val_tf

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Modello
model = deeplabv3_resnet50(pretrained=False, num_classes=len(classes))
model = model.to(device)

# Ottimizzatore e Loss
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

# Training
num_epochs = 20
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)['out']
        loss = criterion(outputs, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    train_losses.append(total_loss / len(train_loader))

    # Valutazione
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)['out']
            loss = criterion(outputs, masks)
            val_loss += loss.item()
    val_losses.append(val_loss / len(val_loader))

    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_losses[-1]:.4f} - Val Loss: {val_losses[-1]:.4f}")

# Salvataggio modello
torch.save(model.state_dict(), "segmenter.pth")

# Plot
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.legend()
plt.title("Loss over epochs")
plt.show()
'''