import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import HybridPneumoniaModel

def train_model():
    print("=" * 60)
    print("      STARTING HYBRID MODEL TRAINING PIPELINE      ")
    print("=" * 60)
    
    # 1. Image preprocessing and data augmentation
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    data_dir = 'dataset'
    
    # Check dataset existence
    if not os.path.exists(data_dir) or not os.path.exists(os.path.join(data_dir, 'train')):
        print("[!] ERROR: 'dataset' folder structure not found.")
        print("[!] Ensure dataset/train and dataset/val contain images.")
        return

    # Load datasets
    try:
        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                          for x in ['train', 'val']}
        dataloaders = {x: DataLoader(image_datasets[x], batch_size=8, shuffle=True, num_workers=0)
                       for x in ['train', 'val']}
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
        print(f"[OK] Training samples loaded  : {dataset_sizes['train']}")
        print(f"[OK] Validation samples loaded: {dataset_sizes['val']}")
    except Exception as e:
        print(f"[!] Dataset Loading Error: {e}")
        return

    # Select hardware device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[OK] Training Device selected: {device}")

    # Initialize model
    model = HybridPneumoniaModel(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)

    epochs = 3
    best_acc = 0.0

    # Training execution loop
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 30)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # Save best checkpoint
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), 'pneumonia_hybrid_model.pth')
                print(f"[OK] Checkpoint saved! Best Validation Accuracy: {best_acc:.4f}")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE: Weights saved to 'pneumonia_hybrid_model.pth'")
    print("=" * 60)

if __name__ == '__main__':
    train_model()