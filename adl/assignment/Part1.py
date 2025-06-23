r"""°°°
# Assignment 2024 S2
## Part 1: Computer vision - Training from scratch vs Transfer learning

°°°"""
# |%%--%%| <I4tYiqQyPI|NwRCe8V77K>
r"""°°°
### Selecting dataset
https://universe.roboflow.com/hung-5yuey/face-emotion-8vfzj

This is a face emotion computer vision dataset.

The reason why I chose this dataset is because it presents an excellent balance of complexity and accessibility for demonstrating both training a deep learning model from scratch and applying transfer learning techniques.  

Face emotion recognition is a computer vision task that requires capturing subtle facial expressions, making it complex enough to showcase the benefits of transfer learning from pre-trained models which have already learned generalizable image features.  

At the same time, face emotion datasets are readily available and often reasonably sized, allowing for training from scratch within the constraints of a typical assignment without requiring excessive computational resources or training time.  

Furthermore, the task itself is intuitively understandable and relatable, making it easier to interpret the performance differences between models trained using different approaches and to appreciate the practical implications of transfer learning in computer vision.

°°°"""
# |%%--%%| <NwRCe8V77K|YPKsZ09vnd>

import logging
import os
from pathlib import Path
import time

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
from dataset import EmotionDataset
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchinfo import summary
from tqdm.notebook import tqdm
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# |%%--%%| <YPKsZ09vnd|IyyJRdBMw1>

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

CONFIG = {
    # Data parameters
    "batch_size": 32,
    "image_size": 224,
    "num_workers": 4,
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
    "base_path": Path("./face_emotion/"),
    
    # Training parameters
    "num_epochs": 10,
    
    # Evaluation parameters
    "num_samples": 10,
}


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
splits = ["train", "valid", "test"]
dataset_paths = {split: CONFIG["base_path"] / split for split in splits}


data_transforms = {
    "train": transforms.Compose(
        [
            transforms.RandomResizedCrop(CONFIG["image_size"]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(CONFIG["mean"], CONFIG["std"]),
        ]
    ),
    "valid": transforms.Compose(
        [
            transforms.Resize(int(CONFIG["image_size"] * 1.14)),
            transforms.CenterCrop(CONFIG["image_size"]),
            transforms.ToTensor(),
            transforms.Normalize(CONFIG["mean"], CONFIG["std"]),
        ]
    ),
    "test": transforms.Compose(
        [
            transforms.Resize(int(CONFIG["image_size"] * 1.14)),
            transforms.CenterCrop(CONFIG["image_size"]),
            transforms.ToTensor(),
            transforms.Normalize(CONFIG["mean"], CONFIG["std"]),
        ]
    ),
}


image_datasets = {
    split: EmotionDataset(str(dataset_paths[split]), data_transforms[split])
    for split in splits
}
dataloaders = {
    split: DataLoader(
        image_datasets[split],
        batch_size=CONFIG["batch_size"],
        shuffle=(split == "train"),
        num_workers=CONFIG["num_workers"],
        pin_memory=True,
        drop_last=(split == "train"),
    )
    for split in splits
}

dataset_sizes = {split: len(image_datasets[split]) for split in splits}
class_names = [name.strip() for name in image_datasets["train"].classes]

# |%%--%%| <IyyJRdBMw1|UyrDwj2fNE>

def train_model(
    model,
    dataloaders,
    criterion,
    optimizer,
    scheduler,
    num_epochs=CONFIG["num_epochs"],
    model_name="Model",
):
    """Train a model and save the best weights"""
    best_val_acc = 0.0
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    # Move model to device once
    model = model.to(device)
    criterion = criterion.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = running_corrects = 0

        train_pbar = tqdm(
            dataloaders["train"], desc="Training Phase", position=1, leave=False
        )

        for inputs, labels in train_pbar:
            # Move batch to device
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Zero gradients
            optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()

            # Regular training
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            with torch.no_grad():
                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        epoch_loss = running_loss / dataset_sizes["train"]
        epoch_acc = running_corrects.float() / dataset_sizes["train"]
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc.item())

        # Validation phase
        model.eval()
        running_loss = running_corrects = 0
        valid_pbar = tqdm(
            dataloaders["valid"], desc="Validation Batches", position=1, leave=False
        )

        with torch.no_grad():
            for inputs, labels in valid_pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                valid_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        epoch_loss = running_loss / dataset_sizes["valid"]
        epoch_acc = running_corrects.float() / dataset_sizes["valid"]
        val_losses.append(epoch_loss)
        val_accs.append(epoch_acc.item())

        # Scheduler and logging phase
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        logger.info(f"Train Loss: {train_losses[-1]:.4f} Acc: {train_accs[-1]:.4f}")
        logger.info(f"Val Loss: {val_losses[-1]:.4f} Acc: {val_accs[-1]:.4f}")

        scheduler.step(epoch_loss)
        if epoch_acc > best_val_acc:
            best_val_acc = epoch_acc
            print("Saving best model checkpoint...")
            torch.save(model.state_dict(), f"best_{model_name}.pth")

    # Plot training history
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(train_losses, label="Train")
    ax1.plot(val_losses, label="Val")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()

    ax2.plot(train_accs, label="Train")
    ax2.plot(val_accs, label="Val")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.legend()

    plt.tight_layout()
    plt.show()

    return train_losses, val_losses, train_accs, val_accs

# |%%--%%| <UyrDwj2fNE|JFhut3t38f>
r"""°°°
### Training from scratch
°°°"""
# |%%--%%| <JFhut3t38f|3lBNCoSyXQ>

class EmotionNet(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Local feature pathway (focuses on smaller facial details)
        self.local_path = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.MaxPool2d(2),
        )

        # Global feature pathway (focuses on overall facial expression)
        self.global_path = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=7, padding=3),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, kernel_size=5, padding=2),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.MaxPool2d(2),
        )

        # Fusion and classification layers
        self.fusion = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(256),
            torch.nn.MaxPool2d(2),
            torch.nn.Dropout2d(0.3),
        )

        # Calculate the size after convolutions and pooling
        # Input size is 224x224
        # After 3 MaxPool2d operations: 28x28
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(256 * 28 * 28, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, num_classes),
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Process through both pathways
        local_features = self.local_path(x)
        global_features = self.global_path(x)

        # Concatenate features from both pathways
        combined = torch.cat([local_features, global_features], dim=1)

        # Fusion and classification
        fused = self.fusion(combined)
        flattened = fused.view(fused.size(0), -1)
        output = self.classifier(flattened)

        return output


model = EmotionNet(num_classes=len(class_names))
summary(model, input_size=(1, 3, 224, 224))

# |%%--%%| <3lBNCoSyXQ|fuWeL0g79f>

criterion = torch.nn.CrossEntropyLoss()
# Higher learning rate for training from scratch
optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.2,
    patience=7,
)

# Train the model using the train_model function
train_losses, val_losses, train_accs, val_accs = train_model(
    model=model,
    dataloaders=dataloaders,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    model_name="Custom_EmotionNet"
)

# |%%--%%| <fuWeL0g79f|Agu2X0S0Rx>
r"""°°°
The custom model shows signs of overfitting.
°°°"""
# |%%--%%| <Agu2X0S0Rx|MZYZ0cJzmN>

def evaluate_model(model, dataloader, device, model_name="Model", num_samples=CONFIG["num_samples"]):
    """Evaluate model with detailed metrics and visualizations"""
    model.eval()
    all_preds, all_labels = [], []
    sample_images, sample_preds, sample_labels = [], [], []

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(tqdm(dataloader, desc=f"Evaluating {model_name}")):
            # Store sample images for visualization
            if len(sample_images) < num_samples:
                idx = min(num_samples - len(sample_images), inputs.size(0))
                sample_images.extend([inputs[:idx]])
                sample_labels.extend(labels[:idx].cpu().numpy())

            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            if len(sample_preds) < num_samples:
                sample_preds.extend(preds[:idx].cpu().numpy())

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    accuracy = sum(1 for x, y in zip(all_preds, all_labels) if x == y) / len(all_labels)
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=3)

    # Visualize sample predictions
    if sample_images:
        sample_images = torch.cat(sample_images, dim=0)
        fig, axes = plt.subplots(1, num_samples, figsize=(20, 4))

        # Denormalize images
        mean = torch.tensor(CONFIG["mean"]).view(3, 1, 1)
        std = torch.tensor(CONFIG["std"]).view(3, 1, 1)
        sample_images = sample_images * std + mean

        for idx in range(num_samples):
            img = sample_images[idx].cpu().permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)
            axes[idx].imshow(img)
            axes[idx].set_title(f"True: {class_names[sample_labels[idx]]}\nPred: {class_names[sample_preds[idx]]}")
            axes[idx].axis("off")

        plt.suptitle(f"{model_name} - Sample Predictions")
        plt.tight_layout()
        plt.show()

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    print(f"\nClassification Report - {model_name}")
    print(report)

    return accuracy, cm, report

# |%%--%%| <MZYZ0cJzmN|7RzskU3srz>

custom_model = EmotionNet(num_classes=len(class_names)).to(device)
custom_model.load_state_dict(torch.load("best_Custom_EmotionNet.pth"))
custom_acc, custom_cm, custom_report = evaluate_model(custom_model, dataloaders["test"], device, "Custom Model")

# |%%--%%| <7RzskU3srz|DZX5SBfw13>
r"""°°°
CustomModel V1 (EmotionNet):
1. Simple dual-pathway architecture:
   - Local path: Small kernels (3x3) for detailed features
   - Global path: Larger kernels (7x7, 5x5) for overall expression
2. Basic fusion with concatenation
3. Large and inefficient architecture (103M parameters)
4. No attention mechanisms
5. Basic linear classifier
Problems:
  * It was like using a sledgehammer to crack a nut (way too big with 103M parameters)
  * Local and Global paths worked independently without sharing any initial info
  * Very basic in how it combined information
  * Not very smart about what parts of the face to focus on
°°°"""
# |%%--%%| <DZX5SBfw13|x0KG8REpov>

class EmotionNetV2(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        # Self-attention module
        class SelfAttention(torch.nn.Module):
            def __init__(self, in_channels):
                super().__init__()
                self.query = torch.nn.Conv2d(in_channels, in_channels//8, 1)
                self.key = torch.nn.Conv2d(in_channels, in_channels//8, 1)
                self.value = torch.nn.Conv2d(in_channels, in_channels, 1)
                self.gamma = torch.nn.Parameter(torch.zeros(1))
                
            def forward(self, x):
                batch_size, C, H, W = x.size()
                
                q = self.query(x).view(batch_size, -1, H*W).permute(0, 2, 1)
                k = self.key(x).view(batch_size, -1, H*W)
                v = self.value(x).view(batch_size, -1, H*W)
                
                attention = torch.bmm(q, k)
                attention = torch.nn.functional.softmax(attention, dim=2)
                
                out = torch.bmm(v, attention.permute(0, 2, 1))
                out = out.view(batch_size, C, H, W)
                
                return self.gamma * out + x

        # Residual block with correct BatchNorm ordering
        class ResidualBlock(torch.nn.Module):
            def __init__(self, in_channels, out_channels, stride=1):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(in_channels, out_channels, 3, stride, 1)
                self.conv2 = torch.nn.Conv2d(out_channels, out_channels, 3, 1, 1)
                
                self.relu = torch.nn.ReLU(inplace=True)
                self.bn1 = torch.nn.BatchNorm2d(out_channels)
                self.bn2 = torch.nn.BatchNorm2d(out_channels)
                
                # Shortcut connection
                self.shortcut = torch.nn.Sequential()
                if stride != 1 or in_channels != out_channels:
                    self.shortcut = torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels, out_channels, 1, stride),
                        torch.nn.BatchNorm2d(out_channels)
                    )
                
            def forward(self, x):
                out = self.conv1(x)
                out = self.relu(out)
                out = self.bn1(out)
                
                out = self.conv2(out)
                out = self.relu(out)
                out = self.bn2(out)
                
                out += self.shortcut(x)
                return out

        # Local pathway with residual connections
        self.local_path = torch.nn.Sequential(
            ResidualBlock(3, 32),
            torch.nn.MaxPool2d(2),
            ResidualBlock(32, 64),
            torch.nn.MaxPool2d(2),
            ResidualBlock(64, 128),
            torch.nn.MaxPool2d(2),
            SelfAttention(128)
        )

        # Global pathway with residual connections
        self.global_path = torch.nn.Sequential(
            ResidualBlock(3, 32),
            torch.nn.MaxPool2d(2),
            ResidualBlock(32, 64),
            torch.nn.MaxPool2d(2),
            ResidualBlock(64, 128),
            torch.nn.MaxPool2d(2),
            SelfAttention(128)
        )

        # Cross-pathway attention
        self.cross_attention = SelfAttention(256)

        # Fusion module with residual connection
        self.fusion = torch.nn.Sequential(
            ResidualBlock(256, 256),
            SelfAttention(256),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Dropout2d(0.3)
        )

        # Classifier with skip connection
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(256, 512),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm1d(512),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, num_classes)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Process through both pathways
        local_features = self.local_path(x)
        global_features = self.global_path(x)
        
        # Combine features
        combined = torch.cat([local_features, global_features], dim=1)
        
        # Apply cross-pathway attention
        attended = self.cross_attention(combined)
        
        # Fusion and classification
        fused = self.fusion(attended)
        flattened = fused.view(fused.size(0), -1)
        output = self.classifier(flattened)
        
        return output


# Create model instance
model = EmotionNetV2(num_classes=len(class_names))
summary(model, input_size=(1, 3, 224, 224))

# |%%--%%| <x0KG8REpov|Ikb9T2UCe0>
r"""°°°
CustomModel V2 (EmotionNetV2):
1. Added several advanced features:
   - Self-attention modules (Like having the ability to say "this part of the face is more important right now")
   - Residual blocks (similar to ResNet) (Like having shortcuts to remember important information from earlier)
   - Cross-pathway attention (Helps the two "eyes" communicate with each other about what they're seeing)
   - Better BatchNorm ordering
2. More efficient architecture (from 103M to 2.1M parameters)
3. Adaptive pooling
4. Improved classifier with BatchNorm

°°°"""
# |%%--%%| <Ikb9T2UCe0|90arztfFEM>

# Adjust learning rate for this architecture
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.01)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.1, patience=5
)

# Train the model using the train_model function
train_losses, val_losses, train_accs, val_accs = train_model(
    model=model,
    dataloaders=dataloaders,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    model_name="Custom_EmotionNetV2"
)

# |%%--%%| <90arztfFEM|V1KRd8Fn1l>

custom_modelv2 = EmotionNetV2(num_classes=len(class_names)).to(device)
custom_modelv2.load_state_dict(torch.load("best_Custom_EmotionNetV2.pth"))

custom_acc, custom_cm, custom_report = evaluate_model(custom_modelv2, dataloaders["test"], device, "Custom Model V2")

# |%%--%%| <V1KRd8Fn1l|wm0AzStAIR>

# Benchmarking function
def benchmark_model(model, input_size=(1, 3, 224, 224), test_dataloader=None):
    model.eval()
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Model size in MB
    model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (
        1024 * 1024
    )

    # Test on actual dataset
    if test_dataloader:
        correct = 0
        total = 0
        test_times = []
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in tqdm(test_dataloader, desc="Testing"):
                inputs, labels = inputs.to(device), labels.to(device)

                start_time = time.time()
                outputs = model(inputs)
                torch.mps.synchronize()
                test_times.append(time.time() - start_time)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = 100 * correct / total
        avg_test_time = sum(test_times) / len(test_times) * 1000  # Convert to ms

        # Calculate per-class metrics
        report = classification_report(
            all_labels, all_preds, target_names=class_names, output_dict=True
        )

        return {
            "Total Parameters": total_params,
            "Trainable Parameters": trainable_params,
            "Model Size (MB)": model_size,
            "Test Accuracy (%)": accuracy,
            "Avg Test Batch Time (ms)": avg_test_time,
            "Per-class Metrics": report,
        }

    return {
        "Total Parameters": total_params,
        "Trainable Parameters": trainable_params,
        "Model Size (MB)": model_size,
        "Avg Batch Inference Time (ms)": avg_inference_time,
    }


def benchmark_models(models_or_benchmarks, test_dataloader=None):
    """Create a comparison report of model benchmarks.

    Args:
        models_or_benchmarks: Either a dictionary of {model_name: model} or list of benchmark results
        test_dataloader: DataLoader for testing (required if passing models dictionary)
    """
    # Determine if we're dealing with models or benchmark results
    if isinstance(models_or_benchmarks, dict):
        # Convert models to benchmarks
        benchmarks = []
        model_names = []
        for name, model in models_or_benchmarks.items():
            benchmarks.append(benchmark_model(model, test_dataloader=test_dataloader))
            model_names.append(name)
    else:
        # Assume we have benchmark results and model names
        benchmarks = models_or_benchmarks[0]
        model_names = models_or_benchmarks[1]

    basic_metrics = [
        "Total Parameters",
        "Trainable Parameters",
        "Model Size (MB)",
        "Test Accuracy (%)",
        "Avg Test Batch Time (ms)",
    ]

    # Print header for basic metrics
    print("\nModel Performance Comparison:")
    header = "Metric".ljust(30)
    for name in model_names:
        header += name.ljust(20)
    print(header)
    print("-" * (30 + 20 * len(model_names)))

    # Print each basic metric row
    for metric in basic_metrics:
        if not any(metric in b for b in benchmarks):
            continue
        row = metric.ljust(30)
        for benchmark in benchmarks:
            val = benchmark.get(metric, "N/A")

            if isinstance(val, (int, float)):
                if metric.endswith("(ms)"):
                    row += f"{val:,.2f}ms".ljust(20)
                elif metric.endswith("(MB)"):
                    row += f"{val:,.2f}MB".ljust(20)
                elif metric.endswith("(%)"):
                    row += f"{val:,.2f}%".ljust(20)
                else:
                    row += f"{val:,}".ljust(20)
            else:
                row += str(val).ljust(20)
        print(row)

# |%%--%%| <wm0AzStAIR|1QwSmqIFzk>

benchmark_models(
    {"Custom Model": custom_model, "Custom Model V2": custom_modelv2},
    test_dataloader=dataloaders["test"],
)

# |%%--%%| <1QwSmqIFzk|iuTR9OBtFP>
r"""°°°
CustomModel V2 (EmotionNetV2):

Problems:
1. The two pathways still started separately - like having two people look at the same thing but starting from different positions
2. The attention system was too complex - like having too many people giving opinions at once
3. Not efficient in processing - took longer to process images (235.82ms vs V1's 115.45ms)
4. Still not focusing well enough on the right features
°°°"""
# |%%--%%| <iuTR9OBtFP|sbXSmz4pzZ>

# Add Channel and Spatial Attention modules
class ChannelAttention(torch.nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.max_pool = torch.nn.AdaptiveMaxPool2d(1)

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(channels, channels // reduction, bias=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(channels // reduction, channels, bias=False),
        )
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x).view(x.size(0), -1))
        max_out = self.fc(self.max_pool(x).view(x.size(0), -1))
        out = self.sigmoid(avg_out + max_out).view(x.size(0), x.size(1), 1, 1)
        return x * out


class SpatialAttention(torch.nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = torch.nn.Conv2d(
            2, 1, kernel_size=kernel_size, padding=kernel_size // 2
        )
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv(x_cat))
        return x * out
        
class EmotionNetV3(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Shared initial layers with better initialization
        self.shared_features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2),
        )

        # Local feature pathway with residual connections
        self.local_conv1 = self._make_local_block(32, 64)
        self.local_conv2 = self._make_local_block(64, 128)

        # Global feature pathway with attention
        self.global_conv1 = self._make_global_block(32, 64)
        self.global_conv2 = self._make_global_block(64, 128)

        # Dual attention mechanisms
        self.channel_attention = ChannelAttention(256)
        self.spatial_attention = SpatialAttention()

        # Final layers with better regularization
        self.final_conv = torch.nn.Sequential(
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(256, num_classes),
        )

    def _make_local_block(self, in_channels, out_channels):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2),
        )

    def _make_global_block(self, in_channels, out_channels):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=5, padding=2),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2),
        )

    def forward(self, x):
        # Initial shared features
        x = self.shared_features(x)

        # Local pathway with residual connection
        local_features = self.local_conv1(x)
        local_features = self.local_conv2(local_features)

        # Global pathway
        global_features = self.global_conv1(x)
        global_features = self.global_conv2(global_features)

        # Combine features
        combined = torch.cat([local_features, global_features], dim=1)

        # Apply dual attention mechanisms
        combined = self.channel_attention(combined)
        combined = self.spatial_attention(combined)

        # Final processing
        features = self.final_conv(combined)
        features = features.view(features.size(0), -1)
        output = self.classifier(features)

        return output


# Create optimizer and scheduler for EmotionNetV3
model = EmotionNetV3(num_classes=len(class_names))
summary(model, input_size=(1, 3, 224, 224))

# |%%--%%| <sbXSmz4pzZ|2XSil5nqwQ>
r"""°°°
CustomModel V3 (EmotionNetV3):
1. Most sophisticated architecture:
   - Shared initial features layer
     * Both eyes look at same spot before splitting their focus
     * Increased efficiency because skip processing same initial information twice
   - Dual attention mechanisms:
     * Channel attention (focuses on important feature channels) (Like having the ability to focus on specific types of features (e.g., "edges are really important for this expression")
     * Spatial attention (focuses on important spatial regions) (Like knowing exactly where to look on the face (e.g., "the mouth area is crucial for this emotion")
   - Better organized local and global pathways
2. Most efficient architecture (1.7M parameters)
3. Better regularization techniques
4. Simplified but effective classifier

°°°"""
# |%%--%%| <2XSil5nqwQ|ypaezYDE1X>

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.001,
    epochs=CONFIG["num_epochs"],
    steps_per_epoch=len(dataloaders["train"]),
    pct_start=0.3,
    div_factor=25,
    final_div_factor=1000,
)

# Train the model using the train_model function
train_losses, val_losses, train_accs, val_accs = train_model(
    model=model,
    dataloaders=dataloaders,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    model_name="Custom_EmotionNetV3"
)

# |%%--%%| <ypaezYDE1X|v0W3QBh575>

custom_modelv3 = EmotionNetV3(num_classes=len(class_names)).to(device)
custom_modelv3.load_state_dict(torch.load("best_Custom_EmotionNetV3.pth"))
custom_acc, custom_cm, custom_report = evaluate_model(custom_modelv3, dataloaders["test"], device, "Custom Model V3")

# |%%--%%| <v0W3QBh575|wKb7pDmWeC>

benchmark_models(
    {"Custom Model": custom_model, "Custom Model V2": custom_modelv2, "Custom Model V3": custom_modelv3},
    test_dataloader=dataloaders["test"],
)

# |%%--%%| <wKb7pDmWeC|cEuhkwEHOD>
r"""°°°
Key Takeaway:
The evolution shows how sometimes:
- Less is more (103M → 1.7M parameters)
- Working smarter beats working harder
- Organization and focus beat raw power
°°°"""
# |%%--%%| <cEuhkwEHOD|lYk2Vx0SVT>
r"""°°°
### Transfer Learning
°°°"""
# |%%--%%| <lYk2Vx0SVT|WQk2VH31vy>

model = models.resnet18(weights="IMAGENET1K_V1")
model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
model = model.to(device)
summary(model, input_size=(1, 3, 224, 224))

# |%%--%%| <WQk2VH31vy|jN5XHmWoc6>

# For ResNet18 (Transfer Learning)
criterion = torch.nn.CrossEntropyLoss()
# Lower learning rate for transfer learning
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode="min", 
    factor=0.1, 
    patience=3,  # More aggressive LR reduction
)

# Train the model using the train_model function
train_losses, val_losses, train_accs, val_accs = train_model(
    model=model,
    dataloaders=dataloaders,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    model_name="ResNet18"
)

# |%%--%%| <jN5XHmWoc6|p8o7NOKSRl>

resnet18_model = models.resnet18(weights=None)
resnet18_model.fc = torch.nn.Linear(resnet18_model.fc.in_features, len(class_names))
resnet18_model = resnet18_model.to(device)
resnet18_model.load_state_dict(torch.load("best_ResNet18.pth"))

resnet_acc, resnet_cm, resnet_report = evaluate_model(resnet18_model, dataloaders["test"], device, "ResNet18")

# |%%--%%| <p8o7NOKSRl|JXwosmS0XT>

benchmark_models(
    {"Custom Model V3": custom_modelv3, "ResNet18": resnet18_model},
    test_dataloader=dataloaders["test"],
)

# |%%--%%| <JXwosmS0XT|7yoFn9wyF2>

model = models.resnet50(weights="IMAGENET1K_V2")
model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
model = model.to(device)

summary(model, input_size=(1, 3, CONFIG["image_size"], CONFIG["image_size"]))

# |%%--%%| <7yoFn9wyF2|jTjuOwWpN6>

# Setup training
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    model.parameters(),
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.1,
    patience=3,
    verbose=True
)

# Train model
train_losses, val_losses, train_accs, val_accs = train_model(
    model=model,
    dataloaders=dataloaders,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    model_name="ResNet50"
)

# |%%--%%| <jTjuOwWpN6|Icnz16cmbw>

resnet50_model = models.resnet50(weights=None)
resnet50_model.fc = torch.nn.Linear(resnet50_model.fc.in_features, len(class_names))
resnet50_model = resnet50_model.to(device)
resnet50_model.load_state_dict(torch.load("best_ResNet50.pth"))

resnet_acc, resnet_cm, resnet_report = evaluate_model(resnet50_model, dataloaders["test"], device, "ResNet50")

# |%%--%%| <Icnz16cmbw|1Zw4pdjek4>

benchmark_models(
    {"Custom Model V3": custom_modelv3, "ResNet18": resnet18_model, "ResNet50": resnet50_model},
    test_dataloader=dataloaders["test"],
)

# |%%--%%| <1Zw4pdjek4|eTCbgVruYL>
r"""°°°
ResNet outperformed the custom model by a huge margin, doing better in almost every class.

This is despite having much lower training times than the custom model.

The custom model overfitted to the training data. Thus, we need to recognize the additional work and experience needed to tune our custom model so that it performs optimally.

Other reasons why ResNet performed better includes:
- ResNet18 has deeper connections (18 trainable layers vs our 6 trainable layers)
    - Thus, ResNet18 fine-tuned is a larger and more complex model but with the use of transfer learning, compute times are shorter than a custom model, even of a smaller size.
    - This is because for transfer learning, the model starts closer to an optimal solution, and less adjustments are needed. 
- ResNet18's architecture is much more optimized than our model, through years of research (https://arxiv.org/abs/1512.03385)
°°°"""
# |%%--%%| <eTCbgVruYL|iWQovql6xv>
r"""°°°
Interestingly, the custom model has much larger model size. This shouldn't be the case, and signifies a problem in our modelling step, since ResNet18 has 18 trainable layers. There was likely a calculation mistake.
°°°"""
# |%%--%%| <iWQovql6xv|5kRHSJT5LL>
r"""°°°
| Aspect | ResNet18 (Transfer Learning) | Custom EmotionNet |
|--------|----------------------------|-------------------|
| **Training Time** | ✅ Faster training (10 epochs sufficient) | ❌ Requires more epochs to converge |
| **Performance** | ✅ Better accuracy and per-class metrics | ❌ Lower overall accuracy |
| **Model Size** | ❌ Larger model (~44M parameters) | ✅ Smaller model size (fewer parameters) |
| **Feature Extraction** | ✅ Pre-learned generic image features | ❌ Has to learn features from scratch |
| **Domain Specificity** | ❌ Generic ImageNet features need adaptation | ✅ Architecture designed specifically for emotion recognition |
| **Resource Requirements** | ❌ Higher memory usage | ✅ Lower memory footprint |
| **Flexibility** | ❌ Fixed architecture constraints | ✅ Can modify architecture easily |
| **Interpretability** | ❌ Complex, harder to interpret | ✅ Simpler architecture, easier to understand |
| **Initialization** | ✅ Starts with meaningful weights | ❌ Random initialization |
| **Data Requirements** | ✅ Works well with smaller datasets | ❌ Needs more data for good performance |
| **Dual Pathway Design** | ❌ Single pathway architecture | ✅ Specialized local and global feature pathways |
| **Computational Cost** | ❌ More computationally intensive | ✅ Lighter computational requirements |

°°°"""