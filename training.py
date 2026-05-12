import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ExponentialLR
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import json

from config import Config
from model import create_model
from augmentation import AugmentationPipeline

# =========================
# REPRODUCIBILIDAD
# =========================
torch.manual_seed(Config.SEED)
np.random.seed(Config.SEED)


# =========================
# DATASET
# =========================
class DogBreedDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = Path(image_dir)
        self.transform = transform

        self.classes = sorted([d.name for d in self.image_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        self.images = []
        self.labels = []

        for cls in self.classes:
            folder = self.image_dir / cls
            for img in folder.glob("*"):
                if img.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    self.images.append(str(img))
                    self.labels.append(self.class_to_idx[cls])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image=np.array(image))["image"]

        return image, label


# =========================
# EARLY STOPPING
# =========================
class EarlyStopping:
    def __init__(self, patience=15, min_delta=1e-4, checkpoint_path="best_model.pth"):
        self.patience = patience
        self.min_delta = min_delta
        self.checkpoint_path = checkpoint_path
        self.best_loss = float("inf")
        self.counter = 0
        self.best_epoch = 0

    def __call__(self, val_loss, model, epoch):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_epoch = epoch
            torch.save(model.state_dict(), self.checkpoint_path)
            print(f"✓ Modelo guardado: {self.checkpoint_path}")
        else:
            self.counter += 1

        return self.counter >= self.patience


# =========================
# TRAINER
# =========================
class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader=None):
        self.model = model.to(Config.DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = self._get_optimizer()
        self.scheduler = self._get_scheduler()

        # =========================
        # AMP FIX (WINDOWS SAFE)
        # =========================
        self.use_amp = Config.USE_AMP and torch.cuda.is_available()
        self.scaler = GradScaler(enabled=self.use_amp)

        self.early_stopping = EarlyStopping(
            patience=Config.EARLY_STOPPING_PATIENCE,
            min_delta=Config.EARLY_STOPPING_MIN_DELTA,
            checkpoint_path=Config.CHECKPOINT_PATH
        ) if Config.EARLY_STOPPING else None

    def _get_optimizer(self):
        if Config.OPTIMIZER == "adam":
            return Adam(self.model.parameters(), lr=Config.INITIAL_LR, weight_decay=Config.WEIGHT_DECAY)
        elif Config.OPTIMIZER == "adamw":
            return AdamW(self.model.parameters(), lr=Config.INITIAL_LR, weight_decay=Config.WEIGHT_DECAY)
        else:
            return SGD(self.model.parameters(), lr=Config.INITIAL_LR, momentum=Config.MOMENTUM)

    def _get_scheduler(self):
        if Config.LR_SCHEDULER == "cosine":
            return CosineAnnealingLR(self.optimizer, T_max=Config.SCHEDULER_T_MAX)
        elif Config.LR_SCHEDULER == "step":
            return StepLR(self.optimizer, step_size=Config.SCHEDULER_STEP_SIZE, gamma=Config.SCHEDULER_GAMMA)
        else:
            return ExponentialLR(self.optimizer, gamma=Config.SCHEDULER_GAMMA)

    # =========================
    # TRAIN
    # =========================
    def train_epoch(self):
        self.model.train()
        total_loss, correct, total = 0, 0, 0

        pbar = tqdm(self.train_loader, desc="Training")

        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(Config.DEVICE), labels.to(Config.DEVICE)

            self.optimizer.zero_grad()

            # =========================
            # AMP SAFE VERSION
            # =========================
            with autocast(enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix({
                "loss": total_loss / (batch_idx + 1),
                "acc": 100. * correct / total
            })

        return total_loss / len(self.train_loader), 100. * correct / total

    # =========================
    # VALIDATION
    # =========================
    def validate(self):
        self.model.eval()
        total_loss, correct, total = 0, 0, 0

        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(Config.DEVICE), labels.to(Config.DEVICE)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

        return total_loss / len(self.val_loader), 100. * correct / total

    # =========================
    # TRAIN LOOP
    # =========================
    def train(self):
        print(f"\n🚀 Entrenando en: {Config.DEVICE}\n")

        best_acc = 0

        for epoch in range(Config.EPOCHS):
            print(f"\nEpoch {epoch+1}/{Config.EPOCHS}")

            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()

            self.scheduler.step()

            print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%")

            if self.early_stopping and self.early_stopping(val_loss, self.model, epoch):
                print("⛔ Early stopping activado")
                break

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.model.state_dict(), Config.CHECKPOINT_PATH)
                print(f"🔥 Mejor modelo guardado: {best_acc:.2f}%")


# =========================
# MAIN
# =========================
def main():
    print("="*60)
    print("DOG BREED CLASSIFIER - TRAINING")
    print("="*60)

    print(f"Device: {Config.DEVICE}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = create_model(num_classes=Config.NUM_CLASSES, pretrained=Config.PRETRAINED)

    train_dataset = DogBreedDataset(
        os.path.join(Config.DATA_PATH, "train"),
        transform=AugmentationPipeline.get_train_transforms(Config.IMAGE_SIZE)
    )

    val_dataset = DogBreedDataset(
        os.path.join(Config.DATA_PATH, "val"),
        transform=AugmentationPipeline.get_val_transforms(Config.IMAGE_SIZE)
    )

    test_dataset = DogBreedDataset(
        os.path.join(Config.DATA_PATH, "test"),
        transform=AugmentationPipeline.get_test_transforms(Config.IMAGE_SIZE)
    )

    # =========================
    # DATALOADERS FIX
    # =========================
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )

    trainer = Trainer(model, train_loader, val_loader, test_loader)
    trainer.train()


if __name__ == "__main__":
    main()