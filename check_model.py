"""
check_model.py
Verifica que el modelo carga bien y muestra info del checkpoint.
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import torch
import torch.nn as nn
from torchvision import models

NUM_CLASSES = 55
MODEL_PATH  = "models/best_efficientnet_b0.pth"

class DogBreedClassifier(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=None)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        return self.backbone(x)

# ── Cargar checkpoint ─────────────────────────────────────────────────────────
print(f"Cargando: {MODEL_PATH}")
ckpt = torch.load(MODEL_PATH, map_location="cpu")

print(f"\nTipo del checkpoint : {type(ckpt)}")

if isinstance(ckpt, dict):
    print(f"Claves del dict     : {list(ckpt.keys())[:10]}")

    # Mostrar métricas si existen
    for k in ["epoch", "best_val_acc", "val_acc", "train_acc", "best_acc"]:
        if k in ckpt:
            print(f"  {k}: {ckpt[k]}")

    # Determinar dónde están los pesos
    if "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
    elif "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    else:
        sd = ckpt  # el dict entero es el state_dict
else:
    sd = ckpt

print(f"\nCapas en state_dict ({len(sd)} entradas):")
for i, (k, v) in enumerate(sd.items()):
    print(f"  {k}: {tuple(v.shape)}")
    if i >= 15:
        print(f"  ... ({len(sd) - 16} más)")
        break

# ── Cargar pesos en el modelo ─────────────────────────────────────────────────
model = DogBreedClassifier()
result = model.load_state_dict(sd, strict=True)
print(f"\n✅ Pesos cargados — missing: {result.missing_keys}, unexpected: {result.unexpected_keys}")

# ── Test con imagen aleatoria ─────────────────────────────────────────────────
model.eval()
dummy = torch.zeros(1, 3, 224, 224)
with torch.no_grad():
    out   = model(dummy)
    probs = torch.softmax(out, dim=1)[0]

top5  = probs.topk(5)
print(f"\nTest con imagen negra (debe ser ~1/55 = 1.8% por clase si no entrenó bien):")
for prob, idx in zip(top5.values, top5.indices):
    print(f"  clase {idx.item():2d}: {prob.item()*100:.2f}%")

print(f"\nSuma de probabilidades: {probs.sum().item():.4f}")