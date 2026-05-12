import torch
from model import create_model
from config import Config

print("CUDA disponible:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("GPU Memory:", torch.cuda.get_device_properties(0).total_memory / 1e9, "GB")

# Crear modelo
print("\nCreando modelo...")
model = create_model(num_classes=Config.NUM_CLASSES, pretrained=Config.PRETRAINED)

# Enviar modelo a GPU/CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

print(f"✓ Modelo enviado a: {device}")
print(f"✓ Modelo en GPU: {next(model.parameters()).is_cuda}")

# Mostrar resumen
model.summary()