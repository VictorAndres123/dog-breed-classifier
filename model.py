import torch
import torch.nn as nn
from torchvision import models
from config import Config

class DogBreedClassifier(nn.Module):
    """Modelo EfficientNetB0 para clasificación de razas de perros"""
    
    def __init__(self, num_classes=Config.NUM_CLASSES, pretrained=True):
        super(DogBreedClassifier, self).__init__()
        
        # Cargar EfficientNetB0 preentrenado
        self.backbone = models.efficientnet_b0(pretrained=pretrained)
        
        # Modificar la capa de clasificación
        in_features = self.backbone.classifier[1].in_features
        
        # Crear clasificador personalizado
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
    
    @staticmethod
    def count_parameters(model):
        """Cuenta el número de parámetros del modelo"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total_params, trainable_params
    
    def freeze_backbone(self, freeze=True):
        """Congela o descongela el backbone"""
        for param in self.backbone.features.parameters():
            param.requires_grad = not freeze
    
    def unfreeze_backbone(self):
        """Descongela el backbone"""
        self.freeze_backbone(freeze=False)
    
    def summary(self):
        """Imprime resumen del modelo"""
        total_params, trainable_params = self.count_parameters(self)
        print("\n" + "="*60)
        print("RESUMEN DEL MODELO")
        print("="*60)
        print(f"Modelo: EfficientNetB0")
        print(f"Parámetros totales: {total_params:,}")
        print(f"Parámetros entrenables: {trainable_params:,}")
        print(f"Parámetros congelados: {total_params - trainable_params:,}")
        print("="*60 + "\n")

def create_model(num_classes=Config.NUM_CLASSES, pretrained=True):
    """Factory function para crear el modelo"""
    model = DogBreedClassifier(num_classes=num_classes, pretrained=pretrained)
    return model
