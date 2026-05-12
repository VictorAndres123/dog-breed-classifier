# Dog Breed Classifier - Pipeline Profesional

## Estructura del Proyecto

```
dog-breed-classifier/
├── dataset/                    # Dataset original (55 razas)
├── data/                       # Dataset dividido (train/val/test)
│   ├── train/
│   ├── val/
│   └── test/
├── models/                     # Modelos guardados
│   └── best_efficientnet_b0.pth
├── logs/                       # Logs de entrenamiento
│   └── training_history.json
│
├── split_dataset.py           # ✅ Dividir train/val/test
├── augmentation.py            # ✅ Data augmentation profesional
├── config.py                  # ✅ Configuración centralizada
├── model.py                   # ✅ Arquitectura EfficientNetB0
├── training.py                # ✅ Script de entrenamiento profesional
├── predict.py                 # Script de predicción
├── requirements_training.txt  # Dependencias
└── README.md
```

## Características

✅ **Dividir dataset**: train/val/test con proporciones configurables
✅ **Data Augmentation**: 15+ técnicas profesionales con Albumentations
✅ **GPU Support**: Automático (CUDA si está disponible)
✅ **Mixed Precision Training**: FP16 automático
✅ **Early Stopping**: Previene overfitting
✅ **Learning Rate Scheduler**: Cosine Annealing, Step, Exponential
✅ **Fine-tuning EfficientNetB0**: Preentrenado en ImageNet
✅ **Guardar mejor modelo**: Automático con checkpoint

## Instalación

```bash
# 1. Instalar dependencias
pip install -r requirements_training.txt

# En Windows, para GPU CUDA:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Uso

### Paso 1: Dividir Dataset

```bash
python split_dataset.py
```

Esto crea:
- `data/train/` - 70% de imágenes para entrenamiento
- `data/val/` - 15% de imágenes para validación
- `data/test/` - 15% de imágenes para prueba

### Paso 2: Configurar Training (opcional)

Editar `config.py` para personalizar:

```python
# Arquitectura
BATCH_SIZE = 32              # Aumentar si tienes más GPU memory
EPOCHS = 100                 # Epochs de entrenamiento
INITIAL_LR = 1e-3           # Learning rate inicial

# Early Stopping
EARLY_STOPPING = True       # Activar early stopping
EARLY_STOPPING_PATIENCE = 15 # Paciencia en epochs

# Scheduler
LR_SCHEDULER = "cosine"     # cosine, step, exponential
SCHEDULER_T_MAX = 100       # Para cosine

# Optimizador
OPTIMIZER = "adamw"         # adam, adamw, sgd

# Augmentación
USE_AUGMENTATION = True     # Activar data augmentation
```

### Paso 3: Entrenar Modelo

```bash
python training.py
```

**Durante el entrenamiento verás:**
- Progreso en barra de tqdm
- Loss y Accuracy en train/val
- Learning rate actual
- Dispositivo (GPU/CPU)
- Mejor modelo guardado automáticamente

**Ejemplo de salida:**
```
============================================================
CONFIGURACIÓN DEL PROYECTO
============================================================
Device: cuda
GPU disponible: True
GPU Name: NVIDIA GeForce RTX 3060
GPU Memory: 12.00 GB

Modelo: efficientnet_b0
Clases: 55
Tamaño de imagen: 224x224

Batch Size (train): 32
Batch Size (val): 64
Epochs: 100
Learning Rate: 0.001
Optimizer: adamw
LR Scheduler: cosine

Early Stopping: True
Patience: 15

Mixed Precision Training: True
Data Augmentation: True
============================================================
```

### Paso 4: Hacer Predicciones

```bash
python predict.py
```

**Predicción en imagen individual:**
```python
from predict import Predictor

predictor = Predictor()
predictions = predictor.predict_single("ruta/a/imagen.jpg", top_k=5)

for rank, (breed, prob) in enumerate(predictions, 1):
    print(f"{rank}. {breed}: {prob*100:.2f}%")
```

**Predicción en directorio:**
```python
results = predictor.predict_directory("ruta/a/directorio", 
                                      output_file="predicciones.json")
```

## Configuración de GPU

### Detectar GPU

```python
import torch
print(torch.cuda.is_available())           # True si GPU está disponible
print(torch.cuda.get_device_name(0))       # Nombre de la GPU
print(torch.cuda.get_device_properties(0)) # Propiedades
```

### Optimizar para GPU

En `config.py`:
- `USE_AMP = True` → Mixed Precision (FP16) - Más rápido, menos memoria
- `BATCH_SIZE = 32` → Aumentar si tienes más VRAM
- `NUM_WORKERS = 4` → Aumentar para más paralelismo

## Early Stopping

El modelo se detiene automáticamente si:
- La validación loss no mejora por 15 epochs
- Se guarda automáticamente el mejor modelo

Configurar en `config.py`:
```python
EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 15      # Epochs de paciencia
EARLY_STOPPING_MIN_DELTA = 1e-4   # Cambio mínimo para mejorar
```

## Learning Rate Scheduler

Elige el scheduler en `config.py`:

1. **Cosine Annealing** (recomendado):
   ```python
   LR_SCHEDULER = "cosine"
   SCHEDULER_T_MAX = 100  # Número total de epochs
   ```

2. **Step Decay**:
   ```python
   LR_SCHEDULER = "step"
   SCHEDULER_STEP_SIZE = 30    # Reducir cada 30 epochs
   SCHEDULER_GAMMA = 0.1       # Multiplicar por 0.1
   ```

3. **Exponential Decay**:
   ```python
   LR_SCHEDULER = "exponential"
   SCHEDULER_GAMMA = 0.95      # Multiplicar por 0.95 cada epoch
   ```

## Data Augmentation

Técnicas aplicadas automáticamente en entrenamiento:

- Rotaciones (±30°)
- Traslaciones y escalado
- Flips horizontales y verticales
- Distorsiones elásticas
- Cambios de brillo y contraste
- Cambios de color (Hue, Saturation)
- Lluvia y niebla
- Ruido (Gaussiano, ISO, Multiplicativo)
- Blur y motion blur
- CLAHE (Adaptive Histogram Equalization)

En validación/test: Solo resize y normalización (sin augmentation)

## Fine-tuning de EfficientNetB0

El modelo:
- Carga pesos preentrenados de ImageNet
- Modifica la capa de clasificación:
  - Dropout (0.2)
  - Linear (1280 → 1024)
  - BatchNorm + ReLU
  - Dropout (0.3)
  - Linear (1024 → 512)
  - BatchNorm + ReLU
  - Dropout (0.2)
  - Linear (512 → 55 clases)

Descongelar backbone después de N epochs (opcional):
```python
model.freeze_backbone(freeze=True)   # Congelar
model.unfreeze_backbone()             # Descongelar
```

## Guardar y Cargar Modelo

**Guardar:**
```python
torch.save(model.state_dict(), "models/best_model.pth")
```

**Cargar:**
```python
model = create_model()
model.load_state_dict(torch.load("models/best_model.pth", 
                                 map_location=Config.DEVICE))
```

## Resolver Problemas

### GPU out of memory
```python
# En config.py:
BATCH_SIZE = 16  # Reducir batch size
USE_AMP = True   # Usar mixed precision
NUM_WORKERS = 2  # Reducir workers
```

### Entrenamiento muy lento
```python
# En config.py:
NUM_WORKERS = 4        # Aumentar workers
PIN_MEMORY = True      # Si USE_AMP = True
BATCH_SIZE = 64        # Aumentar batch si tienes GPU potente
```

### Overfitting
```python
# En config.py:
EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 10  # Reducir paciencia
USE_AUGMENTATION = True       # Usar augmentation
WEIGHT_DECAY = 1e-4          # Aumentar regularización
```

## Monitoreo

El historial de entrenamiento se guarda en `logs/training_history.json`:

```json
{
  "train_loss": [...],
  "val_loss": [...],
  "train_acc": [...],
  "val_acc": [...],
  "lr": [...]
}
```

Puedes visualizar con Tensorboard:
```bash
tensorboard --logdir logs/
```

## Resultados Esperados

Con 55 razas de perros:
- **Train Accuracy**: ~95%+
- **Val Accuracy**: ~85-92%
- **Training Time**: ~30-60 minutos (RTX 3060)

## Arquitectura del Modelo

```
Input (3, 224, 224)
  ↓
EfficientNetB0 Backbone (Preentrenado)
  ↓
Features: 1280
  ↓
Classifier:
  - Dropout(0.2)
  - Linear(1280 → 1024) + BatchNorm + ReLU
  - Dropout(0.3)
  - Linear(1024 → 512) + BatchNorm + ReLU
  - Dropout(0.2)
  - Linear(512 → 55)
  ↓
Output (55 clases)
```

## Licencia

Este proyecto es de código abierto.
