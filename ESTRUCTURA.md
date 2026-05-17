# 📁 ESTRUCTURA DEL PROYECTO - GUÍA COMPLETA

Documentación detallada de cada componente del Dog Breed Classifier.

---

## 📑 Índice

1. [Archivos en Raíz](#archivos-en-raíz)
2. [Carpeta `/models`](#carpeta-models)
3. [Carpeta `/archive`](#carpeta-archive)
4. [Flujo de Datos](#flujo-de-datos)
5. [Cómo Entrenar](#cómo-entrenar)
6. [Cómo Hacer Predicciones](#cómo-hacer-predicciones)

---

## 📄 Archivos en Raíz

### `app.py` - APLICACIÓN PRINCIPAL
**¿Qué es?** La interfaz web Streamlit del proyecto.

**¿Qué hace?**
- ✅ Carga el modelo desde `models/best_efficientnet_b0.pth`
- ✅ Muestra interfaz bilingüe (Español/English)
- ✅ Maneja upload de imágenes
- ✅ Permite captura por cámara web
- ✅ Visualiza predicciones con Plotly

**¿Cómo funciona?**
```
Usuario abre app.py
  ↓
Streamlit carga la interfaz
  ↓
Usuario sube imagen o toma foto
  ↓
app.py preprocesa la imagen
  ↓
Carga modelo de models/best_efficientnet_b0.pth
  ↓
Realiza predicción
  ↓
Muestra resultados (Top 5 + gráficos)
```

**Funciones principales:**
```python
load_model()           # Carga modelo preentrenado
preprocess_image()     # Normaliza entrada
predict()              # Forward pass
visualize_results()    # Gráficos Plotly
```

**Notas:**
- Usa `@st.cache_resource` para cachear el modelo
- Soporta checkpoint completo o solo state_dict
- Interfaz responsive y moderna

---

### `requirements.txt` - DEPENDENCIAS

**¿Qué incluye?**
```
streamlit==1.42.0          # Web framework
torch==2.2.2               # Deep learning
torchvision==0.17.2        # Modelos preentrenados
numpy==1.26.4              # Cálculos numéricos
pandas==2.1.4              # Análisis de datos
Pillow==10.1.0             # Procesamiento de imágenes
plotly==5.24.1             # Gráficos interactivos
```

**Instalación:**
```bash
pip install -r requirements.txt
```

**Para GPU (NVIDIA):**
```bash
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt --no-deps
```

---

### `README.md` - DOCUMENTACIÓN GENERAL

Contiene:
- Overview del proyecto
- Características principales
- Arquitectura del modelo
- Instrucciones de uso
- Troubleshooting

---

## 📁 Carpeta `/models` - MODELOS PREENTRENADOS

### `best_efficientnet_b0.pth` - MODELO ACTIVO ⭐

**¿Qué es?** Archivo PyTorch con pesos del modelo entrenado.

**Características:**
- ✅ Modelo EfficientNetB0
- ✅ 55 clases (razas de perros)
- ✅ Top-1 Accuracy: 92%+
- ✅ Top-5 Accuracy: 98%+
- ✅ Tamaño: ~17 MB

**¿Dónde se usa?**
```
app.py → carga este archivo
archive/predict.py → lo usa para predicciones
```

**¿Cómo se crea?**
```bash
python archive/training.py
```

**Estructura interna:**
```
.pth file = PyTorch state_dict
  ├─ backbone (EfficientNetB0)
  │  ├─ features (extractor)
  │  └─ classifier (cabezal personalizado)
  └─ parámetros entrenables
```

---

## 📁 Carpeta `/archive` - SCRIPTS DE ENTRENAMIENTO

### `model.py` - ARQUITECTURA DEL MODELO

**¿Qué es?** Define la arquitectura del clasificador.

**Clase Principal: `DogBreedClassifier`**
```python
class DogBreedClassifier(nn.Module):
    """
    Clasificador de razas basado en EfficientNetB0
    
    Arquitectura:
    Input (224×224) → EfficientNetB0 → Cabezal Custom → Output (55)
    """
```

**Métodos principales:**
```python
__init__(num_classes=55, pretrained=True)
    # Inicializa modelo con pesos de ImageNet

forward(x)
    # Forward pass: imagen → predicción

freeze_backbone(freeze=True)
    # Congela pesos para fine-tuning

unfreeze_backbone()
    # Descongela para entrenamiento completo

count_parameters()
    # Cuenta parámetros totales/entrenables
```

**Arquitectura Detallada:**
```
INPUT: [B, 3, 224, 224]
  ↓
EfficientNetB0 Backbone: 3.9M parámetros
  ├─ Conv2d layers
  ├─ MBConv blocks
  └─ Output: [B, 1280]
  ↓
Cabezal Personalizado: 0.4M parámetros
  ├─ Dropout(0.2)
  ├─ Linear(1280→1024)
  ├─ BatchNorm1d(1024)
  ├─ ReLU()
  ├─ Dropout(0.3)
  ├─ Linear(1024→512)
  ├─ BatchNorm1d(512)
  ├─ ReLU()
  ├─ Dropout(0.2)
  └─ Linear(512→55)
  ↓
OUTPUT: [B, 55] (logits)
```

---

### `config.py` - CONFIGURACIÓN CENTRALIZADA

**¿Qué es?** Archivo con todos los hiperparámetros del proyecto.

**Por qué?** Cambiar valores sin tocar múltiples archivos.

**Secciones principales:**

#### RUTAS
```python
PROJECT_ROOT = r"c:\Users\ESTEBAN\Desktop\dog-breed-classifier"
DATASET_PATH = "archive/dataset"   # Dataset crudo
DATA_PATH = "archive/data"         # Train/Val/Test split
MODELS_PATH = "models"             # Modelos guardados
LOGS_PATH = "logs"                 # Logs de entrenamiento
```

#### MODELO
```python
MODEL_NAME = "efficientnet_b0"
NUM_CLASSES = 55                   # Razas
IMAGE_SIZE = 224                   # Entrada ImageNet
PRETRAINED = True                  # Usar pesos ImageNet
```

#### ENTRENAMIENTO (RTX 3050)
```python
BATCH_SIZE = 16                    # Máximo para 4GB VRAM
VAL_BATCH_SIZE = 32               # Sin backprop
GRADIENT_ACCUMULATION_STEPS = 2   # Simula batch 32
EPOCHS = 100                       # Con early stopping
INITIAL_LR = 1e-3                 # Learning rate
USE_AMP = True                     # Mixed Precision
```

#### EARLY STOPPING
```python
EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 20       # Épocas sin mejora
EARLY_STOPPING_MIN_DELTA = 1e-4   # Mínima mejora
```

**¿Cómo modificar?**
```python
# Cambiar batch size
Config.BATCH_SIZE = 32

# Cambiar epochs
Config.EPOCHS = 50

# Cambiar learning rate
Config.INITIAL_LR = 1e-4
```

---

### `augmentation.py` - TRANSFORMACIÓN DE DATOS

**¿Qué es?** Pipeline de augmentación con Albumentations.

**¿Por qué?** Aumentar dataset sintéticamente para evitar sobreajuste.

**Clase: `AugmentationPipeline`**

#### `get_train_transforms()` - TRANSFORMACIONES AGRESIVAS

```
Redimensionar → 224×224
Rotaciones → ±30°
Perspectiva → distorsiones
Traslaciones → movimientos
Flips → horizontal/vertical
Distorsiones elásticas
Iluminación → brillo, contraste, gamma
Color → hue, saturation, value
Ruido → Gaussiano, multiplicativo
Blur → suave y motion blur
Normalización → ImageNet stats
Tensor → torch.Tensor
```

**Función:**
```python
@staticmethod
def get_train_transforms(image_size=224):
    return A.Compose([...])
```

#### `get_val_transforms()` - TRANSFORMACIONES MÍNIMAS

```
Redimensionar → 224×224
Normalización → ImageNet stats
Tensor → torch.Tensor
```

**Sin augmentación:** Mide accuracy real del modelo.

#### `get_test_transforms()` - IGUAL A VALIDACIÓN

Asegura evaluación objetiva e imparcial.

**¿Cómo usar?**
```python
from augmentation import AugmentationPipeline

# Para entrenamiento
train_transform = AugmentationPipeline.get_train_transforms(224)

# Para validación
val_transform = AugmentationPipeline.get_val_transforms(224)

# Aplicar a imagen numpy
image_tensor = train_transform(image=np.array(image))["image"]
```

---

### `training.py` - ENTRENAMIENTO COMPLETO

**¿Qué es?** Script completo para entrenar el modelo desde cero.

**Proceso:**
```
1. Cargar dataset (55 razas)
2. Aplicar augmentación
3. Crear DataLoaders
4. Inicializar modelo EfficientNetB0
5. Configurar optimizer (AdamW) + scheduler (CosineAnnealing)
6. Entrenar por 100 épocas
7. Validar cada época
8. Early stopping si val_loss no mejora
9. Guardar best checkpoint
```

**Clases principales:**

#### `DogBreedDataset`
```python
class DogBreedDataset(Dataset):
    """Carga imágenes de directorios organizados por clase"""
    
    __init__(image_dir, transform=None)
    __len__()
    __getitem__(idx)  # Retorna (imagen, etiqueta)
```

#### `EarlyStopping`
```python
class EarlyStopping:
    """Detiene entrenamiento si val_loss no mejora"""
    
    __call__(val_loss, model, epoch)  # Retorna True si debe parar
```

#### `Trainer`
```python
class Trainer:
    """Loop de entrenamiento con AMP y gradient accumulation"""
    
    train_epoch()     # Una época de entrenamiento
    validate()        # Validación
    train()           # Loop completo
```

**¿Cómo ejecutar?**
```bash
python archive/training.py
```

**¿Qué genera?**
- `models/best_efficientnet_b0.pth` - Mejor modelo

---

### `predict.py` - PREDICCIONES AVANZADAS

**¿Qué es?** Módulo para hacer predicciones con el modelo entrenado.

**Clase: `Predictor`**

#### `__init__(model_path)`
```python
# Carga el modelo y clases disponibles
predictor = Predictor("models/best_efficientnet_b0.pth")
```

#### `predict_single(image_path, top_k=5)`
```python
# Predicción en una imagen
predictions = predictor.predict_single("mi_perro.jpg", top_k=5)
# Retorna: [('labrador', 0.95), ('golden_retriever', 0.04), ...]
```

#### `predict_directory(directory_path, output_file=None)`
```python
# Predicciones en todas las imágenes de un directorio
results = predictor.predict_directory("fotos/", output_file="resultados.json")
```

#### `predict_batch(image_paths, top_k=5)`
```python
# Predicciones en lote
results = predictor.predict_batch(["foto1.jpg", "foto2.jpg"], top_k=5)
```

**¿Cómo usar?**
```python
from archive.predict import Predictor

# Crear predictor
predictor = Predictor()

# Predicción
results = predictor.predict_single("labrador.jpg", top_k=5)
for rank, (breed, probability) in enumerate(results, 1):
    print(f"{rank}. {breed}: {probability*100:.2f}%")
```

---

### `dataset/` - DATASET ORIGINAL

**Estructura:**
```
archive/dataset/
├── labrador/              # Clase 1
│   ├── imagen1.jpg
│   ├── imagen2.jpg
│   └── ... (N imágenes)
├── golden_retriever/      # Clase 2
│   ├── imagen1.jpg
│   └── ...
├── german_sheperd/        # Clase 3
└── ... (55 clases totales)
```

**Total:** 55 razas de perros

**Uso:** Se copia/procesa a `archive/data/` para training.

---

### `data/` - DATASET PROCESADO

**Estructura:**
```
archive/data/
├── train/               # 70% del dataset
│   ├── labrador/
│   ├── golden_retriever/
│   └── ...
├── val/                 # 15% del dataset
│   ├── labrador/
│   └── ...
└── test/                # 15% del dataset
    ├── labrador/
    └── ...
```

**Propósito:**
- `train/`: Entrenar el modelo
- `val/`: Validación (early stopping)
- `test/`: Evaluación final (sin ver durante entrenamiento)

---

## 🔄 Flujo de Datos

### A. Flujo de Entrenamiento

```
dataset/ (55 razas)
  ↓
DogBreedDataset
  ├─ Carga imágenes
  ├─ Mapea clase → índice
  ├─ Retorna (imagen_np, label)
  ↓
AugmentationPipeline.get_train_transforms()
  ├─ Rotaciones, flips, ruido
  ├─ Normaliza con ImageNet stats
  ├─ Convierte a tensor
  ↓
DataLoader (batch_size=16)
  ├─ Agrupa 16 muestras
  ├─ Shuffle=True para train
  ↓
Trainer.train_epoch()
  ├─ Forward pass
  ├─ CrossEntropyLoss
  ├─ Backward pass (AMP)
  ├─ Gradient accumulation
  ↓
Model (EfficientNetB0)
  ├─ Actualiza pesos
  ↓
EarlyStopping
  ├─ Monitorea val_loss
  ├─ Guarda best_model si mejora
  ↓
models/best_efficientnet_b0.pth
```

### B. Flujo de Predicción

```
Imagen del usuario (JPG/PNG/BMP)
  ↓
app.py
  ├─ Carga con Pillow
  ├─ Redimensiona a 224×224
  ├─ Convierte a numpy array
  ↓
AugmentationPipeline.get_val_transforms()
  ├─ Normaliza con ImageNet stats
  ├─ Convierte a tensor
  ↓
Model.forward()
  ├─ EfficientNetB0 backbone
  ├─ Cabezal personalizado
  ├─ Retorna logits [1, 55]
  ↓
Softmax
  ├─ Convierte a probabilidades
  ↓
torch.topk(5)
  ├─ Top 5 predicciones
  ↓
Visualización en Streamlit
  ├─ Texto: "Predicción: Labrador (95%)"
  ├─ Gráfico Plotly Top 5
  └─ Información sobre la raza
```

---

## 🚀 Cómo Entrenar

### Paso 1: Preparar Dataset

```bash
# Crear estructura
mkdir archive/data/train archive/data/val archive/data/test

# Copiar/mover imágenes desde archive/dataset/
# Asegúrate de mantener estructura por clase:
# archive/data/train/
#   ├── labrador/
#   ├── golden_retriever/
#   └── ...
```

### Paso 2: Ejecutar Entrenamiento

```bash
cd dog-breed-classifier
python archive/training.py
```

### Paso 3: Monitorear

```
Epoch 1/100
Train Loss: 3.1234 | Acc: 12.35%
Val Loss: 2.9876 | Acc: 18.50%
✓ Modelo guardado: models/best_efficientnet_b0.pth

Epoch 2/100
Train Loss: 2.8765 | Acc: 24.10%
Val Loss: 2.7654 | Acc: 31.25%
✓ Modelo guardado: models/best_efficientnet_b0.pth

...

⛔ Early stopping activado (sin mejora en 20 épocas)
```

### Paso 4: Validar

```bash
python archive/predict.py
```

---

## 🔮 Cómo Hacer Predicciones

### Opción 1: Usar app.py (Recomendado)
```bash
streamlit run app.py
```

### Opción 2: Script Python

```python
from archive.predict import Predictor

# Crear predictor
predictor = Predictor("models/best_efficientnet_b0.pth")

# Predicción individual
results = predictor.predict_single("mi_perro.jpg", top_k=5)

# Mostrar resultados
print("\n🐕 Predicciones:")
for rank, (breed, prob) in enumerate(results, 1):
    print(f"{rank}. {breed}: {prob*100:.2f}%")
```

### Opción 3: En Directorio

```python
from archive.predict import Predictor

predictor = Predictor()

# Predicciones en carpeta
results = predictor.predict_directory(
    "fotos_perros/",
    output_file="predicciones.json"
)
```

---

## 📊 Resumen de Archivos

| Archivo | Propósito | Modificar? |
|---------|----------|-----------|
| `app.py` | Aplicación Streamlit | No (usado en producción) |
| `requirements.txt` | Dependencias | Sí (si necesitas nuevas librerías) |
| `model.py` | Arquitectura | No (está optimizada) |
| `config.py` | Configuración | **Sí** (hiperparámetros) |
| `augmentation.py` | Augmentación | Sí (con cautela) |
| `training.py` | Entrenamiento | Sí (para mejorar) |
| `predict.py` | Predicciones | No (está completo) |
| `dataset/` | Imágenes | Sí (agregar más) |
| `data/` | Train/Val/Test | Generado automáticamente |
| `models/` | Modelos guardados | Generado en entrenamiento |

---

## 🎓 Conceptos Educativos

Este proyecto enseña:
- ✅ **Transfer Learning**: Usar EfficientNetB0 preentrenado
- ✅ **Data Augmentation**: Aumentar dataset sintéticamente
- ✅ **Early Stopping**: Evitar sobreajuste
- ✅ **Mixed Precision**: Optimización de memoria
- ✅ **Gradient Accumulation**: Simular batches grandes
- ✅ **Streamlit**: Deployar ML models
- ✅ **PyTorch**: Framework de deep learning

---

**¡Ahora entiendes la estructura completa del proyecto! 🚀**

*Para preguntas: consulta el README.md o archive/ scripts.*
