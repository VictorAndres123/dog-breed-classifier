# 📖 GUÍA EDUCATIVA - CÓMO FUNCIONA EL CÓDIGO

Explicación paso a paso del código principal para estudiantes y desarrolladores.

---

## 📑 Contenido

1. [¿Cómo funciona app.py?](#-cómo-funciona-apppy)
2. [¿Cómo funciona la predicción?](#-cómo-funciona-la-predicción)
3. [¿Cómo funciona el entrenamiento?](#-cómo-funciona-el-entrenamiento)
4. [¿Qué es EfficientNetB0?](#-qué-es-efficientnetb0)
5. [Conceptos Clave](#-conceptos-clave)

---

## 🎯 ¿Cómo funciona app.py?

### Paso 1: Importaciones

```python
import streamlit as st                    # Framework para web
import torch                              # Deep learning
from torchvision import models           # Modelos preentrenados
from PIL import Image                    # Procesar imágenes
import numpy as np                       # Operaciones numéricas
```

**¿Por qué cada librería?**
- `streamlit`: Crear interfaz web sin HTML/CSS/JS
- `torch`: Usar redes neuronales
- `PIL`: Abrir imágenes (JPG, PNG)
- `numpy`: Trabajar con arrays

### Paso 2: Configuración de Streamlit

```python
st.set_page_config(
    page_title="🐶 AI Dog Breed Classifier",
    page_icon="🐾",
    layout="wide"
)
```

**¿Qué hace?**
- Título en la pestaña del navegador
- Icono (emoji)
- Diseño ancho (usa más pantalla)

### Paso 3: Selector de Idioma

```python
language = st.sidebar.selectbox(
    "🌍 Language / Idioma",
    ["English", "Español"]
)

if language == "Español":
    title_text = "🐶 Clasificador de Razas de Perros"
    # ...textos en español
else:
    title_text = "🐶 AI Dog Breed Classifier"
    # ...textos en inglés
```

**¿Cómo funciona?**
1. `st.sidebar.selectbox()` crea un dropdown en el sidebar
2. El usuario selecciona idioma
3. Se cargan los textos correspondientes

### Paso 4: Cargar el Modelo

```python
@st.cache_resource  # ← Muy importante!
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DogBreedClassifier(num_classes=NUM_CLASSES)
    sd = torch.load(MODEL_PATH, map_location=device)
    
    # Soportar diferentes formatos
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    
    model.load_state_dict(sd)
    model.to(device)
    model.eval()  # ← Modo evaluación
    return model, device

model, device = load_model()
```

**¿Qué hace línea por línea?**

| Línea | ¿Qué hace? |
|-------|-----------|
| `@st.cache_resource` | Carga el modelo UNA SOLA VEZ (no recarga cada update) |
| `torch.device(...)` | Detecta si hay GPU disponible |
| `DogBreedClassifier()` | Crea la arquitectura del modelo |
| `torch.load()` | Lee el archivo `.pth` |
| `load_state_dict()` | Asigna los pesos al modelo |
| `model.to(device)` | Mueve el modelo a GPU o CPU |
| `model.eval()` | Pone en modo evaluación (sin dropout) |

**¿Por qué `@st.cache_resource`?**
- Sin él: el modelo se recargría cada vez que el usuario interactúa
- Con él: se carga una sola vez y se reutiliza
- **Resultado**: interfaz mucho más rápida ✨

### Paso 5: Upload de Imagen

```python
uploaded_file = st.file_uploader(upload_text, type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image")
```

**¿Cómo funciona?**
1. `st.file_uploader()` crea un botón para subir archivos
2. Si el usuario sube archivo: `uploaded_file` no es None
3. `Image.open()` lee la imagen
4. `.convert('RGB')` asegura que sea RGB (no RGBA)
5. `st.image()` muestra la imagen en la pantalla

### Paso 6: Preprocesar Imagen

```python
# Redimensionar a 224×224
image_resized = image.resize((IMG_SIZE, IMG_SIZE))

# Convertir a array numpy (0-255)
image_array = np.array(image_resized) / 255.0

# Convertir a tensor PyTorch (3, 224, 224)
image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0)

# Normalizar con ImageNet stats
image_tensor = transform(image_tensor)
```

**¿Qué pasa en cada paso?**

```
Imagen original: 640×480 JPG
    ↓ Redimensionar a 224×224
[224, 224, 3] numpy array (0-255)
    ↓ Convertir a tensor + normalizar
[1, 3, 224, 224] tensor PyTorch (-1 a +1)
    ↓
Listo para modelo
```

**¿Por qué 224×224?** Porque EfficientNetB0 fue entrenado con ImageNet (224×224).

**¿Por qué normalizar?** Porque el modelo espera datos con media=0 y desviación=1.

### Paso 7: Realizar Predicción

```python
with torch.no_grad():  # ← No calcular gradientes
    outputs = model(image_tensor)  # [1, 55] logits
    probabilities = torch.softmax(outputs, dim=1)  # [1, 55] (0-1)

top_probs, top_indices = torch.topk(probabilities, 5)
```

**¿Qué son los logits?**
- Salida ANTES de aplicar softmax
- Valores pueden ser negativos o > 1
- No son probabilidades aún

**¿Por qué `with torch.no_grad()`?**
- No necesitamos gradientes para predicciones
- Ahorrar memoria y velocidad

**¿Qué es softmax?**
```
Logits:          [2.1, 0.5, 1.2, -0.1, ...]
    ↓ Softmax
Probabilidades:  [0.65, 0.15, 0.18, 0.02, ...]
```
- Convierte valores a rango 0-1
- Suma total siempre = 1

### Paso 8: Visualizar Resultados

```python
# Top 1 predicción
best_pred = registered_breeds[top_indices[0][0]]
best_conf = top_probs[0][0].item()

st.markdown(f"## 🎯 {best_pred}: {best_conf*100:.2f}%")

# Gráfico Top 5 con Plotly
import plotly.express as px

fig = px.bar(
    x=top_probs[0].cpu().numpy(),
    y=[registered_breeds[i] for i in top_indices[0]],
    orientation='h'
)
st.plotly_chart(fig)
```

**¿Qué hace?**
- Muestra la mejor predicción en grande
- Crea gráfico interactivo de Top 5
- Usuario puede hacer hover para ver valores exactos

---

## 🔮 ¿Cómo funciona la predicción?

### Conceptos Previos

#### ¿Qué es una red neuronal?
```
Input → Capa 1 → Capa 2 → ... → Output

Ejemplo:
Imagen (224×224) 
    ↓ múltiples capas
[1280 características]  ← Representación comprimida
    ↓ cabezal
[55 probabilidades]  ← Resultado final
```

#### ¿Qué es un tensor?
Un tensor es como un array multidimensional:
- Tensor 1D: [1, 2, 3, 4] (vector)
- Tensor 2D: [[1, 2], [3, 4]] (matriz)
- Tensor 3D: [[[1, 2], [3, 4]]] (cubo)
- Tensor 4D: [[[...]]] (batch de imágenes)

### Flujo de Predicción Detallado

#### Paso 1: Cargar Imagen
```python
image = Image.open("labrador.jpg")  # PIL Image
# Resultado: <PIL.Image.Image object>
```

#### Paso 2: Redimensionar
```python
image = image.resize((224, 224))  # Redimensiona
# Resultado: Imagen 224×224
```

#### Paso 3: Convertir a Array NumPy
```python
image_array = np.array(image)  # numpy array
# Shape: [224, 224, 3]
# Valores: 0-255 (RGB)
```

#### Paso 4: Normalizar
```python
# Restar media
image_tensor = (image_array - [0.485, 0.456, 0.406] * 255)

# Dividir por desviación
image_tensor /= ([0.229, 0.224, 0.225] * 255)

# Rango final: [-2 a +2] aproximadamente
```

#### Paso 5: Convertir a Tensor PyTorch
```python
image_tensor = torch.from_numpy(image_array)
# Shape: [3, 224, 224]
image_tensor = image_tensor.unsqueeze(0)
# Shape: [1, 3, 224, 224]  ← 1 = batch size
```

#### Paso 6: Forward Pass
```python
with torch.no_grad():
    logits = model(image_tensor)  # [1, 55]
```

**¿Qué hace el modelo?**
```
Input [1, 3, 224, 224]
  ↓ EfficientNetB0 Backbone
[1, 1280]  ← Features comprimidas
  ↓ Cabezal Personalizado
[1, 55]  ← Logits (antes de softmax)
```

#### Paso 7: Aplicar Softmax
```python
probabilities = torch.softmax(logits, dim=1)
# [1, 55] con valores 0-1 que suman 1
```

#### Paso 8: Obtener Top-5
```python
top_probs, top_indices = torch.topk(probabilities, 5)

# top_probs: [0.95, 0.04, 0.01, 0.0, 0.0]
# top_indices: [37, 30, 12, 15, 20]  ← indices de clases
```

#### Paso 9: Mapear a Nombres de Razas
```python
predictions = []
for prob, idx in zip(top_probs[0], top_indices[0]):
    breed_name = registered_breeds[idx]
    confidence = prob.item()  # Convertir tensor a float
    predictions.append((breed_name, confidence))

# Resultado:
# [
#   ('Labrador', 0.95),
#   ('Golden Retriever', 0.04),
#   ('Black Lab', 0.01),
#   ...
# ]
```

---

## 🚂 ¿Cómo funciona el entrenamiento?

### Conceptos Previos

#### ¿Qué es una época?
Una época = pasar TODOS los datos de entrenamiento por la red

```
Époc 1: Procesa 1000 imágenes
Época 2: Procesa las mismas 1000 imágenes
...
Época 100: Procesa las mismas 1000 imágenes
```

#### ¿Qué es un batch?
Un lote pequeño de imágenes procesadas juntas

```
Dataset: 10,000 imágenes
Batch size: 16
Batches por época: 10,000 / 16 = 625 batches

Cada batch:
  ├─ 16 imágenes
  ├─ Forward pass
  ├─ Calcular loss
  ├─ Backward pass
  └─ Actualizar pesos
```

### Entrenamiento Paso a Paso

#### 1. Cargar Dataset

```python
class DogBreedDataset(Dataset):
    def __init__(self, image_dir):
        # Escanear carpetas
        # labrador/ → etiqueta 0
        # golden_retriever/ → etiqueta 1
        # ...
        
    def __getitem__(self, idx):
        # Retorna (imagen, etiqueta)
```

#### 2. Crear DataLoader

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=16,        # Procesa 16 imágenes a la vez
    shuffle=True,         # Mezcla aleatoriamente
    num_workers=2         # Carga en paralelo (2 threads)
)
```

#### 3. Inicializar Modelo

```python
model = DogBreedClassifier(num_classes=55, pretrained=True)
model.to(device)  # Mover a GPU
```

**¿Qué es `pretrained=True`?**
- Carga pesos entrenados en ImageNet
- El modelo YA sabe reconocer conceptos visuales
- Solo necesitamos fine-tunear para perros
- **Ventaja**: Converge mucho más rápido

#### 4. Configurar Optimizer

```python
optimizer = AdamW(
    model.parameters(),      # ← Qué parámetros actualizar
    lr=1e-3,                 # Learning rate
    weight_decay=1e-5        # Regularización L2
)
```

**¿Qué es AdamW?**
- Algoritmo para actualizar pesos
- Similar a SGD pero más inteligente
- Adapta el learning rate por parámetro
- W = "Weight decay" (regularización)

#### 5. Configurar Learning Rate Scheduler

```python
scheduler = CosineAnnealingLR(
    optimizer,
    T_max=100  # Número de épocas
)
```

**¿Qué hace?**
```
Learning rate en época 1:   1e-3
Learning rate en época 50:  0.5e-3  (disminuye)
Learning rate en época 100: ~0      (tiende a cero)
```

**¿Por qué?**
- Primeras épocas: LR alto → aprendizaje rápido
- Últimas épocas: LR bajo → ajustes finos

#### 6. Loop de Entrenamiento

```python
for epoch in range(100):
    total_loss = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        # 1. Mover a GPU
        images = images.to(device)
        labels = labels.to(device)
        
        # 2. Forward pass
        optimizer.zero_grad()  # ← Limpiar gradientes previos
        outputs = model(images)  # [16, 55]
        
        # 3. Calcular loss
        loss = criterion(outputs, labels)
        
        # 4. Backward pass
        loss.backward()  # ← Calcular gradientes
        
        # 5. Actualizar pesos
        optimizer.step()  # ← Cambiar pesos basado en gradientes
        
        total_loss += loss.item()
    
    # 6. Validar
    val_loss, val_acc = validate()
    
    # 7. Actualizar learning rate
    scheduler.step()
    
    print(f"Epoch {epoch+1}: Train Loss={total_loss:.4f}, Val Loss={val_loss:.4f}")
```

#### 7. Early Stopping

```python
class EarlyStopping:
    def __call__(self, val_loss, model, epoch):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            # Guardar modelo
            torch.save(model.state_dict(), "best_model.pth")
        else:
            self.counter += 1
        
        return self.counter >= self.patience  # Retorna True si parar
```

**¿Qué hace?**
- Monitorea validación loss
- Si mejora: guarda modelo, reinicia contador
- Si no mejora: incrementa contador
- Si contador > patience (ej: 20): PARAR

**¿Por qué?**
- Evita sobreajuste
- No desperdicia tiempo entrenando sin mejora

---

## 🧠 ¿Qué es EfficientNetB0?

### Historia

```
1. LeNet (1998) → 60K parámetros
2. AlexNet (2012) → 60M parámetros
3. ResNet (2015) → 26M-152M parámetros
4. MobileNet (2017) → 4M parámetros
5. EfficientNet (2019) → 4-1200M parámetros (elegir escala)
```

**EfficientNetB0** = Versión pequeña y rápida

### Arquitectura

```
INPUT: 224×224
  ↓ Stem
  └─ Conv 3×3, stride 2 → 112×112
  ↓ MBConv Blocks (Mobile Inverted Convolutions)
  ├─ Bloque 1: Depthwise Conv 3×3
  ├─ Bloque 2: Pointwise Conv 1×1
  ├─ Bloque 3-18: Más capas similares
  └─ Output: 1×1×1280
  ↓ Head
  └─ Global Average Pooling → [1280]
  ↓ Cabezal personalizado
  └─ [1280] → [55]
```

### ¿Por qué EfficientNetB0?

| Modelo | Parámetros | Velocidad | Accuracy | GPU Memory |
|--------|-----------|----------|----------|-----------|
| MobileNet | 4M | ⚡⚡⚡ | 70% | 512MB |
| **EfficientNetB0** | **4.3M** | **⚡⚡** | **92%** | **1GB** |
| ResNet50 | 25M | ⚡ | 92% | 2GB |
| ResNet152 | 60M | ⚠️ | 93% | 4GB |

**Conclusión**: Mejor balance entre velocidad, memoria y accuracy 🎯

### Características de Optimización

```python
# 1. Backbone congelado inicialmente
for param in model.backbone.features.parameters():
    param.requires_grad = False

# 2. Solo entrenar cabezal (mucho más rápido)
for param in model.backbone.classifier.parameters():
    param.requires_grad = True

# 3. Después descongelar todo
for param in model.parameters():
    param.requires_grad = True
```

---

## 📚 Conceptos Clave

### 1. Transfer Learning

```
Opción A: Entrenar desde cero
Input → Network (inicializado aleatoriamente) → Output
❌ Lento (necesita mucho datos)

Opción B: Transfer Learning
Input → Network (pesos de ImageNet) → Output
✅ Rápido (reutiliza conocimiento)
```

**Proceso:**
1. Cargar modelo preentrenado en ImageNet
2. Congelar primeras capas (características genéricas)
3. Entrenar solo últimas capas (características específicas)
4. Opcionalmente: descongelar y ajustar todo

### 2. Data Augmentation

**Problema:** Solo 55 razas → no suficientes datos
**Solución:** Crear datos sintéticos

```
Original imagen:
  └─ Rotación ±30° → Nueva imagen
  └─ Flip horizontal → Nueva imagen
  └─ Cambiar brillo → Nueva imagen
  └─ Agregar ruido → Nueva imagen
  └─ ...
```

**Resultado:** 1 imagen → 10 imágenes virtuales

**¿Por qué?**
- Modelo más robusto
- Evita sobreajuste
- Mejor generalización

### 3. Mixed Precision (AMP)

**Problema:** GPU requiere mucha memoria
**Solución:** Usar precisión mixta

```
Precisión estándar: float32 (32 bits)
Precisión reducida: float16 (16 bits)

✅ float16: Usa 50% menos memoria
❌ float16: Menos precisión

Solución: Usar ambas
  ├─ Capas importantes: float32
  └─ Capas menos importantes: float16
```

### 4. Gradient Accumulation

**Problema:** Batch size 16 es pequeño (necesitamos 32)
**Solución:** Acumular gradientes

```
Batch 1 (16 imágenes)
  ↓ Forward + Backward
  ↓ Gradientes acumulados
Batch 2 (16 imágenes)
  ↓ Forward + Backward
  ↓ Gradientes acumulados
Batch 3 (16 imágenes)
  ↓ Forward + Backward
  ↓ Actualizar pesos (como si fue batch 48)
```

### 5. Early Stopping

**Problema:** ¿Cuándo parar el entrenamiento?
**Solución:** Monitorear validación loss

```
Epoch 1: Val Loss = 2.5 ✓ (mejora)
Epoch 2: Val Loss = 2.3 ✓ (mejora)
Epoch 3: Val Loss = 2.1 ✓ (mejora)
...
Epoch 50: Val Loss = 1.8 ✓ (mejora)
Epoch 51: Val Loss = 1.9 ✗ (empeora)
Epoch 52: Val Loss = 1.9 ✗ (sin cambio)
...
Epoch 70: Val Loss = 1.9 ✗ (20 épocas sin mejora)
⛔ PARAR (early stopping)
```

---

## 🎓 Resumen de Flujo

### Entrenamiento
```
Datos → Augmentación → Red Neuronal
  ↓
Loss Function (CrossEntropy)
  ↓
Calcular gradientes (backward)
  ↓
Actualizar pesos (optimizer)
  ↓
Repetir 100 épocas o hasta Early Stopping
```

### Predicción
```
Imagen → Preprocesar → Red Neuronal
  ↓
Softmax (convertir a probabilidades)
  ↓
Top-K (mejores predicciones)
  ↓
Mostrar resultados
```

---

**¡Ahora entiendes cómo funciona el código! 🚀**

*Para más información: lee los comentarios en cada archivo.*
