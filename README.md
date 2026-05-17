# 🐶 AI Dog Breed Classifier

Aplicación de inteligencia artificial para clasificación automática de razas de perros usando **PyTorch**, **EfficientNetB0** y **Streamlit**.

[![Streamlit](https://img.shields.io/badge/Streamlit-1.42.0-FF4B4B?style=flat-square&logo=streamlit)](https://streamlit.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.2-EE4C2C?style=flat-square&logo=pytorch)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python)](https://www.python.org/)
[![Model](https://img.shields.io/badge/Model-EfficientNetB0-00A0DF?style=flat-square)](https://github.com/lukemelas/EfficientNet-PyTorch)
[![Breeds](https://img.shields.io/badge/Razas-55-4CAF50?style=flat-square)](#-razas-soportadas)

---

## 📋 Tabla de Contenidos

- [🎯 Descripción](#-descripción)
- [✨ Características](#-características)
- [🚀 Uso Rápido](#-uso-rápido)
- [🏗️ Arquitectura](#-arquitectura)
- [📂 Estructura del Proyecto](#-estructura-del-proyecto)
- [🔧 Instalación Detallada](#-instalación-detallada)
- [📊 Flujo del Proyecto](#-flujo-del-proyecto)
- [📈 Resultados](#-resultados)
- [🎓 Tecnologías](#-tecnologías)
- [💡 Cómo Funciona](#-cómo-funciona)
- [🐛 Troubleshooting](#-troubleshooting)

---

## 🎯 Descripción

Este proyecto es una **aplicación web completa** para clasificación de razas de perros usando deep learning. Sube una foto de un perro y el modelo identificará automáticamente la raza con **92%+ de precisión**.

**¿Qué incluye?**
- ✅ Modelo preentrenado optimizado
- ✅ Interfaz Streamlit bilingüe (Español/English)
- ✅ Predicciones Top-5 con confianza
- ✅ Soporte para cámara y archivos
- ✅ Código profesional y documentado

---

## ✨ Características Principales

### 🎯 Predicción Inteligente
- Clasificación automática de **55 razas**
- **Top 5 predicciones** con probabilidades
- **Confianza visual** mediante gráficos
- Interfaz intuitiva y amigable

### 🌍 Multiidioma
- **Español** 🇪🇸
- **English** 🇺🇸
- Selector en sidebar

### 📊 Visualización Interactiva
- Gráficos Plotly en tiempo real
- Barras de confianza
- Información sobre razas

### ⚡ Optimizaciones GPU
- **Mixed Precision** (AMP)
- **EfficientNetB0** ligero y rápido
- **Inferencia <1 segundo**
- Solo **4.3M parámetros**

---

## 🚀 Uso Rápido

### Instalación (2 minutos)
```bash
# 1. Descargar proyecto
cd dog-breed-classifier

# 2. Crear entorno virtual
python -m venv venv
source venv/Scripts/activate  # En Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Ejecutar aplicación
streamlit run app.py
```

### Usar la Aplicación
1. Abre `http://localhost:8501` en tu navegador
2. Selecciona idioma (Español/English) en el sidebar
3. Sube una imagen de un perro o toma una foto con tu cámara
4. ¡La IA clasificará la raza automáticamente!

---

## 🏗️ Arquitectura del Modelo

```
INPUT: 224×224 RGB
       ↓
BACKBONE: EfficientNetB0 (ImageNet pretrained)
       ↓
CLASSIFIER HEAD:
  1280 → 1024 (BatchNorm + ReLU + Dropout)
  1024 → 512  (BatchNorm + ReLU + Dropout)
  512  → 55   (Logits)
       ↓
OUTPUT: 55 razas (softmax para probabilidades)
```

**Características de Optimización:**
- Dropout progresivo: 0.2 → 0.3 → 0.2
- BatchNormalization para estabilidad
- Transfer Learning (pesos ImageNet)
- Total de parámetros: 4.3M

---

## 📂 Estructura del Proyecto

```
dog-breed-classifier/
│
├── 📄 app.py                      ← 🎯 APLICACIÓN PRINCIPAL
│                                     • Interfaz Streamlit
│                                     • Upload/Cámara
│                                     • Visualización de predicciones
│
├── 📁 models/                     ← Modelos preentrenados
│   └── best_efficientnet_b0.pth   ← 🔴 MODELO ACTIVO (92% accuracy)
│
├── 📁 archive/                    ← Scripts de entrenamiento
│   ├── training.py                ← 🚂 Entrenamiento completo
│   ├── predict.py                 ← 🔮 Predicciones avanzadas
│   ├── model.py                   ← 🧠 Arquitectura EfficientNetB0
│   ├── config.py                  ← ⚙️  Configuración centralizada
│   ├── augmentation.py            ← 🎨 Aumentación de datos
│   └── dataset/                   ← 55 razas de perros
│
├── 📄 requirements.txt            ← Dependencias
├── 📄 README.md                   ← Este archivo
└── 📁 venv_new/                   ← Entorno virtual Python
```

---

## 🔧 Instalación Detallada

### Requisitos Previos
- Python 3.11+
- GPU NVIDIA (opcional)
- 2-3 GB de espacio

### Paso a Paso

**A. Descargar**
```bash
git clone https://github.com/usuario/dog-breed-classifier.git
cd dog-breed-classifier
```

**B. Entorno Virtual**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

**C. Instalar Dependencias**
```bash
# CPU
pip install -r requirements.txt

# GPU CUDA (si tienes NVIDIA)
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt --no-deps
```

**D. Ejecutar**
```bash
streamlit run app.py
```

---

## 📊 Flujo del Proyecto

### 1️⃣ **Entrenamiento** (archive/training.py)
```
Dataset (55 razas)
  ↓ Augmentación (rotaciones, flips, luz, ruido)
  ↓ DataLoader (batch=16)
  ↓ EfficientNetB0 + Cabezal Custom
  ↓ AdamW Optimizer + CosineAnnealingLR
  ↓ Gradient Accumulation (simula batch 32)
  ↓ Mixed Precision (AMP)
  ↓ Early Stopping
  ↓ best_efficientnet_b0.pth
```

### 2️⃣ **Predicción** (archive/predict.py)
```
Imagen del usuario
  ↓ Redimensionar a 224×224
  ↓ Normalizar con ImageNet stats
  ↓ Forward pass
  ↓ Softmax
  ↓ Top-5 predicciones
  ↓ Mostrar en Streamlit
```

---

## 📈 Resultados

| Métrica | Valor |
|---------|-------|
| **Top-1 Accuracy** | 92%+ |
| **Top-5 Accuracy** | 98%+ |
| **Parámetros** | 4.3M |
| **Tamaño del Modelo** | 17 MB |
| **Tiempo de Inferencia** | <1 segundo/imagen |
| **GPU Optimizada** | RTX 3050 (4GB VRAM) |

---

## 🎓 Tecnologías

| Librería | Versión | Uso |
|----------|---------|-----|
| **PyTorch** | 2.2.2 | Deep Learning Framework |
| **torchvision** | 0.17.2 | Modelos preentrenados |
| **Streamlit** | 1.42.0 | Interfaz Web |
| **Albumentations** | Latest | Augmentación de datos |
| **Plotly** | 5.24.1 | Gráficos interactivos |
| **NumPy** | 1.26.4 | Operaciones numéricas |
| **Pillow** | 10.1.0 | Procesamiento de imágenes |

---

## 💡 Cómo Funciona

### Entrada
- 📤 Upload de imágenes (JPG, PNG, BMP)
- 📸 Captura con cámara web
- Cualquier tamaño (se redimensiona a 224×224)

### Procesamiento
1. Cargar imagen como RGB
2. Redimensionar a 224×224
3. Normalizar con ImageNet stats
4. Forward pass por EfficientNetB0
5. Aplicar Softmax para probabilidades

### Salida
- 🎯 Predicción principal + confianza
- 🏆 Top 5 razas con barras
- ⚠️ Advertencia si confianza < 70%

---

## 🐾 Razas Soportadas

El modelo clasifica **55 razas de perros**:
Labrador, Golden Retriever, German Sheperd, Bulldog, Beagle, Pug, Husky, Chihuahua, Dachshund, Boxer, y 45 más...

[Ver lista completa en el código]

---

## 🐛 Troubleshooting

### ❌ "No se encontró el modelo"
```bash
# Verifica que el archivo existe
ls models/best_efficientnet_b0.pth
```

### ❌ "Error de GPU/CUDA"
```bash
# Usa CPU en lugar de GPU
# En app.py: device = torch.device("cpu")
```

### ❌ "Predicciones lentas"
- Instala PyTorch con CUDA
- Usa GPU en lugar de CPU

---

## 📚 Archivos Principales

### `app.py` - Aplicación Web
Carga el modelo y muestra interfaz Streamlit para predicciones en tiempo real.

### `archive/training.py` - Entrenamiento
Script para entrenar el modelo desde cero con tu propio dataset.

### `archive/model.py` - Arquitectura
Define la arquitectura EfficientNetB0 + cabezal personalizado.

### `archive/config.py` - Configuración
Centraliza todos los hiperparámetros y rutas del proyecto.

### `archive/augmentation.py` - Aumentación
Pipeline de transformaciones Albumentations para robustez del modelo.

### `archive/predict.py` - Predicciones
Módulo para predicciones individuales, en lote o directorios completos.

---

## 🚀 Despliegue en Streamlit Cloud

1. Sube el proyecto a GitHub
2. Ve a [streamlit.io/cloud](https://streamlit.io/cloud)
3. Click en "New app"
4. Selecciona tu repositorio
5. Main file: `app.py`
6. ¡Deploy!

---

## 🤝 Contribuciones

¡Bienvenidas! Fork → Feature Branch → Pull Request

---

## 📝 Licencia

Código abierto para propósitos educativos.

---

**¡Gracias por usar Dog Breed Classifier! 🐾**

*Versión 1.0 | Mayo 2026*
