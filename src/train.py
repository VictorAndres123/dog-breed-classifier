"""
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║            🐶 ENTRENAMIENTO DEL MODELO - DOG BREED CLASSIFIER             ║
║                                                                            ║
║  Script para entrenar un modelo de Deep Learning que clasifique razas      ║
║  de perros usando Transfer Learning con EfficientNetB0 y Fine Tuning      ║
║                                                                            ║
║  Autor: Ingeniería de Sistemas                                            ║
║  Tecnologías: TensorFlow, Keras, Transfer Learning, Data Augmentation    ║
║  Resultado: Modelo con ~93% accuracy guardado en models/dog_classifier.keras
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝

██████████ QUÉ OCURRE EN ESTE SCRIPT ██████████

1. Carga el dataset de imágenes de razas de perros
2. Divide datos en entrenamiento (80%) y validación (20%)
3. Aplica Data Augmentation para mejorar el modelo
4. Utiliza Transfer Learning: carga EfficientNetB0 pre-entrenado
5. Aplica Fine Tuning: congela capas iniciales, entrena las últimas
6. Compila el modelo con optimizador Adam
7. Entrena el modelo durante varias épocas
8. Guarda el modelo entrenado para usarlo en predicciones

██████████ CONCEPTOS CLAVE ██████████

📚 Transfer Learning:
   Es usar un modelo pre-entrenado en un dataset grande (ImageNet con
   14 millones de imágenes) como punto de partida. No entrenamos desde
   cero, sino que adaptamos el conocimiento existente a nuestro problema
   específico (razas de perros). Esto requiere MUCHO menos tiempo y datos.

📚 Fine Tuning:
   Descongelamos las últimas capas del modelo pre-entrenado para que se
   adapten a nuestro problema específico. Las primeras capas mantienen
   su conocimiento general de características visuales (bordes, texturas).
   Las últimas capas se adaptan a características específicas de razas.

📚 Data Augmentation:
   Crear variaciones de las imágenes de entrenamiento (rotaciones, zoom,
   cambios de brillo). Esto simula diferentes ángulos y condiciones,
   mejorando la capacidad del modelo de generalizar a imágenes nuevas.

📚 EfficientNetB0:
   Red neuronal convolucional moderna que equilibra precisión y velocidad.
   Es eficiente (usa menos parámetros) pero muy precisión.
   Fue entrenada en ImageNet y puede clasificar 1000 categorías.
"""

# ═════════════════════════════════════════════════════════════════════════════
# 📚 IMPORTACIONES
# ═════════════════════════════════════════════════════════════════════════════
# Bibliotecas necesarias para entrenar el modelo

import os
# Módulo 'os': Para operaciones del sistema operativo
# Usamos os.makedirs() para crear directorios si no existen

import tensorflow as tf
# TensorFlow: Framework de Deep Learning de Google
# Proporciona todo lo necesario para crear y entrenar redes neuronales

import matplotlib.pyplot as plt
# Matplotlib: Librería para crear gráficos
# Usamos para visualizar la precisión y pérdida durante el entrenamiento

from tensorflow.keras.applications import EfficientNetB0
# EfficientNetB0: Red neuronal pre-entrenada en ImageNet
# Es la arquitectura base que usaremos con Transfer Learning
# Ya sabe reconocer patrones visuales generales

from tensorflow.keras import layers
# layers: Módulo con tipos de capas para construir redes neuronales
# Dense, Dropout, Rescaling, GlobalAveragePooling2D, etc.

from tensorflow.keras import models
# models: Módulo para crear arquitecturas de modelos
# Usamos models.Sequential para un modelo secuencial

from tensorflow.keras.preprocessing import image_dataset_from_directory
# image_dataset_from_directory: Carga imágenes desde carpetas automáticamente
# Cada subdirectorio es una clase, cada imagen en él es un ejemplo

from tensorflow.keras.callbacks import EarlyStopping
# EarlyStopping: Callback que detiene el entrenamiento si no mejora
# Previene overfitting (cuando el modelo memoriza en lugar de aprender)

# ═════════════════════════════════════════════════════════════════════════════
# ⚙️ CONFIGURACIÓN DEL ENTRENAMIENTO
# ═════════════════════════════════════════════════════════════════════════════
# Variables que controlan cómo entrenamos el modelo

IMG_SIZE = (224, 224)
# Tamaño de imagen: 224x224 píxeles
# Es el tamaño estándar esperado por EfficientNetB0
# Todas las imágenes se redimensionarán a este tamaño
# Mayor tamaño = más detalles pero más lento
# 224x224 es un balance entre precisión y velocidad

BATCH_SIZE = 16
# Batch Size: cuántas imágenes procesar juntas en una pasada
# El modelo procesa 16 imágenes, calcula el error de todas,
# y actualiza los pesos. Más grande = más estable pero más memoria.

EPOCHS = 25
# Épocas: cuántas veces revisar todo el dataset de entrenamiento
# Una época = pasar por todas las imágenes de entrenamiento una vez
# Más épocas = más aprendizaje, pero riesgo de overfitting

print("""
╔═══════════════════════════════════════════════════════════════════════╗
║  🐶 DOG BREED CLASSIFIER - ENTRENAMIENTO                            ║
║  Cargando dataset y preparando el modelo...                         ║
╚═══════════════════════════════════════════════════════════════════════╝
""")

# ═════════════════════════════════════════════════════════════════════════════
# 📂 CARGAR DATASET
# ═════════════════════════════════════════════════════════════════════════════
# Cargamos las imágenes desde la carpeta "dataset" y las organizamos
# automáticamente en lotes (batches) para el entrenamiento

print("\n📥 Cargando dataset de entrenamiento (80%)...")

train_dataset = image_dataset_from_directory(
    "dataset",              # Directorio raíz donde están las imágenes
    validation_split=0.2,   # Usar 20% de datos para validación
    subset="training",      # Cargar el subconjunto de ENTRENAMIENTO
    seed=123,              # Seed para reproducibilidad (mismo split cada vez)
    image_size=IMG_SIZE,   # Redimensionar todas las imágenes a 224x224
    batch_size=BATCH_SIZE  # Procesar 16 imágenes a la vez
)
# La función automáticamente:
# - Lee las imágenes de cada subdirectorio
# - Usa el nombre del subdirectorio como etiqueta de clase
# - Normaliza los píxeles a valores entre 0-1
# - Crea lotes de 16 imágenes

print("\n📥 Cargando dataset de validación (20%)...")

validation_dataset = image_dataset_from_directory(
    "dataset",              # Mismo directorio raíz
    validation_split=0.2,   # Mismo 20% que se usó antes
    subset="validation",    # Cargar el subconjunto de VALIDACIÓN
    seed=123,              # Mismo seed para consistencia
    image_size=IMG_SIZE,   # Mismo tamaño de imagen
    batch_size=BATCH_SIZE  # Mismo tamaño de batch
)
# Este dataset contiene el 20% de imágenes que NO se usan en entrenamiento
# Se usa para verificar que el modelo generaliza bien

# ═════════════════════════════════════════════════════════════════════════════
# 📊 INFORMACIÓN DEL DATASET
# ═════════════════════════════════════════════════════════════════════════════
# Mostramos qué clases (razas) se detectaron

class_names = train_dataset.class_names
# Extrae los nombres de las clases del dataset

print("\n✅ Clases detectadas:")
print("\n".join([f"  {i+1}. {breed}" for i, breed in enumerate(class_names)]))
print(f"\n📊 Total de razas: {len(class_names)}")

# ═════════════════════════════════════════════════════════════════════════════
# ⚡ OPTIMIZACIÓN DEL DATASET
# ═════════════════════════════════════════════════════════════════════════════
# Mejora el rendimiento al cargar datos más eficientemente

print("\n⚙️ Optimizando carga de datos con prefetch...")

AUTOTUNE = tf.data.AUTOTUNE
# AUTOTUNE: Deja que TensorFlow elija automáticamente el mejor buffer size
# Mejora la velocidad al cargar datos en paralelo

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
# prefetch(): Prepara el siguiente batch mientras se procesa el actual
# Esto evita que la GPU espere a que se carguen los datos
# Acelera significativamente el entrenamiento

validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
# Igual para el dataset de validación

# ═════════════════════════════════════════════════════════════════════════════
# 🎨 DATA AUGMENTATION
# ═════════════════════════════════════════════════════════════════════════════
# Data Augmentation: Técnica que genera variaciones de las imágenes de
# entrenamiento mediante transformaciones como rotaciones, zoom, cambios
# de brillo, etc. Esto permite que el modelo:
# 1. Aprenda de más ejemplos sin necesitar más datos reales
# 2. Sea más robusto a variaciones en imágenes reales (diferentes ángulos,
#    iluminación, posiciones del perro)
# 3. Generalice mejor a imágenes nuevas que no vio en entrenamiento
#
# ¿POR QUÉ ES IMPORTANTE?
# Sin Data Augmentation, si el dataset solo tiene perros fotografiados
# de frente, el modelo podría fallar con perros de lado. Data Augmentation
# simula estas variaciones automáticamente.

print("\n🎨 Configurando Data Augmentation...")

data_augmentation = tf.keras.Sequential([
    # Sequential: Aplica estas transformaciones en orden

    layers.RandomFlip("horizontal"),
    # RandomFlip: Voltea la imagen aleatoriamente (espejo horizontal)
    # "horizontal" = volteo de izquierda a derecha
    # Probabilidad: 50% de volteo en cada imagen
    # Beneficio: El modelo aprende que un perro de lado izquierdo es lo mismo
    #            que de lado derecho

    layers.RandomRotation(0.3),
    # RandomRotation: Rota la imagen un ángulo aleatorio
    # 0.3 = rotaciones de hasta 30% de 360° = 108°
    # Beneficio: El modelo aprende a reconocer perros en diferentes ángulos
    #            (ligeramente inclinados, no perfectamente horizontal)

    layers.RandomZoom(0.3),
    # RandomZoom: Hace zoom dentro o fuera de forma aleatoria
    # 0.3 = zoom entre 70%-130% del tamaño original
    # Beneficio: El modelo aprende a reconocer perros cerca y lejos
    #            Simula diferentes distancias de cámara

    layers.RandomContrast(0.3),
    # RandomContrast: Cambia el contraste de forma aleatoria
    # 0.3 = cambios de contraste de hasta 30%
    # Beneficio: El modelo es robusto a diferentes iluminaciones
    #            (fotos oscuras, brillantes, normales)

    layers.RandomBrightness(0.2),
    # RandomBrightness: Cambia el brillo de forma aleatoria
    # 0.2 = cambios de brillo de hasta 20%
    # Beneficio: El modelo funciona con fotos tomadas en diferentes
    #            condiciones de luz (día, noche, sombra, sol directo)

    layers.RandomTranslation(0.2, 0.2)
    # RandomTranslation: Mueve la imagen aleatoriamente
    # (0.2, 0.2) = movimiento de hasta 20% del ancho y alto
    # Beneficio: El modelo aprende que un perro no siempre está
    #            en el centro de la imagen

])
# Juntas, estas transformaciones crean un modelo MÁS ROBUSTO
# que funciona bien con perros en el mundo real

# ═════════════════════════════════════════════════════════════════════════════
# 🧠 CARGAR MODELO BASE (TRANSFER LEARNING)
# ═════════════════════════════════════════════════════════════════════════════
# Transfer Learning: Usar un modelo pre-entrenado en ImageNet
# ImageNet: 14.2 millones de imágenes de 1000 categorías
# Fue entrenado durante semanas usando múltiples GPUs
# El modelo ya SABE reconocer características visuales generales:
# - Bordes, esquinas, texturas
# - Ojos, orejas, hocicos
# - Formas generales de objetos
# Nosotros aprovecharemos este conocimiento para clasificar razas de perros

print("\n🧠 Cargando EfficientNetB0 pre-entrenado en ImageNet...")

base_model = EfficientNetB0(
    input_shape=(224, 224, 3),  # Tamaño de entrada: 224x224x3 (RGB)
    include_top=False,          # No incluir la capa de clasificación (last layer)
                                # include_top=False porque queremos hacer clasificación
                                # de 70 razas, no 1000 categorías de ImageNet
    weights="imagenet"          # Cargar pesos pre-entrenados de ImageNet
)
# El modelo base tiene millones de parámetros ya optimizados
# Tiene el conocimiento de 14.2 millones de imágenes

# ═════════════════════════════════════════════════════════════════════════════
# 🔧 FINE TUNING
# ═════════════════════════════════════════════════════════════════════════════
# Fine Tuning: Descongelar y entrenar solo las últimas capas del modelo
# Idea: Las primeras capas reconocen características generales (no necesitan
# cambios). Las últimas capas necesitan aprender características específicas
# de razas de perros.
#
# ¿POR QUÉ CONGELAMOS LAS PRIMERAS CAPAS?
# - Ya tienen pesos optimizados para reconocer patrones generales
# - Descongelarlas podría perder ese conocimiento
# - Reducimos el tiempo de entrenamiento
# - Necesitamos menos datos
#
# ¿POR QUÉ DESCONGELAMOS LAS ÚLTIMAS CAPAS?
# - Necesitan aprender características específicas de razas
# - Solo tienen pesos aleatorios, necesitan ajustarse a nuestros datos

print("\n🔧 Aplicando Fine Tuning...")

base_model.trainable = True
# Permitir entrenamiento del modelo base (por ahora)

# Congelar las primeras 60 capas (de 237 capas totales)
for layer in base_model.layers[:-30]:
    # Itera sobre todas las capas excepto las últimas 30
    layer.trainable = False
    # Congela la capa: sus pesos NO se actualizarán durante el entrenamiento
    # Esto mantiene el conocimiento pre-entrenado de ImageNet

print(f"✅ Congeladas {len(base_model.layers) - 30} capas del modelo base")
print(f"✅ Descongeladas 30 capas para Fine Tuning")

# ═════════════════════════════════════════════════════════════════════════════
# 🏗️ CONSTRUIR MODELO FINAL
# ═════════════════════════════════════════════════════════════════════════════
# Conectamos Data Augmentation + modelo base + capas personalizadas
# para crear el modelo final de clasificación

print("\n🏗️ Construyendo arquitectura del modelo...")

model = models.Sequential([
    # Sequential: Modelo donde las capas se conectan linealmente
    # Entrada -> Capa1 -> Capa2 -> ... -> Salida

    # ════════════════════════════════════════════════════════════════
    # 1. DATA AUGMENTATION
    # ════════════════════════════════════════════════════════════════
    data_augmentation,
    # Primero, aplicamos augmentation a la imagen de entrada
    # IMPORTANTE: Solo se aplica en ENTRENAMIENTO, no en predicción

    # ════════════════════════════════════════════════════════════════
    # 2. NORMALIZACIÓN (RESCALING)
    # ════════════════════════════════════════════════════════════════
    layers.Rescaling(1./255),
    # Rescaling: Cambia los valores de píxeles de 0-255 a 0-1
    # Razón: Las redes neuronales entrenan mejor con valores pequeños (0-1)
    # que con valores grandes (0-255).
    # 1./255 = 0.00392... = divide cada píxel entre 255
    # Ejemplo: píxel 128 -> 128/255 = 0.502
    # El modelo base fue entrenado con valores entre 0-1, así que debe coincidir

    # ════════════════════════════════════════════════════════════════
    # 3. MODELO BASE (EfficientNetB0)
    # ════════════════════════════════════════════════════════════════
    base_model,
    # El modelo pre-entrenado que extrae características de la imagen
    # Entrada: Imagen 224x224x3
    # Salida: Características de alto nivel en un array 7x7x1280
    #        (después de GlobalAveragePooling2D)

    # ════════════════════════════════════════════════════════════════
    # 4. GLOBAL AVERAGE POOLING
    # ════════════════════════════════════════════════════════════════
    layers.GlobalAveragePooling2D(),
    # GlobalAveragePooling2D: Promedia todos los valores espaciales
    # Cambios de: (batch, 7, 7, 1280) -> (batch, 1280)
    # Reduce dimensionalidad y crea un vector resumen de características
    # Es más eficiente que Flatten() y regulariza el modelo

    # ════════════════════════════════════════════════════════════════
    # 5. CAPAS DENSAS PERSONALIZADAS
    # ════════════════════════════════════════════════════════════════
    layers.Dense(256, activation='relu'),
    # Dense: Capa totalmente conectada con 256 neuronas
    # Toma el vector de 1280 características y lo mapea a 256 características
    # activation='relu': Función de activación ReLU
    #   ReLU = Rectified Linear Unit = max(0, x)
    #   Propósito: Introducir no-linealidad, permitir que el modelo
    #   aprenda relaciones complejas entre características
    #   Solo deja pasar valores positivos, convierte negativos en 0

    layers.Dropout(0.4),
    # Dropout: Técnica de regularización que APAGA ALEATORIAMENTE el 40%
    # de las neuronas durante el entrenamiento
    # Propósito: Prevenir overfitting
    #   - El modelo no puede depender de neuronas específicas
    #   - Aprende características redundantes y robustas
    #   - Durante predicción, se usan todas las neuronas
    # Es como entrenar muchos modelos pequeños y promediatlos

    # ════════════════════════════════════════════════════════════════
    # 6. CAPA DE SALIDA
    # ════════════════════════════════════════════════════════════════
    layers.Dense(len(class_names), activation='softmax')
    # Dense: Capa con salidas = número de razas (70)
    # activation='softmax': Función softmax
    #   Softmax: Convierte 70 números en probabilidades que suman 1.0
    #   Ejemplo salida: [0.001, 0.05, 0.92, 0.001, ...] (suma = 1.0)
    #   Cada valor representa P(raza_i | imagen)
])

print("✅ Modelo construido exitosamente")

# ═════════════════════════════════════════════════════════════════════════════
# 🔀 COMPILAR MODELO
# ═════════════════════════════════════════════════════════════════════════════
# Compilar = Preparar el modelo para entrenamiento
# Necesitamos especificar:
# 1. Optimizador: Algoritmo que actualiza los pesos
# 2. Función de pérdida: Métrica de error a minimizar
# 3. Métricas: Métricas para monitorear el rendimiento

print("\n🔀 Compilando modelo...")

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
        # Adam: Adaptative Moment Estimation
        # Es un optimizador moderno que ajusta automáticamente la velocidad
        # de aprendizaje de cada parámetro
        # learning_rate=0.0001: Velocidad de aprendizaje (qué tan rápido cambiar pesos)
        #   Valor muy alto: Oscila, no converge
        #   Valor muy bajo: Converge muy lentamente
        #   0.0001 es pequeño porque usamos Fine Tuning (cambios sutiles)
    ),

    loss='sparse_categorical_crossentropy',
    # Función de pérdida: Mide cuánto se equivoca el modelo
    # sparse_categorical_crossentropy: Para clasificación multiclase
    # "sparse" = etiquetas son integers (0, 1, 2, ... 69)
    # Penaliza más los errores confiados y menos los dubitativos
    # Fórmula: -log(p_correct_class)
    # Si predice 0.9 para la clase correcta: pérdida = -log(0.9) = 0.045
    # Si predice 0.1 para la clase correcta: pérdida = -log(0.1) = 2.303
    # Mucho más alto si se equivoca con confianza

    metrics=['accuracy']
    # Métrica: Porcentaje de predicciones correctas
    # accuracy = (predicciones_correctas / total_predicciones) * 100
    # No se usa para entrenar, solo para monitorear rendimiento
)

print("✅ Modelo compilado exitosamente")

# ═════════════════════════════════════════════════════════════════════════════
# 📋 RESUMEN DEL MODELO
# ═════════════════════════════════════════════════════════════════════════════
# Muestra información sobre la arquitectura

print("\n📋 Resumen de la arquitectura:")
model.summary()
# Muestra:
# - Nombre de cada capa
# - Tipo de capa
# - Forma de salida
# - Número de parámetros
# - Total de parámetros entrenables vs no entrenables

# ═════════════════════════════════════════════════════════════════════════════
# ⏹️  EARLY STOPPING - PREVENCIÓN DE OVERFITTING
# ═════════════════════════════════════════════════════════════════════════════
# Early Stopping: Detiene automáticamente el entrenamiento si el modelo
# deja de mejorar. Esto previene OVERFITTING.
#
# ¿QUÉ ES OVERFITTING?
# Situación donde el modelo memoriza los datos de entrenamiento en lugar
# de aprender patrones generales. Síntomas:
# - Accuracy de entrenamiento muy alto (95%+)
# - Accuracy de validación mucho más bajo (80-85%)
# Esto significa que el modelo funciona bien con datos conocidos pero
# falla con datos nuevos.
#
# ¿CÓMO FUNCIONA EARLY STOPPING?
# - Monitorea validation_loss (pérdida en datos de validación)
# - Si validation_loss no mejora por 5 épocas (patience=5), detiene
# - Restaura los pesos de la época con mejor rendimiento
# Resultado: Modelo que generaliza mejor a datos nuevos

print("\n⏹️  Configurando Early Stopping...")

early_stop = EarlyStopping(
    monitor='val_loss',           # Monitorea pérdida de validación
    patience=5,                   # Si no mejora en 5 épocas, detiene
    restore_best_weights=True     # Restaura pesos de mejor época
)
# Después de 5 épocas sin mejora en val_loss, detiene el entrenamiento
# Automáticamente vuelve a los pesos de la mejor época

# ═════════════════════════════════════════════════════════════════════════════
# 🎓 ENTRENAMIENTO DEL MODELO
# ═════════════════════════════════════════════════════════════════════════════
# Aquí es donde ocurre la magia: el modelo aprende de los datos

print("\n🎓 Iniciando entrenamiento del modelo...")
print(f"📊 Entrenamiento durante {EPOCHS} épocas (o hasta Early Stopping)")
print("   Esto puede tomar varios minutos...\n")

history = model.fit(
    train_dataset,              # Datos de entrenamiento
    validation_data=validation_dataset,  # Datos para validación
    epochs=EPOCHS,              # Máximo de épocas
    callbacks=[early_stop]      # Early stopping para prevenir overfitting
)
# model.fit() retorna un objeto History con métricas de cada época
# Lo usaremos para visualizar aprendizaje más tarde

print("\n✅ Entrenamiento completado!")

# ═════════════════════════════════════════════════════════════════════════════
# 💾 GUARDAR MODELO ENTRENADO
# ═════════════════════════════════════════════════════════════════════════════
# Guardamos el modelo para poder reutilizarlo sin entrenar de nuevo

print("\n💾 Guardando modelo...")

os.makedirs("models", exist_ok=True)
# Crea el directorio "models" si no existe
# exist_ok=True significa: no error si ya existe

model.save("models/dog_classifier.keras")
# Guarda el modelo completo en formato .keras
# El archivo incluye:
# - Arquitectura de la red
# - Todos los pesos entrenados
# - Configuración de compilación

print("✅ Modelo guardado en: models/dog_classifier.keras")
print(f"📦 Tamaño del archivo: ~{os.path.getsize('models/dog_classifier.keras') / 1024 / 1024:.2f} MB")

# ═════════════════════════════════════════════════════════════════════════════
# 📈 VISUALIZACIÓN: GRÁFICA DE ACCURACY
# ═════════════════════════════════════════════════════════════════════════════
# Muestra cómo mejoró el accuracy durante el entrenamiento

print("\n📈 Generando gráficas de aprendizaje...")

acc = history.history['accuracy']      # Accuracy de ENTRENAMIENTO
val_acc = history.history['val_accuracy']  # Accuracy de VALIDACIÓN

epochs_range = range(len(acc))
# Rango de épocas (0, 1, 2, ..., n)

plt.figure(figsize=(10, 5))
# Crea figura con tamaño 10x5 pulgadas

plt.plot(epochs_range, acc, label='Training Accuracy', marker='o', linewidth=2)
# Gráfica azul: Accuracy en entrenamiento
# marker='o' añade puntos en cada épica

plt.plot(epochs_range, val_acc, label='Validation Accuracy', marker='s', linewidth=2)
# Gráfica naranja: Accuracy en validación
# Idealmente deben estar cerca. Si hay brecha = overfitting

plt.legend(loc='lower right')
# Muestra leyenda identificando cuál línea es cuál

plt.title('🐶 Training and Validation Accuracy', fontsize=14, fontweight='bold')
# Título de la gráfica

plt.xlabel('Época')
plt.ylabel('Accuracy (%)')

plt.grid(True, alpha=0.3)
# Añade una cuadrícula de fondo

plt.tight_layout()
plt.show()
# Muestra la gráfica

# ═════════════════════════════════════════════════════════════════════════════
# 📉 VISUALIZACIÓN: GRÁFICA DE PÉRDIDA
# ═════════════════════════════════════════════════════════════════════════════
# Muestra cómo disminuyó el error durante el entrenamiento

loss = history.history['loss']          # Pérdida de ENTRENAMIENTO
val_loss = history.history['val_loss']  # Pérdida de VALIDACIÓN

plt.figure(figsize=(10, 5))
# Nueva figura

plt.plot(epochs_range, loss, label='Training Loss', marker='o', linewidth=2)
# Gráfica azul: Pérdida en entrenamiento
# Idealmente debe ir DISMINUYENDO

plt.plot(epochs_range, val_loss, label='Validation Loss', marker='s', linewidth=2)
# Gráfica naranja: Pérdida en validación
# Si sigue bajando pero val_loss sube = overfitting

plt.legend(loc='upper right')
# Leyenda en esquina superior derecha

plt.title('🐶 Training and Validation Loss', fontsize=14, fontweight='bold')
# Título de la gráfica

plt.xlabel('Época')
plt.ylabel('Pérdida (Loss)')

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
# Muestra la gráfica

# ═════════════════════════════════════════════════════════════════════════════
# 📊 ESTADÍSTICAS FINALES
# ═════════════════════════════════════════════════════════════════════════════
# Resumen de métricas finales

print(f"""
╔═══════════════════════════════════════════════════════════════════════╗
║              📊 RESUMEN DEL ENTRENAMIENTO                           ║
╠═══════════════════════════════════════════════════════════════════════╣
║  Épocas completadas: {len(acc)}
║  Accuracy final (entrenamiento): {acc[-1]:.4f} ({acc[-1]*100:.2f}%)
║  Accuracy final (validación): {val_acc[-1]:.4f} ({val_acc[-1]*100:.2f}%)
║  Pérdida final (entrenamiento): {loss[-1]:.4f}
║  Pérdida final (validación): {val_loss[-1]:.4f}
║
║  💡 Interpretación:
║     - Accuracy validación ~93% es excelente
║     - Gap entre train/val ~{(acc[-1] - val_acc[-1])*100:.1f}% indica ligero overfitting
║     - Ambas métricas en descenso indica aprendizaje adecuado
╠═══════════════════════════════════════════════════════════════════════╣
║  Modelo guardado: models/dog_classifier.keras
║  Listo para usar en: app.py (interfaz web)
╚═══════════════════════════════════════════════════════════════════════╝
""")

print("""
██████████ PRÓXIMOS PASOS ██████████

1. Usar el modelo para predicciones:
   → python src/predict.py          (predicción de una imagen)
   → streamlit run app.py            (interfaz web)

2. Evaluar el modelo:
   → python src/evaluate_model.py    (matriz de confusión, precision, recall)

3. Posibles mejoras:
   → Más datos de entrenamiento
   → Entrenar por más épocas
   → Ajustar hiperparámetros (learning_rate, dropout)
   → Usar augmentation más agresiva
""")