"""
rebuild_model.py
Reconstruye la arquitectura correcta y transfiere los pesos del modelo original.
Ejecutar con: python rebuild_model.py
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf

print(f"TensorFlow : {tf.__version__}")
print(f"Keras      : {tf.keras.__version__}")
print()

MODEL_KERAS = "models/dog_classifier_fixed.keras"
MODEL_OUT   = "models/dog_classifier_rebuilt.h5"
NUM_CLASSES = 55
IMG_SIZE    = 224

# ── 1. Extraer solo los pesos del .keras (sin cargar la arquitectura) ─────────
print("⏳ Extrayendo pesos del modelo original...")

import zipfile, tempfile, shutil

tmp_dir    = tempfile.mkdtemp()
weights_h5 = os.path.join(tmp_dir, "model.weights.h5")

try:
    with zipfile.ZipFile(MODEL_KERAS, "r") as z:
        z.extract("model.weights.h5", tmp_dir)
    print("✅ Pesos extraídos correctamente")
except Exception as e:
    print(f"❌ Error extrayendo pesos: {e}")
    shutil.rmtree(tmp_dir)
    raise

# ── 2. Reconstruir la arquitectura correcta ───────────────────────────────────
print("\n⏳ Reconstruyendo arquitectura EfficientNetB0 + clasificador...")

# Base EfficientNetB0 sin la capa top
base = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights=None,           # no cargar ImageNet, usaremos los pesos guardados
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    pooling=None
)

# Construir el modelo Sequential igual que en tu training.py
model = tf.keras.Sequential([
    # Augmentation (las mismas capas que tenía el modelo original)
    tf.keras.layers.RandomFlip("horizontal", input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomTranslation(0.1, 0.1),

    # Rescaling (EfficientNet espera [0,255], pero si tu modelo usaba /255 descomenta)
    # tf.keras.layers.Rescaling(1./255),

    # Backbone
    base,

    # Pooling
    tf.keras.layers.GlobalAveragePooling2D(),

    # Clasificador
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1024),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation("relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(512),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation("relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(NUM_CLASSES, activation="softmax"),
])

# Build con un forward pass dummy
model.build((None, IMG_SIZE, IMG_SIZE, 3))
print(f"✅ Arquitectura reconstruida — parámetros: {model.count_params():,}")
print(f"   Input  shape: {model.input_shape}")
print(f"   Output shape: {model.output_shape}")

# ── 3. Cargar los pesos ───────────────────────────────────────────────────────
print("\n⏳ Cargando pesos guardados...")
try:
    model.load_weights(weights_h5)
    print("✅ Pesos cargados correctamente")
except Exception as e:
    print(f"❌ Error cargando pesos: {e}")
    print()
    print("Esto puede pasar si la arquitectura del modelo original")
    print("era diferente (ej: sin augmentation layers, con Rescaling, etc.)")
    print("Revisa tu model.py y ajusta las capas en este script.")
    shutil.rmtree(tmp_dir)
    raise

# ── 4. Test de predicción ─────────────────────────────────────────────────────
dummy = np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
pred  = model.predict(dummy, verbose=0)
print(f"✅ Test predicción OK — clases: {pred.shape[1]}")

# ── 5. Guardar ────────────────────────────────────────────────────────────────
print(f"\n💾 Guardando modelo en '{MODEL_OUT}'...")
model.save(MODEL_OUT)
shutil.rmtree(tmp_dir)

print("✅ Modelo guardado exitosamente")
print()
print("━" * 60)
print("Actualiza app.py con la nueva ruta:")
print(f'   "models/dog_classifier_rebuilt.h5"')
print("━" * 60)