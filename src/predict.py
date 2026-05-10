"""
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║         🐶 PREDICCIÓN CON IMAGEN - DOG BREED CLASSIFIER                  ║
║                                                                            ║
║  Script para hacer predicción de una SOLA imagen usando el modelo         ║
║  entrenado. Muestra la predicción y el nivel de confianza.                ║
║                                                                            ║
║  Uso: python src/predict.py                                              ║
║  (Modifica el archivo para cambiar la imagen a predecir)                  ║
║                                                                            ║
║  Autor: Ingeniería de Sistemas                                            ║
║  Tecnologías: TensorFlow, Keras, PIL, Matplotlib                         ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝

██████████ CÓMO FUNCIONA ESTE SCRIPT ██████████

1. Carga el modelo pre-entrenado
2. Define la lista de razas (clases)
3. Carga una imagen de prueba
4. Preprocesa la imagen (redimensiona, normaliza)
5. Hace predicción con el modelo
6. Obtiene la raza predicha y confianza
7. Muestra la imagen con el resultado

██████████ PREPROCESAMIENTO DE IMÁGENES ██████████

El procesamiento de una imagen se divide en pasos:

1. CARGAR IMAGEN:
   - Lee el archivo de imagen desde disco
   - PIL (Pillow) maneja diferentes formatos (JPG, PNG, etc.)

2. REDIMENSIONAR:
   - Cambia el tamaño a 224x224
   - EfficientNetB0 requiere exactamente este tamaño
   - Se puede estirar, comprimir o con padding

3. CONVERTIR A ARRAY:
   - PIL Image -> NumPy array
   - Array de forma (224, 224, 3)
   - Valores de píxeles: 0-255

4. NORMALIZAR:
   - El modelo fue entrenado con píxeles normalizados (0-1)
   - Dividimos entre 255
   - Esto evita inestabilidad numérica

5. EXPANDIR DIMENSIÓN DE BATCH:
   - Array (224, 224, 3) -> (1, 224, 224, 3)
   - El primer 1 es batch_size
   - El modelo espera un lote, incluso si es una sola imagen

6. PREDICCIÓN:
   - Pasar por el modelo
   - Obtener 70 probabilidades
   - Encontrar la máxima (np.argmax)
"""

# ═════════════════════════════════════════════════════════════════════════════
# 📚 IMPORTACIONES
# ═════════════════════════════════════════════════════════════════════════════

import tensorflow as tf
# TensorFlow: Para cargar el modelo y hacer predicciones

import numpy as np
# NumPy: Para operaciones con arrays
# Usamos np.expand_dims y np.argmax

from tensorflow.keras.preprocessing import image
# image: Módulo de TensorFlow con utilidades de procesamiento de imágenes
# - image.load_img: Carga imagen con PIL
# - image.img_to_array: Convierte PIL Image a NumPy array

import matplotlib.pyplot as plt
# Matplotlib: Para visualizar la imagen y resultado

# ═════════════════════════════════════════════════════════════════════════════
# 🧠 CARGAR MODELO ENTRENADO
# ═════════════════════════════════════════════════════════════════════════════

print("""
╔═══════════════════════════════════════════════════════════════════════╗
║  🐶 PREDICCIÓN - DOG BREED CLASSIFIER                               ║
║  Cargando modelo...                                                  ║
╚═══════════════════════════════════════════════════════════════════════╝
""")

print("\n🧠 Cargando modelo entrenado...")

model = tf.keras.models.load_model(
    "models/dog_classifier.keras"
)
# Carga el modelo con todos sus pesos optimizados

print("✅ Modelo cargado exitosamente")

# ═════════════════════════════════════════════════════════════════════════════
# 📋 DEFINIR CLASES (RAZAS DE PERROS)
# ═════════════════════════════════════════════════════════════════════════════
# Lista de las 10 razas que el modelo puede identificar
# Este script usa un subconjunto, pero el modelo completo puede 70

class_names = [
    "beagle",                # Pequeño perro de caza, orejas largas
    "boxer",                 # Perro grande y musculoso
    "bulldog",              # Cabeza grande, cara plana
    "dachshund",            # Perro pequeño y largo ("salchicha")
    "german_shepherd",      # Gran perro pastor alemán
    "golden_retriever",     # Perro grande y amigable
    "labrador_retriever",   # Perro grande, excelente nadador
    "poodle",              # Perro inteligente, pelaje rizado
    "rottweiler",          # Perro grande y protector
    "yorkshire_terrier"    # Perro pequeño con pelaje largo
]

print(f"\n📚 Clases (razas) que puede identificar: {len(class_names)}")
for i, breed in enumerate(class_names, 1):
    print(f"   {i:2d}. {breed.replace('_', ' ').title()}")

# ═════════════════════════════════════════════════════════════════════════════
# 🖼️  CARGAR IMAGEN PARA PREDECIR
# ═════════════════════════════════════════════════════════════════════════════
# Especifica aquí la ruta de la imagen que deseas predecir

print("\n🖼️  Cargando imagen...")

img_path = "test2.jpg"
# Ruta de la imagen a predecir
# NOTA: Cambia esta ruta para probar con diferentes imágenes
# Ejemplo: img_path = "test.jpg", "test5.jpg", "mi_perro.jpg", etc.

print(f"   Imagen: {img_path}")

# ═════════════════════════════════════════════════════════════════════════════
# 🔄 PREPROCESAMIENTO DE LA IMAGEN
# ═════════════════════════════════════════════════════════════════════════════
# Prepara la imagen exactamente como se prepararon en entrenamiento

print("\n🔄 Preprocesando imagen...")

# ────────────────────────────────────────────────────────────────────────────
# 1. CARGAR IMAGEN
# ────────────────────────────────────────────────────────────────────────────

img = image.load_img(
    img_path,               # Ruta del archivo
    target_size=(224, 224)  # Redimensionar a 224x224
)
# image.load_img():
#   - Lee imagen desde archivo (soporta JPG, PNG, BMP, etc.)
#   - Convierte a RGB si necesario
#   - target_size: Redimensiona automáticamente a 224x224
#   - Retorna objeto PIL Image

print("   ✓ Imagen cargada")

# ────────────────────────────────────────────────────────────────────────────
# 2. CONVERTIR A ARRAY NUMÉRICO
# ────────────────────────────────────────────────────────────────────────────

img_array = image.img_to_array(img)
# image.img_to_array():
#   - Convierte PIL Image a NumPy array
#   - Forma: (224, 224, 3)
#   - Valores: 0-255 (sin normalizar aún)
#   - Tipo de dato: float32

print("   ✓ Array creado: forma", img_array.shape)

# ────────────────────────────────────────────────────────────────────────────
# 3. EXPANDIR DIMENSIÓN DE BATCH
# ────────────────────────────────────────────────────────────────────────────

img_array = np.expand_dims(img_array, axis=0)
# np.expand_dims(array, axis=0):
#   - Agrega una nueva dimensión al principio
#   - Cambio: (224, 224, 3) -> (1, 224, 224, 3)
#   - El 1 representa: batch_size = 1 (una sola imagen)
#   - El modelo espera un lote de imágenes (incluso si es una)
#
# ¿POR QUÉ?
# - Durante entrenamiento procesamos lotes de 16 imágenes
# - El modelo espera un tensor 4D: (batch, height, width, channels)
# - Para una imagen, batch=1

print("   ✓ Dimensión de batch expandida: forma", img_array.shape)

# ════════════════════════════════════════════════════════════════════════════
# 🔮 HACER PREDICCIÓN
# ════════════════════════════════════════════════════════════════════════════
# Enviamos la imagen preprocesada al modelo

print("\n🔮 Haciendo predicción...")

prediction = model.predict(img_array)
# model.predict(img_array):
#   - Pasa la imagen por todas las capas del modelo
#   - Retorna un array de forma (1, 10) o (1, 70)
#   - Cada valor es la probabilidad de una raza
#   - Los valores están entre 0-1 y suman 1.0
#
# Ejemplo de salida:
# [[0.001, 0.05, 0.92, 0.001, 0.005, ...]]
#    ↑     ↑    ↑    ↑    ↑
#    raza0 raza1 raza2 raza3 raza4 ...

print("   ✓ Predicción realizada")

# ════════════════════════════════════════════════════════════════════════════
# 📊 PROCESAR RESULTADOS
# ════════════════════════════════════════════════════════════════════════════

# ────────────────────────────────────────────────────────────────────────────
# Obtener índice de mayor probabilidad
# ────────────────────────────────────────────────────────────────────────────

predicted_index = np.argmax(prediction)
# np.argmax(prediction):
#   - Encuentra el índice del valor máximo
#   - Si prediction = [[0.001, 0.05, 0.92, 0.001, ...]]
#   - Retorna: 2 (el índice de 0.92)

print(f"   ✓ Índice predicho: {predicted_index}")

# ────────────────────────────────────────────────────────────────────────────
# Obtener nombre de la raza
# ────────────────────────────────────────────────────────────────────────────

predicted_class = class_names[predicted_index]
# Usa el índice para obtener el nombre de la raza
# Ejemplo: class_names[2] = "bulldog"

print(f"   ✓ Raza predicha: {predicted_class}")

# ────────────────────────────────────────────────────────────────────────────
# Obtener confianza de la predicción
# ────────────────────────────────────────────────────────────────────────────

confidence = np.max(prediction)
# np.max(prediction):
#   - Obtiene el valor máximo del array
#   - Este es el nivel de confianza
#   - Si prediction = [[0.001, 0.05, 0.92, 0.001, ...]]
#   - Retorna: 0.92
#   - Significado: 92% confianza en que es bulldog

print(f"   ✓ Confianza: {confidence:.4f} ({confidence*100:.2f}%)")

# ════════════════════════════════════════════════════════════════════════════
# 🎨 VISUALIZAR RESULTADOS
# ════════════════════════════════════════════════════════════════════════════
# Muestra la imagen con la predicción

print("\n🎨 Mostrando resultado...")

# Crear figura
plt.figure(figsize=(10, 6))

# Mostrar imagen
plt.imshow(img)

# Crear título con la predicción
titulo = (
    f"🐶 Predicción: {predicted_class.replace('_', ' ').title()}\n"
    f"Confianza: {confidence:.2%}"
)

plt.title(titulo, fontsize=14, fontweight='bold', pad=20)

# Ocultar ejes
plt.axis("off")

plt.tight_layout()
plt.show()

# ════════════════════════════════════════════════════════════════════════════
# 📋 MOSTRAR EN CONSOLA
# ════════════════════════════════════════════════════════════════════════════

print(f"""
╔═══════════════════════════════════════════════════════════════════════╗
║                     ✅ RESULTADO DE PREDICCIÓN                      ║
╠═══════════════════════════════════════════════════════════════════════╣
║
║  🖼️  Imagen: {img_path}
║
║  🐕 Raza Predicha: {predicted_class.replace('_', ' ').title()}
║
║  📊 Confianza: {confidence:.2%} ({confidence*100:.2f}%)
║
║  💡 Interpretación:
║     {self._interpretar_confianza(confidence)}
║
╚═══════════════════════════════════════════════════════════════════════╝
""")

def _interpretar_confianza(confidence):
    if confidence > 0.90:
        return "Muy alta confianza. La predicción es muy probable que sea correcta."
    elif confidence > 0.75:
        return "Alta confianza. La predicción es probablemente correcta."
    elif confidence > 0.50:
        return "Confianza moderada. La predicción puede ser correcta."
    else:
        return "Baja confianza. La predicción es poco confiable."

print(_interpretar_confianza(confidence))

# ════════════════════════════════════════════════════════════════════════════
# 🏆 TOP 5 PREDICCIONES
# ════════════════════════════════════════════════════════════════════════════
# Mostrar las 5 razas más probables

print("\n🏆 Top 5 predicciones:")
print("   " + "-" * 50)

# Obtener índices ordenados (de menor a mayor probabilidad)
top_5_indices = prediction[0].argsort()[-5:][::-1]
# prediction[0]: Toma el primer lote (única imagen)
# .argsort(): Ordena índices de menor a mayor probabilidad
# [-5:]: Toma los últimos 5 (los más altos)
# [::-1]: Invierte para tener de mayor a menor

for i, idx in enumerate(top_5_indices, 1):
    breed = class_names[idx]
    prob = float(prediction[0][idx] * 100)
    
    print(f"   {i}. {breed.replace('_', ' ').title():20} → {prob:6.2f}%")

print("   " + "-" * 50)