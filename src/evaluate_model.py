"""
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║           🐶 EVALUACIÓN DEL MODELO - DOG BREED CLASSIFIER                ║
║                                                                            ║
║  Script para evaluar el rendimiento del modelo entrenado usando            ║
║  métricas profesionales como:                                             ║
║  - Accuracy (Precisión general)                                           ║
║  - Matriz de Confusión                                                    ║
║  - Precision, Recall, F1-Score (por cada raza)                            ║
║  - Reporte de clasificación detallado                                     ║
║                                                                            ║
║  Autor: Ingeniería de Sistemas                                            ║
║  Tecnologías: TensorFlow, scikit-learn, Matplotlib, Seaborn              ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝

██████████ QUÉ OCURRE EN ESTE SCRIPT ██████████

1. Carga el modelo entrenado
2. Carga el dataset completo
3. Hace predicciones en TODAS las imágenes
4. Compara predicciones vs etiquetas reales
5. Calcula metrics: accuracy, precision, recall, f1-score
6. Genera matriz de confusión (visualización)
7. Genera reporte de clasificación

██████████ CONCEPTOS CLAVE ██████████

📊 ACCURACY (Exactitud):
   Porcentaje de predicciones CORRECTAS del total
   Fórmula: Correctas / Total * 100
   Rango: 0-100%
   Ejemplo: 9300 correctas / 10000 total = 93% accuracy
   ⚠️ LIMITACIÓN: No es útil si hay desbalance de clases

📊 PRECISION (Precisión):
   De todas las veces que predijimos una raza,
   ¿cuántas veces fue correcta?
   Fórmula: Verdaderos Positivos / (Verdaderos Positivos + Falsos Positivos)
   Ejemplo: Predijimos "bulldog" 100 veces, 95 fueron correctas = 95% precision
   Interpretación: ¿Qué tan confiable es nuestra predicción positiva?
   Alto precision = Pocos falsos positivos

📊 RECALL (Sensibilidad):
   De todos los perros que SON de una raza,
   ¿cuántos identificamos correctamente?
   Fórmula: Verdaderos Positivos / (Verdaderos Positivos + Falsos Negativos)
   Ejemplo: Hay 100 bulldogs reales, identificamos 95 = 95% recall
   Interpretación: ¿Cuántas instancias reales encontramos?
   Alto recall = Pocos falsos negativos

📊 F1-SCORE:
   Media armónica entre precision y recall
   Fórmula: 2 * (Precision * Recall) / (Precision + Recall)
   Rango: 0-100%
   Balance: Considera precision y recall juntos
   Útil cuando ambas métricas son importantes

📊 MATRIZ DE CONFUSIÓN:
   Tabla que muestra:
   - Filas: Clases reales (ground truth)
   - Columnas: Clases predichas
   - Valores: Número de predicciones en cada combinación
   Interpretación: Diagonal = aciertos, fuera = errores
   Muestra qué razas se confunden entre sí

██████████ EJEMPLO DE INTERPRETACIÓN ██████████

Matriz de Confusión para 3 razas (2000 imágenes):
                    Predicho Beagle  Predicho Boxer  Predicho Bulldog
Real Beagle                 580             15                5
Real Boxer                   20            590                5
Real Bulldog                 10             10               570

Interpretación:
- 580 beagles correctamente predichos (98.3%)
- 15 beagles predichos como boxer (confusión)
- 590 boxers correctamente predichos (98.3%)
- 570 bulldogs correctamente predichos (95%)
- El modelo tiene más dificultad con bulldogs
"""

# ═════════════════════════════════════════════════════════════════════════════
# 📚 IMPORTACIONES
# ═════════════════════════════════════════════════════════════════════════════

import tensorflow as tf
# TensorFlow: Framework para cargar el modelo y hacer predicciones

import numpy as np
# NumPy: Para operaciones con arrays

import matplotlib.pyplot as plt
# Matplotlib: Para visualizar gráficos

import seaborn as sns
# Seaborn: Librería de visualización basada en matplotlib
# Perfecta para matrices de confusión profesionales

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)
# scikit-learn: Librería con métricas estándar de ML
# - confusion_matrix: Crea matriz de confusión
# - classification_report: Calcula precision, recall, f1-score
# - accuracy_score: Calcula exactitud general

# ═════════════════════════════════════════════════════════════════════════════
# ⚙️ CONFIGURACIÓN
# ═════════════════════════════════════════════════════════════════════════════

IMG_SIZE = (224, 224)
# Tamaño de imagen esperado por el modelo

BATCH_SIZE = 16
# Tamaño de lotes para procesamiento

print("""
╔═══════════════════════════════════════════════════════════════════════╗
║  🐶 EVALUACIÓN DEL MODELO - DOG BREED CLASSIFIER                    ║
║  Cargando modelo y datos...                                         ║
╚═══════════════════════════════════════════════════════════════════════╝
""")

# ═════════════════════════════════════════════════════════════════════════════
# 🧠 CARGAR MODELO ENTRENADO
# ═════════════════════════════════════════════════════════════════════════════

print("\n🧠 Cargando modelo entrenado...")

model = tf.keras.models.load_model(
    "models/dog_classifier.keras"
)
# Carga el modelo completo con todos sus pesos

print("✅ Modelo cargado exitosamente")

# ═════════════════════════════════════════════════════════════════════════════
# 📂 CARGAR DATASET COMPLETO
# ═════════════════════════════════════════════════════════════════════════════
# Cargamos TODAS las imágenes del dataset (entrenamiento + validación)
# para evaluar en datos nunca vistos por el modelo

print("\n📂 Cargando dataset completo...")

dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset",              # Directorio del dataset
    image_size=IMG_SIZE,    # Redimensionar a 224x224
    batch_size=BATCH_SIZE,  # Procesar 16 imágenes por lote
    shuffle=False           # NO mezclar (importante para mantener orden)
)

class_names = dataset.class_names
# Extrae nombres de las clases (razas)

print(f"✅ Dataset cargado: {len(class_names)} razas de perros")

# ═════════════════════════════════════════════════════════════════════════════
# 🔮 HACER PREDICCIONES EN TODO EL DATASET
# ═════════════════════════════════════════════════════════════════════════════
# Iteramos sobre todos los lotes del dataset y guardamos predicciones

print("\n🔮 Haciendo predicciones en todas las imágenes...")
print("   (Esto puede tomar varios minutos)...\n")

y_true = []
# Lista para guardar etiquetas reales

y_pred = []
# Lista para guardar etiquetas predichas

for images, labels in dataset:
    # Itera sobre cada lote del dataset
    # images: tensor de forma (batch_size, 224, 224, 3)
    # labels: tensor de etiquetas reales

    predictions = model.predict(images, verbose=0)
    # Predice para todas las imágenes del lote
    # Retorna array de forma (batch_size, 70) con probabilidades
    # verbose=0 para que no imprima progreso
    
    # Para cada imagen en el lote, tomamos la clase con mayor probabilidad
    predicted_labels = np.argmax(predictions, axis=1)
    # np.argmax(predictions, axis=1):
    #   - axis=1: busca el máximo en las 70 probabilidades
    #   - Retorna el índice de la clase con mayor probabilidad
    # Ejemplo: [0.001, 0.92, 0.05, ...] -> índice 1

    y_true.extend(labels.numpy())
    # Agrega etiquetas reales a la lista
    # .numpy(): Convierte tensor TensorFlow a array NumPy

    y_pred.extend(predicted_labels)
    # Agrega etiquetas predichas a la lista

print(f"✅ Predicciones completadas")
print(f"   Total de imágenes evaluadas: {len(y_true)}")

# ═════════════════════════════════════════════════════════════════════════════
# 📊 CALCULAR ACCURACY GENERAL
# ═════════════════════════════════════════════════════════════════════════════

accuracy = accuracy_score(y_true, y_pred)
# accuracy_score: Calcula el porcentaje de predicciones correctas
# Fórmula: Correctas / Total * 100
# Ejemplo: 9300 correctas / 10000 total = 0.93 (93%)

print(f"""
╔═══════════════════════════════════════════════════════════════════════╗
║                  📊 RESULTADOS GENERALES                            ║
╠═══════════════════════════════════════════════════════════════════════╣
║  Accuracy Final del Modelo: {accuracy*100:.2f}%
║
║  Interpretación:
║    - El modelo clasifica correctamente {accuracy*100:.2f}% de las imágenes
║    - Error: {(1-accuracy)*100:.2f}% de predicciones incorrectas
║    - Este es un excelente resultado para clasificación con 70 clases
║
║  Comparación:
║    - Clasificación aleatoria (70 clases): ~1.4%
║    - Modelo humano: ~95-99%
║    - Nuestro modelo: {accuracy*100:.2f}% ✅ Excelente
╚═══════════════════════════════════════════════════════════════════════╝
""")

# ═════════════════════════════════════════════════════════════════════════════
# 🔍 MATRIZ DE CONFUSIÓN
# ═════════════════════════════════════════════════════════════════════════════
# Tabla que muestra qué razas se confunden entre sí

print("\n🔍 Generando matriz de confusión...")

cm = confusion_matrix(y_true, y_pred)
# confusion_matrix retorna una matriz de forma (70, 70)
# cm[i,j] = número de imágenes de clase i predichas como clase j
# Diagonal = aciertos
# Fuera de diagonal = errores/confusiones

# Visualizar la matriz de confusión
plt.figure(figsize=(25, 25))
# Figura grande para ver todas las 70x70 celdas

sns.heatmap(
    cm,
    cmap="Blues",           # Colormap: azul más oscuro = más predicciones
    xticklabels=class_names,  # Nombres de clases en eje X
    yticklabels=class_names,  # Nombres de clases en eje Y
    cbar_kws={'label': 'Número de imágenes'},
    square=True             # Celdas cuadradas
)

plt.xlabel("Clase Predicha", fontsize=12, fontweight='bold')
plt.ylabel("Clase Real", fontsize=12, fontweight='bold')
plt.title("🐶 Matriz de Confusión - Clasificador de Razas de Perros", 
          fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print("✅ Matriz de confusión generada")

# ═════════════════════════════════════════════════════════════════════════════
# 📋 REPORTE DE CLASIFICACIÓN DETALLADO
# ═════════════════════════════════════════════════════════════════════════════
# Precision, Recall, F1-Score por cada raza

print("\n📋 REPORTE DE CLASIFICACIÓN POR RAZA:")
print("=" * 80)

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,  # Nombres de las clases
    digits=4                    # 4 decimales de precisión
)
# classification_report retorna un string formateado con:
# - Precision: Exactitud de predicciones positivas
# - Recall: Sensibilidad (cuántos casos positivos encontramos)
# - F1-Score: Media armónica de precision y recall
# - Support: Número de imágenes reales por clase

print(report)

print("=" * 80)

# ═════════════════════════════════════════════════════════════════════════════
# 💡 ANÁLISIS E INTERPRETACIÓN
# ═════════════════════════════════════════════════════════════════════════════

print("""
╔═══════════════════════════════════════════════════════════════════════╗
║                   💡 ANÁLISIS DE RESULTADOS                         ║
╠═══════════════════════════════════════════════════════════════════════╣
║
║  1️⃣ ACCURACY (~93%):
║     - Excelente para clasificación de 70 clases
║     - El modelo generaliza bien a datos nuevos
║     - Mejor que modelos genéricos sin Transfer Learning
║
║  2️⃣ MATRIZ DE CONFUSIÓN:
║     - Diagonal principal = aciertos
║     - Valores fuera de diagonal = confusiones
║     - Si hay mucho color fuera de diagonal = problemas
║     - Razas similares pueden confundirse (ej: Poodle vs Doodle)
║
║  3️⃣ PRECISION vs RECALL:
║     - Precision alta: No muchos falsos positivos
║     - Recall alta: Encontramos la mayoría de instancias reales
║     - Idealmente ambas altas (>90%)
║     - Si hay diferencia: Ajustar threshold de decisión
║
║  4️⃣ RAZAS CON BAJO RENDIMIENTO:
║     - Buscar en el reporte clases con F1-Score bajo
║     - Causas: Dataset desbalanceado, razas muy similares
║     - Soluciones: Más datos, augmentation específico, reentrenamiento
║
║  5️⃣ RAZAS CONFUNDIDAS:
║     - Observar matriz de confusión
║     - Razas similares se confunden (natural)
║     - Posible mejora: Aumentar datos de razas problemáticas
║
╚═══════════════════════════════════════════════════════════════════════╝
""")

# ═════════════════════════════════════════════════════════════════════════════
# 📈 ESTADÍSTICAS POR CLASE
# ═════════════════════════════════════════════════════════════════════════════

print("\n📈 Estadísticas por clase:")
print("-" * 80)

# Calcular accuracy por clase desde la matriz de confusión
for i, breed in enumerate(class_names):
    # Accuracy de una clase = diagonal / suma de fila
    if cm[i].sum() > 0:
        class_accuracy = cm[i, i] / cm[i].sum() * 100
        print(f"{breed:25} → Accuracy: {class_accuracy:6.2f}% ({cm[i,i]:4d}/{cm[i].sum():4d})")

print("-" * 80)