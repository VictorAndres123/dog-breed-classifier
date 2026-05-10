import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)

# ==========================
# CONFIG
# ==========================

IMG_SIZE = (224, 224)

BATCH_SIZE = 16

# ==========================
# CARGAR MODELO
# ==========================

model = tf.keras.models.load_model(
    "models/dog_classifier.keras"
)

# ==========================
# CARGAR DATASET
# ==========================

dataset = tf.keras.preprocessing.image_dataset_from_directory(

    "dataset",

    image_size=IMG_SIZE,

    batch_size=BATCH_SIZE,

    shuffle=False

)

class_names = dataset.class_names

print("\nClases detectadas:\n")

print(class_names)

# ==========================
# PREDICCIONES
# ==========================

y_true = []

y_pred = []

for images, labels in dataset:

    predictions = model.predict(images)

    predicted_labels = np.argmax(predictions, axis=1)

    y_true.extend(labels.numpy())

    y_pred.extend(predicted_labels)

# ==========================
# ACCURACY
# ==========================

accuracy = accuracy_score(y_true, y_pred)

print(f"\nAccuracy final: {accuracy*100:.2f}%")

# ==========================
# MATRIZ CONFUSIÓN
# ==========================

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(25,25))

sns.heatmap(

    cm,

    cmap="Blues",

    xticklabels=class_names,

    yticklabels=class_names

)

plt.xlabel("Predicción")

plt.ylabel("Real")

plt.title("Matriz de Confusión")

plt.show()

# ==========================
# REPORTE CLASIFICACIÓN
# ==========================

print("\nCLASSIFICATION REPORT:\n")

print(

    classification_report(

        y_true,

        y_pred,

        target_names=class_names

    )

)