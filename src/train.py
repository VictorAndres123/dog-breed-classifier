import os
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.callbacks import EarlyStopping

# =========================
# CONFIGURACIÓN
# =========================

IMG_SIZE = (224, 224)

BATCH_SIZE = 16

EPOCHS = 25

# =========================
# CARGAR DATASET
# =========================

train_dataset = image_dataset_from_directory(

    "dataset",

    validation_split=0.2,

    subset="training",

    seed=123,

    image_size=IMG_SIZE,

    batch_size=BATCH_SIZE

)

validation_dataset = image_dataset_from_directory(

    "dataset",

    validation_split=0.2,

    subset="validation",

    seed=123,

    image_size=IMG_SIZE,

    batch_size=BATCH_SIZE

)

# =========================
# NOMBRES CLASES
# =========================

class_names = train_dataset.class_names

print("\nClases detectadas:\n")

print(class_names)

# =========================
# OPTIMIZACIÓN DATASET
# =========================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)

validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)

# =========================
# DATA AUGMENTATION
# =========================

data_augmentation = tf.keras.Sequential([

    layers.RandomFlip("horizontal"),

    layers.RandomRotation(0.3),

    layers.RandomZoom(0.3),

    layers.RandomContrast(0.3),

    layers.RandomBrightness(0.2),

    layers.RandomTranslation(0.2, 0.2)

])

# =========================
# MODELO BASE
# =========================

base_model = EfficientNetB0(

    input_shape=(224, 224, 3),

    include_top=False,

    weights="imagenet"

)

# =========================
# FINE TUNING
# =========================

base_model.trainable = True

# Congelar primeras capas
for layer in base_model.layers[:-30]:

    layer.trainable = False

# =========================
# MODELO FINAL
# =========================

model = models.Sequential([

    data_augmentation,

    layers.Rescaling(1./255),

    base_model,

    layers.GlobalAveragePooling2D(),

    layers.Dense(256, activation='relu'),

    layers.Dropout(0.4),

    layers.Dense(len(class_names), activation='softmax')

])

# =========================
# COMPILAR MODELO
# =========================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),

    loss='sparse_categorical_crossentropy',

    metrics=['accuracy']

)

# =========================
# RESUMEN
# =========================

model.summary()

# =========================
# EARLY STOPPING
# =========================

early_stop = EarlyStopping(

    monitor='val_loss',

    patience=5,

    restore_best_weights=True

)

# =========================
# ENTRENAMIENTO
# =========================

history = model.fit(

    train_dataset,

    validation_data=validation_dataset,

    epochs=EPOCHS,

    callbacks=[early_stop]

)

# =========================
# CREAR CARPETA MODELS
# =========================

os.makedirs("models", exist_ok=True)

# =========================
# GUARDAR MODELO
# =========================

model.save("models/dog_classifier.keras")

print("\nModelo guardado correctamente")

# =========================
# GRÁFICA ACCURACY
# =========================

acc = history.history['accuracy']

val_acc = history.history['val_accuracy']

epochs_range = range(len(acc))

plt.figure(figsize=(10,5))

plt.plot(epochs_range, acc, label='Training Accuracy')

plt.plot(epochs_range, val_acc, label='Validation Accuracy')

plt.legend(loc='lower right')

plt.title('Training and Validation Accuracy')

plt.show()

# =========================
# GRÁFICA LOSS
# =========================

loss = history.history['loss']

val_loss = history.history['val_loss']

plt.figure(figsize=(10,5))

plt.plot(epochs_range, loss, label='Training Loss')

plt.plot(epochs_range, val_loss, label='Validation Loss')

plt.legend(loc='upper right')

plt.title('Training and Validation Loss')

plt.show()