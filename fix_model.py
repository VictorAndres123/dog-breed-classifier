import tensorflow as tf

print("TF version:", tf.__version__)
print("Keras version:", tf.keras.__version__)

# Cargar modelo original
model = tf.keras.models.load_model(
    "models/dog_classifier.keras",
    compile=False
)

# Reexportar en formato .keras limpio
model.save("models/dog_classifier_fixed.keras")

print("✅ Modelo reexportado correctamente")