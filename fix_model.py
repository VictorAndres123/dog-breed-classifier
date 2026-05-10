import tensorflow as tf

# Cargar modelo original
model = tf.keras.models.load_model("models/dog_classifier.keras")

# Guardarlo nuevamente en formato compatible
model.save("models/dog_classifier_fixed.h5")

print("Modelo convertido correctamente")