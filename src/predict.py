import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# =========================
# CARGAR MODELO
# =========================

model = tf.keras.models.load_model(
    "models/dog_classifier.keras"
)

# =========================
# CLASES
# =========================

class_names = [

    "beagle",
    "boxer",
    "bulldog",
    "dachshund",
    "german_shepherd",
    "golden_retriever",
    "labrador_retriever",
    "poodle",
    "rottweiler",
    "yorkshire_terrier"

]

# =========================
# IMAGEN A PREDECIR
# =========================

img_path = "test2.jpg"

# =========================
# CARGAR IMAGEN
# =========================

img = image.load_img(

    img_path,

    target_size=(224,224)

)

img_array = image.img_to_array(img)

img_array = np.expand_dims(img_array, axis=0)

# =========================
# PREDICCIÓN
# =========================

prediction = model.predict(img_array)

predicted_index = np.argmax(prediction)

predicted_class = class_names[predicted_index]

confidence = np.max(prediction)

# =========================
# MOSTRAR RESULTADO
# =========================

plt.imshow(img)

plt.title(

    f"Predicción: {predicted_class}\n"

    f"Confianza: {confidence:.2%}"

)

plt.axis("off")

plt.show()

print("\nPredicción:", predicted_class)

print("Confianza:", confidence)