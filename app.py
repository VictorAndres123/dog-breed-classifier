"""
🐶 AI DOG BREED CLASSIFIER
Aplicación web con Streamlit + TensorFlow
"""

# ═════════════════════════════════════════════════════════════
# 📚 IMPORTACIONES
# ═════════════════════════════════════════════════════════════

import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd

from PIL import Image

# ═════════════════════════════════════════════════════════════
# 🎨 CONFIGURACIÓN STREAMLIT
# ═════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="🐶 AI Dog Breed Classifier",
    page_icon="🐾",
    layout="wide"
)

# ═════════════════════════════════════════════════════════════
# 🎨 ESTILOS CSS
# ═════════════════════════════════════════════════════════════

st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

h1, h2, h3 {
    color: white;
}

.stButton>button {
    border-radius: 12px;
    height: 3em;
    width: 100%;
    font-size: 18px;
}

</style>
""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════
# ⚙️ CONFIGURACIÓN
# ═════════════════════════════════════════════════════════════

IMG_SIZE = (224, 224)

# ⚠️ IMPORTANTE:
# Pon aquí EXACTAMENTE las clases con las que entrenaste el modelo
# en el mismo orden del entrenamiento.

class_names = [
    "beagle",
    "boxer",
    "bulldog",
    "chihuahua",
    "german_shepherd",
    "golden_retriever",
    "labrador_retriever",
    "poodle",
    "rottweiler",
    "siberian_husky"
]

# ═════════════════════════════════════════════════════════════
# 📚 INFORMACIÓN DE RAZAS
# ═════════════════════════════════════════════════════════════

breed_info = {
    "golden_retriever":
    "🐕 Friendly, intelligent and devoted dog.",

    "poodle":
    "🐩 Extremely intelligent and elegant breed.",

    "beagle":
    "🐶 Curious and energetic hunting dog.",

    "rottweiler":
    "🛡️ Strong, loyal and protective dog.",

    "siberian_husky":
    "❄️ Energetic sled dog with wolf-like appearance.",
}

# ═════════════════════════════════════════════════════════════
# 🧠 CARGAR MODELO
# ═════════════════════════════════════════════════════════════

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
    "models/dog_classifier_fixed.keras",
    compile=False  # ya lo tienes, bien ✅
)

    return model

# Manejo de errores
try:
    model = load_model()

except Exception as e:
    st.error("❌ Error loading model")
    st.error(str(e))
    st.stop()

# ═════════════════════════════════════════════════════════════
# 📋 SIDEBAR
# ═════════════════════════════════════════════════════════════

st.sidebar.title("🐾 AI Dog Classifier")

st.sidebar.markdown("""
### Features

✅ Deep Learning  
✅ TensorFlow  
✅ Streamlit  
✅ Top 5 Predictions  
✅ Real-time Classification  
""")

st.sidebar.info(
    "Upload a dog image and let the AI identify the breed."
)

# ═════════════════════════════════════════════════════════════
# 🏠 INTERFAZ PRINCIPAL
# ═════════════════════════════════════════════════════════════

st.title("🐶 AI Dog Breed Classifier")

st.markdown("""
Upload an image of a dog and the AI model will identify the breed.
""")

# ═════════════════════════════════════════════════════════════
# 📤 SUBIR IMAGEN
# ═════════════════════════════════════════════════════════════

uploaded_file = st.file_uploader(
    "📤 Upload a dog image",
    type=["jpg", "jpeg", "png"]
)

# ═════════════════════════════════════════════════════════════
# 🔍 PREDICCIÓN
# ═════════════════════════════════════════════════════════════

if uploaded_file is not None:

    col1, col2 = st.columns([1, 1])

    # 📸 Mostrar imagen
    with col1:

        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

    # 🧠 Procesamiento
    image_resized = image.resize(IMG_SIZE)

    image_array = np.array(image_resized)

    # Normalización
    image_array = image_array / 255.0

    # Batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # Predicción
    with st.spinner("🧠 AI is analyzing the dog..."):

        prediction = model.predict(image_array)

    predicted_index = np.argmax(prediction)

    predicted_class = class_names[predicted_index]

    confidence = np.max(prediction)

    # 📊 Resultados
    with col2:

        st.markdown("## 🎯 Prediction")

        st.success(
            f"### {predicted_class.replace('_', ' ').title()}"
        )

        st.metric(
            "Confidence",
            f"{confidence * 100:.2f}%"
        )

        if confidence < 0.50:

            st.warning(
                "⚠️ The model is not very confident about this prediction."
            )

        if predicted_class in breed_info:

            st.info(
                breed_info[predicted_class]
            )

    # ═════════════════════════════════════════════════════════
    # 🏆 TOP 5
    # ═════════════════════════════════════════════════════════

    st.markdown("---")

    st.subheader("🏆 Top 5 Predictions")

    top_5_indices = prediction[0].argsort()[-5:][::-1]

    top_data = []

    for idx in top_5_indices:

        breed = class_names[idx]

        prob = float(prediction[0][idx] * 100)

        top_data.append({
            "Breed": breed.replace("_", " ").title(),
            "Confidence (%)": round(prob, 2)
        })

        st.progress(prob / 100)

        st.write(
            f"**{breed.replace('_', ' ').title()}** — {prob:.2f}%"
        )

    df = pd.DataFrame(top_data)

    st.dataframe(
        df,
        use_container_width=True
    )

# ═════════════════════════════════════════════════════════════
# 🔚 FOOTER
# ═════════════════════════════════════════════════════════════

st.markdown("---")

st.caption(
    "🚀 Built with TensorFlow and Streamlit"
)