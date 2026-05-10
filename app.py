import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd

from PIL import Image
from tensorflow.keras.preprocessing import image_dataset_from_directory

# =========================================================
# CONFIGURACIÓN DE PÁGINA
# =========================================================

st.set_page_config(
    page_title="🐶 AI Dog Breed Classifier",
    page_icon="🐾",
    layout="wide"
)

# =========================================================
# ESTILOS PERSONALIZADOS
# =========================================================

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

.prediction-box {
    padding: 20px;
    border-radius: 15px;
    background-color: #1E1E1E;
    margin-top: 20px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# CONFIGURACIÓN
# =========================================================

IMG_SIZE = (224, 224)

BATCH_SIZE = 16

# =========================================================
# CARGAR MODELO
# =========================================================

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        "models/dog_classifier.keras"
    )

    return model

model = load_model()

# =========================================================
# CARGAR CLASES
# =========================================================

@st.cache_resource
def load_class_names():

    dataset = image_dataset_from_directory(

        "dataset",

        image_size=IMG_SIZE,

        batch_size=BATCH_SIZE,

        shuffle=False

    )

    return dataset.class_names

class_names = load_class_names()

# =========================================================
# INFORMACIÓN DE RAZAS
# =========================================================

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

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🐾 AI Dog Classifier")

st.sidebar.markdown("""
### Features

✅ 70 Dog Breeds  
✅ Deep Learning  
✅ EfficientNetB0  
✅ 93% Accuracy  
✅ Top 5 Predictions  
✅ Transfer Learning  
""")

st.sidebar.info(
    "Upload a dog image and let the AI identify the breed."
)

# =========================================================
# TÍTULO PRINCIPAL
# =========================================================

st.title("🐶 AI Dog Breed Classifier")

st.markdown("""
Upload an image of a dog and the AI model will try to identify the breed.
""")

# =========================================================
# SUBIR IMAGEN
# =========================================================

uploaded_file = st.file_uploader(

    "📤 Upload a dog image",

    type=["jpg", "jpeg", "png"]

)

# =========================================================
# SI HAY IMAGEN
# =========================================================

if uploaded_file is not None:

    col1, col2 = st.columns([1, 1])

    # =====================================================
    # COLUMNA IMAGEN
    # =====================================================

    with col1:

        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

    # =====================================================
    # PREPROCESAMIENTO
    # =====================================================

    image_resized = image.resize(IMG_SIZE)

    image_array = np.array(image_resized)

    image_array = np.expand_dims(image_array, axis=0)

    # =====================================================
    # PREDICCIÓN
    # =====================================================

    with st.spinner("🧠 AI is analyzing the dog..."):

        prediction = model.predict(image_array)

    predicted_index = np.argmax(prediction)

    predicted_class = class_names[predicted_index]

    confidence = np.max(prediction)

    # =====================================================
    # COLUMNA RESULTADOS
    # =====================================================

    with col2:

        st.markdown("## 🎯 Prediction")

        st.success(
            f"### {predicted_class.replace('_', ' ').title()}"
        )

        st.metric(
            "Confidence",
            f"{confidence*100:.2f}%"
        )

        # =================================================
        # ALERTA CONFIANZA
        # =================================================

        if confidence < 0.50:

            st.warning(
                "⚠️ The model is not very confident about this prediction."
            )

        # =================================================
        # INFORMACIÓN DE RAZA
        # =================================================

        if predicted_class in breed_info:

            st.info(
                breed_info[predicted_class]
            )

    # =====================================================
    # TOP 5 PREDICCIONES
    # =====================================================

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

    # =====================================================
    # TABLA TOP 5
    # =====================================================

    df = pd.DataFrame(top_data)

    st.dataframe(
        df,
        use_container_width=True
    )

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "🚀 Built with TensorFlow, EfficientNetB0 and Streamlit"
)