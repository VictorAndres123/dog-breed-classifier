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
import plotly.express as px

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

class_names = [
    "afghan", "african_wild_dog", "airedale", "american_hairless",
    "american_spaniel", "basenji", "basset", "beagle", "bearded_collie",
    "bermaise", "bichon_frise", "blenheim", "bloodhound", "bluetick",
    "border_collie", "borzoi", "boston_terrier", "boxer", "bull_mastiff",
    "bull_terrier", "bulldog", "cairn", "chihuahua", "chinese_crested",
    "chow", "clumber", "cockapoo", "cocker", "collie", "corgi", "coyote",
    "dalmation", "dhole", "dingo", "doberman", "elk_hound", "french_bulldog",
    "german_sheperd", "golden_retriever", "great_dane", "great_perenees",
    "greyhound", "groenendael", "irish_spaniel", "irish_wolfhound",
    "japanese_spaniel", "komondor", "labradoodle", "labrador", "lhasa",
    "malinois", "maltese", "mex_hairless", "newfoundland", "pekinese",
    "pit_bull", "pomeranian", "poodle", "pug", "rhodesian", "rottweiler",
    "saint_bernard", "schnauzer", "scotch_terrier", "shar_pei", "shiba_inu",
    "shih_tzu", "siberian_husky", "vizsla", "yorkie"
]

# ═════════════════════════════════════════════════════════════
# 📚 INFORMACIÓN DE RAZAS
# ═════════════════════════════════════════════════════════════

breed_info = {
    "golden_retriever": "🐕 Friendly, intelligent and devoted dog.",
    "poodle": "🐩 Extremely intelligent and elegant breed.",
    "beagle": "🐶 Curious and energetic hunting dog.",
    "rottweiler": "🛡️ Strong, loyal and protective dog.",
    "siberian_husky": "❄️ Energetic sled dog with wolf-like appearance.",
}

# ═════════════════════════════════════════════════════════════
# 🧠 CARGAR MODELO
# ═════════════════════════════════════════════════════════════

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        "models/dog_classifier_fixed.keras",
        compile=False
    )
    return model

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
✅ 70 Dog Breeds  
✅ Camera Support
""")

st.sidebar.info(
    "Upload a dog image or use your camera."
)

# ═════════════════════════════════════════════════════════════
# 🏠 INTERFAZ PRINCIPAL
# ═════════════════════════════════════════════════════════════

st.title("🐶 AI Dog Breed Classifier")

st.markdown("""
Upload an image of a dog and the AI model will identify the breed among **70 different breeds**.
""")

# MÉTRICAS

col_a, col_b, col_c = st.columns(3)

col_a.metric("Accuracy", "93%")
col_b.metric("Dog Breeds", "70")
col_c.metric("Model", "EfficientNetB0")

# ═════════════════════════════════════════════════════════════
# 📤 SUBIR IMAGEN
# ═════════════════════════════════════════════════════════════

uploaded_file = st.file_uploader(
    "📤 Upload a dog image",
    type=["jpg", "jpeg", "png"]
)

camera_image = st.camera_input(
    "📸 Take a picture of a dog"
)

# ═════════════════════════════════════════════════════════════
# 🔍 PREDICCIÓN
# ═════════════════════════════════════════════════════════════

if uploaded_file is not None or camera_image is not None:

    col1, col2 = st.columns([1, 1])

    # 📸 IMAGEN
    with col1:

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
        else:
            image = Image.open(camera_image).convert("RGB")

        st.image(
            image,
            caption="Dog Image",
            use_container_width=True
        )

    # 🧠 PREPROCESAMIENTO

    image_resized = image.resize(IMG_SIZE)

    image_array = np.array(
        image_resized,
        dtype=np.float32
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # 🤖 PREDICCIÓN

    with st.spinner("🧠 AI is analyzing the dog..."):

        prediction = model.predict(image_array)

    predicted_index = np.argmax(prediction)

    predicted_class = class_names[predicted_index]

    confidence = np.max(prediction)

    # 📊 RESULTADOS

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

    # 🐾 INFORMACIÓN DE LA RAZA
    if predicted_class in breed_info:

        info = breed_info[predicted_class]

        st.markdown("## 🐾 Breed Information")

        colA, colB = st.columns(2)

        with colA:
            st.metric("🌍 Origin", info["origin"])
            st.metric("📏 Size", info["size"])
            st.metric("⚖️ Weight", info["weight"])
            st.metric("⏳ Life Span", info["life_span"])

        with colB:
            st.metric("🏃 Exercise Needs", info["exercise"])
            st.metric("👨‍👩‍👧 Good With Kids", info["good_with_kids"])

        st.info(f"🧠 Temperament: {info['temperament']}")

        st.success(info["description"])

    # ═════════════════════════════════════════════════════════════
    # 🏆 TOP 5 PREDICCIONES
    # ═════════════════════════════════════════════════════════════

    st.markdown("---")

    st.subheader("🏆 Top 5 Predictions")

    top_5_indices = prediction[0].argsort()[-5:][::-1]

    top_data = []

    for idx in top_5_indices:

        breed = class_names[idx]

        prob = float(
            prediction[0][idx] * 100
        )

        top_data.append({
            "Breed": breed.replace("_", " ").title(),
            "Confidence": round(prob, 2)
        })

    df = pd.DataFrame(top_data)

    # 📊 GRÁFICO INTERACTIVO

    fig = px.bar(
        df,
        x="Confidence",
        y="Breed",
        orientation="h",
        text="Confidence",
        title="Top 5 Most Probable Breeds"
    )

    fig.update_layout(
        height=500,
        yaxis={'categoryorder': 'total ascending'}
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # 📋 TABLA

    st.dataframe(
        df,
        use_container_width=True
    )

# ═════════════════════════════════════════════════════════════
# 🔚 FOOTER
# ═════════════════════════════════════════════════════════════

st.markdown("---")

st.caption(
    "🚀 Built with TensorFlow, EfficientNetB0 and Streamlit"
)