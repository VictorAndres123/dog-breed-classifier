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
# 🌍 LANGUAGE SELECTOR
# ═════════════════════════════════════════════════════════════

language = st.sidebar.selectbox(
    "🌍 Language / Idioma",
    ["English", "Español"]
)

if language == "Español":

    title_text = "🐶 Clasificador de Razas de Perros"
    subtitle_text = "Sube una imagen de un perro y la IA identificará la raza."
    upload_text = "📤 Subir imagen"
    camera_text = "📸 Tomar foto"
    prediction_text = "## 🎯 Predicción"
    confidence_text = "Confianza"
    analyzing_text = "🧠 La IA está analizando el perro..."
    warning_text = "⚠️ El modelo no está muy seguro de esta predicción."
    top5_text = "🏆 Top 5 Predicciones"
    ask_text = "## 🐾 Preguntas sobre esta raza"
    choose_question = "Selecciona una pregunta:"
    footer_text = "🚀 Desarrollado con TensorFlow, EfficientNetB0 y Streamlit"

    questions = {
        "🌍 Origen":
            "Esta raza tiene orígenes ligados a regiones específicas.",

        "⏳ Tiempo de vida":
            "La esperanza de vida promedio suele ser entre 10 y 15 años.",

        "🏙️ ¿Ciudad o campo?":
            "Algunas razas se adaptan mejor a apartamentos y otras al campo.",

        "👨‍👩‍👧 ¿Buena con niños?":
            "Muchas razas son amigables y protectoras con familias.",

        "🏃 Necesidad de ejercicio":
            "Las razas activas requieren caminatas y ejercicio diario.",

        "🧠 Inteligencia":
            "Esta raza es conocida por su capacidad de aprendizaje.",

        "🛡️ Temperamento":
            "El temperamento puede variar entre tranquilo y energético."
    }

else:

    title_text = "🐶 AI Dog Breed Classifier"
    subtitle_text = "Upload a dog image and the AI will identify the breed."
    upload_text = "📤 Upload a dog image"
    camera_text = "📸 Take a picture of a dog"
    prediction_text = "## 🎯 Prediction"
    confidence_text = "Confidence"
    analyzing_text = "🧠 AI is analyzing the dog..."
    warning_text = "⚠️ The model is not very confident about this prediction."
    top5_text = "🏆 Top 5 Predictions"
    ask_text = "## 🐾 Ask About This Breed"
    choose_question = "Choose a question:"
    footer_text = "🚀 Built with TensorFlow, EfficientNetB0 and Streamlit"

    questions = {
        "🌍 Origin":
            "This breed has origins linked to specific regions.",

        "⏳ Life Span":
            "Average life expectancy is usually between 10 and 15 years.",

        "🏙️ Better for City or Countryside?":
            "Some breeds adapt better to apartments while others prefer open spaces.",

        "👨‍👩‍👧 Good With Kids?":
            "Most breeds are affectionate and protective with families.",

        "🏃 Exercise Needs":
            "Active breeds require daily walks and exercise.",

        "🧠 Intelligence":
            "This breed is known for learning ability and obedience.",

        "🛡️ Temperament":
            "Temperament can range from calm to energetic."
    }

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

# ═════════════════════════════════════════════════════════════
# 🏠 MAIN INTERFACE
# ═════════════════════════════════════════════════════════════

st.title(title_text)

st.markdown(subtitle_text)

# METRICS

col_a, col_b, col_c = st.columns(3)

col_a.metric("Accuracy", "93%")
col_b.metric("Dog Breeds", "70")
col_c.metric("Model", "EfficientNetB0")

# ═════════════════════════════════════════════════════════════
# 📤 IMAGE INPUT
# ═════════════════════════════════════════════════════════════

uploaded_file = st.file_uploader(
    upload_text,
    type=["jpg", "jpeg", "png"]
)

camera_image = st.camera_input(
    camera_text
)

# ═════════════════════════════════════════════════════════════
# 🔍 PREDICTION
# ═════════════════════════════════════════════════════════════

if uploaded_file is not None or camera_image is not None:

    col1, col2 = st.columns([1, 1])

    # IMAGE

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

    # PREPROCESSING

    image_resized = image.resize(IMG_SIZE)

    image_array = np.array(
        image_resized,
        dtype=np.float32
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # PREDICTION

    with st.spinner(analyzing_text):

        prediction = model.predict(image_array)

    predicted_index = np.argmax(prediction)

    predicted_class = class_names[predicted_index]

    confidence = np.max(prediction)

    # RESULTS

    with col2:

        st.markdown(prediction_text)

        st.success(
            f"### {predicted_class.replace('_', ' ').title()}"
        )

        st.metric(
            confidence_text,
            f"{confidence * 100:.2f}%"
        )

        if confidence < 0.50:

            st.warning(warning_text)

        # QUESTIONS

        st.markdown(ask_text)

        selected_question = st.selectbox(
            choose_question,
            list(questions.keys())
        )

        st.info(
            questions[selected_question]
        )

    # TOP 5

    st.markdown("---")

    st.subheader(top5_text)

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

    # CHART

    fig = px.bar(
        df,
        x="Confidence",
        y="Breed",
        orientation="h",
        text="Confidence",
        title=top5_text
    )

    fig.update_layout(
        height=500,
        yaxis={'categoryorder': 'total ascending'}
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # TABLE

    st.dataframe(
        df,
        use_container_width=True
    )

# ═════════════════════════════════════════════════════════════
# 🔚 FOOTER
# ═════════════════════════════════════════════════════════════

st.markdown("---")

st.caption(footer_text)