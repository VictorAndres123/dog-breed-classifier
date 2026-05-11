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
    "afghan":           "🐕 Elegant and aloof sighthound with a silky coat.",
    "african_wild_dog": "🐾 Highly social and endangered African predator.",
    "airedale":         "🐶 Largest of the terriers, bold and adventurous.",
    "american_hairless":"🐾 Lively, alert and friendly hairless breed.",
    "american_spaniel": "🐕 Gentle, happy and active sporting dog.",
    "basenji":          "🔕 Ancient African breed that doesn't bark.",
    "basset":           "🐶 Low-set, scent-driven and easygoing hound.",
    "beagle":           "🐶 Curious and energetic hunting dog.",
    "bearded_collie":   "🐕 Bouncy, charismatic and active herding dog.",
    "bermaise":         "🐾 Large, calm and loyal mountain dog.",
    "bichon_frise":     "☁️ Cheerful, gentle and playful fluffy companion.",
    "blenheim":         "🍂 Sweet and gentle Cavalier King Charles Spaniel.",
    "bloodhound":       "🔍 Tenacious tracker with an extraordinary nose.",
    "bluetick":         "🐾 Tenacious and energetic coonhound.",
    "border_collie":    "🧠 Highly intelligent and energetic herding dog.",
    "borzoi":           "🐕 Graceful and independent Russian sighthound.",
    "boston_terrier":   "🎩 Friendly, bright and amusing American gentleman.",
    "boxer":            "🥊 Playful, energetic and loyal family dog.",
    "bull_mastiff":     "🛡️ Reliable, devoted and alert guardian breed.",
    "bull_terrier":     "🥚 Mischievous, playful and courageous terrier.",
    "bulldog":          "🐾 Calm, courageous and friendly breed.",
    "cairn":            "🌿 Alert, spirited and hardy Scottish terrier.",
    "chihuahua":        "🐕 Tiny but bold and loyal companion.",
    "chinese_crested":  "✨ Elegant, alert and entertaining toy breed.",
    "chow":             "🦁 Dignified, serious and loyal northern breed.",
    "clumber":          "🐕 Gentle, loyal and hardworking spaniel.",
    "cockapoo":         "🐾 Affectionate, cheerful Cocker-Poodle mix.",
    "cocker":           "🐕 Merry, gentle and trusting spaniel.",
    "collie":           "🐑 Devoted, graceful and loyal herding dog.",
    "corgi":            "👑 Bold, tenacious and friendly herding dog.",
    "coyote":           "🐺 Adaptable and clever wild canine.",
    "dalmation":        "🔴 Outgoing, active and dignified coach dog.",
    "dhole":            "🐾 Social and cooperative Asian wild dog.",
    "dingo":            "🦊 Alert and agile Australian wild dog.",
    "doberman":         "🛡️ Alert, loyal and fearless guard dog.",
    "elk_hound":        "🏔️ Bold, energetic Norwegian hunting dog.",
    "french_bulldog":   "🐶 Adaptable, playful and smart city dog.",
    "german_sheperd":   "🐕‍🦺 Confident, courageous and smart working dog.",
    "golden_retriever": "🐕 Friendly, intelligent and devoted dog.",
    "great_dane":       "🏛️ Friendly, patient and dependable gentle giant.",
    "great_perenees":   "⛰️ Calm, patient and devoted livestock guardian.",
    "greyhound":        "💨 Athletic, gentle and noble sighthound.",
    "groenendael":      "🐕 Intelligent, versatile Belgian shepherd.",
    "irish_spaniel":    "🍀 Clownish, friendly and active sporting dog.",
    "irish_wolfhound":  "🏰 Gentle, patient and commanding giant hound.",
    "japanese_spaniel": "🏯 Aristocratic, loving and cat-like companion.",
    "komondor":         "🧶 Loyal, dignified mop-coated flock guardian.",
    "labradoodle":      "🐾 Sociable, gentle and intelligent Labrador-Poodle mix.",
    "labrador":         "🦮 Gentle, outgoing and trusting retriever.",
    "lhasa":            "🏔️ Confident, smart and comical Tibetan breed.",
    "malinois":         "⚡ Intense, athletic and hardworking Belgian shepherd.",
    "maltese":          "🤍 Gentle, playful and fearless toy breed.",
    "mex_hairless":     "🌵 Calm, loyal and ancient Mexican breed.",
    "newfoundland":     "🌊 Sweet, patient and devoted working dog.",
    "pekinese":         "🏮 Regal, willful and affectionate toy breed.",
    "pit_bull":         "💪 Confident, loyal and courageous terrier.",
    "pomeranian":       "🦊 Lively, bold and inquisitive toy breed.",
    "poodle":           "🐩 Extremely intelligent and elegant breed.",
    "pug":              "😄 Charming, mischievous and loving small dog.",
    "rhodesian":        "🦁 Dignified, strong and loyal ridgeback hound.",
    "rottweiler":       "🛡️ Strong, loyal and protective dog.",
    "saint_bernard":    "🏔️ Gentle, patient and friendly rescue dog.",
    "schnauzer":        "👴 Spirited, obedient and friendly terrier.",
    "scotch_terrier":   "🏴󠁧󠁢󠁳󠁣󠁴󠁿 Independent, feisty and loyal Scottish terrier.",
    "shar_pei":         "📜 Loyal, calm and independent wrinkled breed.",
    "shiba_inu":        "🦊 Alert, active and spirited Japanese breed.",
    "shih_tzu":         "🏮 Affectionate, playful and outgoing companion.",
    "siberian_husky":   "❄️ Energetic sled dog with wolf-like appearance.",
    "vizsla":           "🏕️ Gentle, energetic and affectionate pointer.",
    "yorkie":           "✨ Feisty, brave and affectionate toy breed.",
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
""")

st.sidebar.info(
    "Upload a dog image and let the AI identify the breed."
)

# ═════════════════════════════════════════════════════════════
# 🏠 INTERFAZ PRINCIPAL
# ═════════════════════════════════════════════════════════════

st.title("🐶 AI Dog Breed Classifier")

st.markdown("""
Upload an image of a dog and the AI model will identify the breed among **70 different breeds**.
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
    image_array = np.array(image_resized, dtype=np.float32)
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
            st.info(breed_info[predicted_class])

    # 🏆 TOP 5 PREDICCIONES

    st.markdown("---")

    st.subheader("🏆 Top 5 Predictions")

    top_5_indices = prediction[0].argsort()[-5:][::-1]

    top_data = []

    for idx in top_5_indices:

        breed = class_names[idx]

        prob = float(prediction[0][idx] * 100)

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
        yaxis={'categoryorder':'total ascending'}
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
    yaxis={'categoryorder':'total ascending'}
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
st.caption("🚀 Built with TensorFlow and Streamlit")