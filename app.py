"""
🐶 AI DOG BREED CLASSIFIER
Aplicación web con Streamlit + PyTorch (EfficientNetB0)
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
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
    title_text      = "🐶 Clasificador de Razas de Perros"
    subtitle_text   = "Sube una imagen de un perro y la IA identificará la raza."
    upload_text     = "📤 Subir imagen"
    camera_text     = "📸 Tomar foto"
    prediction_text = "## 🎯 Predicción"
    confidence_text = "Confianza"
    analyzing_text  = "🧠 La IA está analizando el perro..."
    warning_text    = "⚠️ El modelo no está muy seguro de esta predicción."
    top5_text       = "🏆 Top 5 Predicciones"
    ask_text        = "## 🐾 Preguntas sobre esta raza"
    choose_question = "Selecciona una pregunta:"
    footer_text     = "🚀 Desarrollado con PyTorch, EfficientNetB0 y Streamlit"
    questions = {
        "🌍 Origen":                  "Esta raza tiene orígenes ligados a regiones específicas.",
        "⏳ Tiempo de vida":           "La esperanza de vida promedio suele ser entre 10 y 15 años.",
        "🏙️ ¿Ciudad o campo?":        "Algunas razas se adaptan mejor a apartamentos y otras al campo.",
        "👨‍👩‍👧 ¿Buena con niños?":       "Muchas razas son amigables y protectoras con familias.",
        "🏃 Necesidad de ejercicio":   "Las razas activas requieren caminatas y ejercicio diario.",
        "🧠 Inteligencia":             "Esta raza es conocida por su capacidad de aprendizaje.",
        "🛡️ Temperamento":             "El temperamento puede variar entre tranquilo y energético."
    }
else:
    title_text      = "🐶 AI Dog Breed Classifier"
    subtitle_text   = "Upload a dog image and the AI will identify the breed."
    upload_text     = "📤 Upload a dog image"
    camera_text     = "📸 Take a picture of a dog"
    prediction_text = "## 🎯 Prediction"
    confidence_text = "Confidence"
    analyzing_text  = "🧠 AI is analyzing the dog..."
    warning_text    = "⚠️ The model is not very confident about this prediction."
    top5_text       = "🏆 Top 5 Predictions"
    ask_text        = "## 🐾 Ask About This Breed"
    choose_question = "Choose a question:"
    footer_text     = "🚀 Built with PyTorch, EfficientNetB0 and Streamlit"
    questions = {
        "🌍 Origin":                        "This breed has origins linked to specific regions.",
        "⏳ Life Span":                      "Average life expectancy is usually between 10 and 15 years.",
        "🏙️ Better for City or Countryside?": "Some breeds adapt better to apartments while others prefer open spaces.",
        "👨‍👩‍👧 Good With Kids?":                "Most breeds are affectionate and protective with families.",
        "🏃 Exercise Needs":                 "Active breeds require daily walks and exercise.",
        "🧠 Intelligence":                   "This breed is known for learning ability and obedience.",
        "🛡️ Temperament":                   "Temperament can range from calm to energetic."
    }

# ═════════════════════════════════════════════════════════════
# 🎨 CSS
# ═════════════════════════════════════════════════════════════

st.markdown("""
<style>
.main { background-color: #0E1117; }
h1, h2, h3 { color: white; }
.stButton>button { border-radius: 12px; height: 3em; width: 100%; font-size: 18px; }
</style>
""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════
# ⚙️ CLASES — orden exacto del dataset (sorted con mayúsculas primero)
# ═════════════════════════════════════════════════════════════

NUM_CLASSES = 55
IMG_SIZE    = 224

# ⚠️ Scottish_deerhound va en índice 0 porque la S mayúscula
#    ordena antes que las minúsculas en Python/os.listdir
class_names = [
    "Scottish_deerhound",  #  0
    "afghan",              #  1
    "african_wild_dog",    #  2
    "airedale",            #  3
    "american_hairless",   #  4
    "basenji",             #  5
    "basset",              #  6
    "beagle",              #  7
    "bermaise",            #  8
    "blenheim",            #  9
    "bloodhound",          # 10
    "bluetick",            # 11
    "border_collie",       # 12
    "borzoi",              # 13
    "boston_terrier",      # 14
    "boxer",               # 15
    "bull_mastiff",        # 16
    "bulldog",             # 17
    "cairn",               # 18
    "chihuahua",           # 19
    "chow",                # 20
    "clumber",             # 21
    "cocker",              # 22
    "collie",              # 23
    "corgi",               # 24
    "dhole",               # 25
    "dingo",               # 26
    "doberman",            # 27
    "elk_hound",           # 28
    "german_sheperd",      # 29
    "golden_retriever",    # 30
    "great_dane",          # 31
    "great_perenees",      # 32
    "groenendael",         # 33
    "irish_wolfhound",     # 34
    "japanese_spaniel",    # 35
    "komondor",            # 36
    "labrador",            # 37
    "lhasa",               # 38
    "malinois",            # 39
    "maltese",             # 40
    "newfoundland",        # 41
    "pekinese",            # 42
    "pomeranian",          # 43
    "poodle",              # 44
    "pug",                 # 45
    "rhodesian",           # 46
    "rottweiler",          # 47
    "saint_bernard",       # 48
    "schnauzer",           # 49
    "scotch_terrier",      # 50
    "shih_tzu",            # 51
    "siberian_husky",      # 52
    "vizsla",              # 53
    "yorkie",              # 54
]

registered_breeds = [
    "Scottish Deerhound",
    "Afghan",
    "African Wild Dog",
    "Airedale",
    "American Hairless",
    "Basenji",
    "Basset",
    "Beagle",
    "Bermaise",
    "Blenheim",
    "Bloodhound",
    "Bluetick",
    "Border Collie",
    "Borzoi",
    "Boston Terrier",
    "Boxer",
    "Bull Mastiff",
    "Bulldog",
    "Cairn",
    "Chihuahua",
    "Chow",
    "Clumber",
    "Cocker",
    "Collie",
    "Corgi",
    "Dhole",
    "Dingo",
    "Doberman",
    "Elk Hound",
    "German Sheperd",
    "Golden Retriever",
    "Great Dane",
    "Great Perenees",
    "Groenendael",
    "Irish Wolfhound",
    "Japanese Spaniel",
    "Komondor",
    "Labrador",
    "Lhasa",
    "Malinois",
    "Maltese",
    "Newfoundland",
    "Pekinese",
    "Pomeranian",
    "Poodle",
    "Pug",
    "Rhodesian",
    "Rottweiler",
    "Saint Bernard",
    "Schnauzer",
    "Scotch Terrier",
    "Shih Tzu",
    "Siberian Husky",
    "Vizsla",
    "Yorkie",
]

assert len(class_names) == NUM_CLASSES, f"class_names tiene {len(class_names)}, esperados {NUM_CLASSES}"
assert len(registered_breeds) == NUM_CLASSES

# ═════════════════════════════════════════════════════════════
# 🔄 PREPROCESAMIENTO — igual que augmentation.py val_transforms
# ═════════════════════════════════════════════════════════════

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),                          # [0,255] → [0,1]
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ═════════════════════════════════════════════════════════════
# 🧠 ARQUITECTURA — copia exacta de model.py
# ═════════════════════════════════════════════════════════════

class DogBreedClassifier(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=None)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        return self.backbone(x)

# ═════════════════════════════════════════════════════════════
# 🧠 CARGAR MODELO
# ═════════════════════════════════════════════════════════════

MODEL_PATH = "models/best_efficientnet_b0.pth"

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = DogBreedClassifier(num_classes=NUM_CLASSES)
    sd     = torch.load(MODEL_PATH, map_location=device)
    # Soporta checkpoint completo o state_dict directo
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    elif isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    model.load_state_dict(sd)
    model.to(device)
    model.eval()
    return model, device

try:
    model, device = load_model()
except FileNotFoundError:
    st.error(f"❌ No se encontró el modelo en '{MODEL_PATH}'")
    st.stop()
except Exception as e:
    st.error("❌ Error cargando el modelo")
    st.error(str(e))
    st.stop()

# ═════════════════════════════════════════════════════════════
# 📋 SIDEBAR
# ═════════════════════════════════════════════════════════════

st.sidebar.title("🐾 AI Dog Classifier")
st.sidebar.markdown("""
### Features
✅ Deep Learning  
✅ PyTorch  
✅ Streamlit  
✅ Top 5 Predictions  
✅ Real-time Classification  
✅ 55 Dog Breeds  
✅ Camera Support
""")
gpu_info = "🟢 GPU" if torch.cuda.is_available() else "🔵 CPU"
st.sidebar.info(f"Device: {gpu_info}")

# ═════════════════════════════════════════════════════════════
# 🏠 MAIN INTERFACE
# ═════════════════════════════════════════════════════════════

st.title(title_text)
st.markdown(subtitle_text)

col_a, col_b, col_c = st.columns(3)
col_a.metric("Accuracy", "~90%")
col_b.metric("Dog Breeds", "55")
col_c.metric("Model", "EfficientNetB0")

st.markdown("---")

if language == "Español":
    st.subheader("📚 Razas Registradas")
    st.markdown("Selecciona una de las razas disponibles en el modelo:")
else:
    st.subheader("📚 Registered Dog Breeds")
    st.markdown("Browse all breeds available in the trained model:")

breeds_per_column = 19
breed_columns = st.columns(3)
for idx, column in enumerate(breed_columns):
    start = idx * breeds_per_column
    end = start + breeds_per_column
    column.markdown("\n".join(f"- {breed}" for breed in registered_breeds[start:end]))

# ═════════════════════════════════════════════════════════════
# 📤 IMAGE INPUT
# ═════════════════════════════════════════════════════════════

uploaded_file = st.file_uploader(upload_text, type=["jpg", "jpeg", "png"])

camera_image = None
with st.expander("📸 Cámara (opcional)", expanded=False):
    st.write("Usa la cámara solo si quieres capturar una foto en tiempo real.")
    camera_image = st.camera_input(camera_text)

# ═════════════════════════════════════════════════════════════
# 🔍 PREDICTION
# ═════════════════════════════════════════════════════

if uploaded_file is not None or camera_image is not None:

    col1, col2 = st.columns([1, 1])

    with col1:
        image = Image.open(
            uploaded_file if uploaded_file is not None else camera_image
        ).convert("RGB")
        st.image(image, caption="Dog Image", use_container_width=True)

    # Preprocesar igual que val_transforms de augmentation.py
    tensor = transform(image).unsqueeze(0).to(device)

    with st.spinner(analyzing_text):
        with torch.no_grad():
            outputs = model(tensor)
            probs   = torch.softmax(outputs, dim=1).cpu().numpy()[0]

    predicted_index = int(np.argmax(probs))
    predicted_class = class_names[predicted_index]
    confidence      = float(probs[predicted_index])

    with col2:
        st.markdown(prediction_text)
        st.success(f"### {predicted_class.replace('_', ' ').title()}")
        st.metric(confidence_text, f"{confidence * 100:.2f}%")

        if confidence < 0.50:
            st.warning(warning_text)

        st.markdown(ask_text)
        selected_question = st.selectbox(choose_question, list(questions.keys()))
        st.info(questions[selected_question])

    # TOP 5
    st.markdown("---")
    st.subheader(top5_text)

    top5_idx = probs.argsort()[-5:][::-1]
    top_data = [
        {
            "Breed":      class_names[i].replace("_", " ").title(),
            "Confidence": round(float(probs[i]) * 100, 2)
        }
        for i in top5_idx
    ]
    df = pd.DataFrame(top_data)

    fig = px.bar(
        df, x="Confidence", y="Breed", orientation="h",
        text="Confidence", title=top5_text
    )
    fig.update_layout(height=500, yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df, use_container_width=True)

# ═════════════════════════════════════════════════════════════
# 🔚 FOOTER
# ═════════════════════════════════════════════════════════════

st.markdown("---")
st.caption(footer_text)