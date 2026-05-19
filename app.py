"""Aplicacion Streamlit para clasificar razas de perros con EfficientNetB0."""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import models, transforms

# Configurar pagina de Streamlit

st.set_page_config(
    page_title="🐶 AI Dog Breed Classifier",
    page_icon="🐾",
    layout="wide"
)

# Selector de idioma para toda la interfaz

language = st.sidebar.selectbox("🌍 Language / Idioma", ["English", "Español"])

TEXTS = {
    "English": {
        "title": "🐶 AI Dog Breed Classifier",
        "subtitle": "Upload a dog image and the AI will identify the breed.",
        "upload": "📤 Upload a dog image",
        "camera_input": "📸 Take a picture of a dog",
        "prediction": "## 🎯 Prediction",
        "confidence": "Confidence",
        "analyzing": "🧠 AI is analyzing the dog...",
        "warning": "⚠️ The model is not very confident about this prediction.",
        "top5": "🏆 Top 5 Predictions",
        "confidence_gauge": "📊 Top-1 Confidence Gauge",
        "history": "🕘 Prediction history (session)",
        "clear_history": "Clear history",
        "history_empty": "No predictions in this session yet.",
        "footer": "🚀 Built with PyTorch, EfficientNetB0 and Streamlit",
        "model_not_found": "❌ Model not found at '{model_path}'",
        "loading_error": "❌ Error loading model",
        "sidebar_title": "🐾 AI Dog Classifier",
        "sidebar_features": "### Features\n✅ Deep Learning  \n✅ PyTorch  \n✅ Streamlit  \n✅ Top 5 Predictions  \n✅ Real-time Classification  \n✅ 55 Dog Breeds  \n✅ Camera Support",
        "metric_accuracy": "Accuracy",
        "metric_breeds": "Dog Breeds",
        "metric_model": "Model",
        "registered_title": "📚 Registered Dog Breeds",
        "registered_caption": "Explore all 55 breeds available in the trained model.",
        "camera_expander": "📸 Camera (optional)",
        "camera_help": "Use the camera only if you want to take a real-time photo.",
        "dog_image": "Dog Image",
        "col_breed": "Breed",
    },
    "Español": {
        "title": "🐶 Clasificador de Razas de Perros",
        "subtitle": "Sube una imagen de un perro y la IA identificará la raza.",
        "upload": "📤 Subir imagen de perro",
        "camera_input": "📸 Tomar una foto del perro",
        "prediction": "## 🎯 Predicción",
        "confidence": "Confianza",
        "analyzing": "🧠 La IA está analizando el perro...",
        "warning": "⚠️ El modelo no está muy seguro de esta predicción.",
        "top5": "🏆 Top 5 Predicciones",
        "confidence_gauge": "📊 Velocímetro de confianza Top-1",
        "history": "🕘 Historial de predicciones (sesión)",
        "clear_history": "Limpiar historial",
        "history_empty": "Aún no hay predicciones en esta sesión.",
        "footer": "🚀 Desarrollado con PyTorch, EfficientNetB0 y Streamlit",
        "model_not_found": "❌ No se encontró el modelo en '{model_path}'",
        "loading_error": "❌ Error al cargar el modelo",
        "sidebar_title": "🐾 Clasificador IA de Perros",
        "sidebar_features": "### Funciones\n✅ Deep Learning  \n✅ PyTorch  \n✅ Streamlit  \n✅ Top 5 predicciones  \n✅ Clasificación en tiempo real  \n✅ 55 razas de perro  \n✅ Soporte de cámara",
        "metric_accuracy": "Precisión",
        "metric_breeds": "Razas",
        "metric_model": "Modelo",
        "registered_title": "📚 Razas registradas en nuestro modelo",
        "registered_caption": "Explora las 55 razas disponibles en el modelo entrenado.",
        "camera_expander": "📸 Cámara (opcional)",
        "camera_help": "Usa la cámara solo si quieres capturar una foto en tiempo real.",
        "dog_image": "Imagen del perro",
        "col_breed": "Raza",
    },
}

t = TEXTS[language]

# Estilos visuales basicos de la app

st.markdown("""
<style>
.main { background-color: #0E1117; }
h1, h2, h3 { color: white; }
.stButton>button { border-radius: 12px; height: 3em; width: 100%; font-size: 18px; }
</style>
""", unsafe_allow_html=True)

# Clases del modelo en el mismo orden usado al entrenar

NUM_CLASSES = 55
IMG_SIZE    = 224
DISPLAY_IMAGE_SIZE = 700

# Mantener este orden evita errores al mapear indice -> raza
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

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

HISTORY_BREED_KEY = "breed"
HISTORY_CONFIDENCE_KEY = "confidence"

# Transformar imagen para que el modelo pueda analizarla

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),                          # [0,255] → [0,1]
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Crear arquitectura EfficientNetB0 para inferencia

class DogBreedClassifier(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        # Backbone preentrenado: extrae rasgos visuales de la imagen
        self.backbone = models.efficientnet_b0(weights=None)
        in_features = self.backbone.classifier[1].in_features
        # Cabezal final: convierte rasgos en probabilidades por raza
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
        # Entrada: tensor [batch, 3, 224, 224] / Salida: logits [batch, clases]
        return self.backbone(x)

# Ruta del archivo .pth entrenado

MODEL_PATH = "models/best_efficientnet_b0.pth"

@st.cache_resource
def load_model():
    # Cargar pesos entrenados en CPU o GPU disponible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = DogBreedClassifier(num_classes=NUM_CLASSES)
    sd     = torch.load(MODEL_PATH, map_location=device)
    # Soporta dos formatos: checkpoint completo o state_dict directo
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
    st.error(t["model_not_found"].format(model_path=MODEL_PATH))
    st.stop()
except Exception as e:
    st.error(t["loading_error"])
    st.error(str(e))
    st.stop()

# Sidebar con informacion rapida

st.sidebar.title(t["sidebar_title"])
st.sidebar.markdown(t["sidebar_features"])
# Interfaz principal

st.title(t["title"])
st.markdown(t["subtitle"])

col_a, col_b, col_c = st.columns(3)
col_a.metric(t["metric_accuracy"], "~90%")
col_b.metric(t["metric_breeds"], "55")
col_c.metric(t["metric_model"], "EfficientNetB0")

st.markdown("---")

st.subheader(t["registered_title"])
st.caption(t["registered_caption"])

st.markdown(
    """
    <style>
    .breed-pill {
        background: linear-gradient(135deg, rgba(45, 88, 160, 0.25), rgba(18, 35, 62, 0.45));
        border: 1px solid rgba(120, 170, 255, 0.25);
        border-radius: 12px;
        padding: 8px 12px;
        margin-bottom: 8px;
        color: #EAF2FF;
        font-weight: 500;
        line-height: 1.2;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

breeds_per_column = 19
breed_columns = st.columns(3)
for idx, column in enumerate(breed_columns):
    start = idx * breeds_per_column
    end = start + breeds_per_column
    with column:
        for breed in registered_breeds[start:end]:
            st.markdown(f"<div class='breed-pill'>{breed}</div>", unsafe_allow_html=True)

# Cargar imagen desde archivo o camara

uploaded_file = st.file_uploader(t["upload"], type=["jpg", "jpeg", "png"])

camera_image = None
with st.expander(t["camera_expander"], expanded=False):
    st.write(t["camera_help"])
    camera_image = st.camera_input(t["camera_input"])

# Realizar prediccion cuando exista una imagen

if uploaded_file is not None or camera_image is not None:

    col1, col2 = st.columns([1, 1])

    with col1:
        image = Image.open(
            uploaded_file if uploaded_file is not None else camera_image
        ).convert("RGB")
        display_image = ImageOps.fit(
            image,
            (DISPLAY_IMAGE_SIZE, DISPLAY_IMAGE_SIZE),
            method=Image.Resampling.LANCZOS,
        )
        st.image(display_image, caption=t["dog_image"], use_container_width=True)

    # Transformar imagen igual que en validacion
    tensor = transform(image).unsqueeze(0).to(device)

    with st.spinner(t["analyzing"]):
        with torch.no_grad():
            outputs = model(tensor)
            probs   = torch.softmax(outputs, dim=1).cpu().numpy()[0]

    predicted_index = int(np.argmax(probs))
    confidence      = float(probs[predicted_index])

    with col2:
        st.markdown(t["prediction"])
        st.success(f"### {registered_breeds[predicted_index]}")

        if confidence < 0.50:
            st.warning(t["warning"])

    # Top 5: mostrar las cinco razas con mayor probabilidad
    st.markdown("---")
    st.subheader(t["top5"])

    top5_idx = probs.argsort()[-5:][::-1]
    top_data = [
        {
            t["col_breed"]: registered_breeds[i],
            t["confidence"]: round(float(probs[i]) * 100, 2)
        }
        for i in top5_idx
    ]
    df = pd.DataFrame(top_data)

    fig = px.bar(
        df, x=t["confidence"], y=t["col_breed"], orientation="h",
        text=t["confidence"], title=t["top5"]
    )
    fig.update_layout(height=500, yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)

    top1_conf = float(df.iloc[0][t["confidence"]])

    # Guardar resultado top-1 en historial de la sesion
    st.session_state.prediction_history.append(
        {
            HISTORY_BREED_KEY: registered_breeds[predicted_index],
            HISTORY_CONFIDENCE_KEY: round(top1_conf, 2),
        }
    )

    st.markdown("---")
    # Mostrar confianza top-1 en formato velocimetro
    st.subheader(t["confidence_gauge"])
    gauge_fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=top1_conf,
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#4AA3FF"},
                "steps": [
                    {"range": [0, 50], "color": "#5A1F1F"},
                    {"range": [50, 80], "color": "#5A4A1F"},
                    {"range": [80, 100], "color": "#1F5A35"},
                ],
                "threshold": {
                    "line": {"color": "#FFFFFF", "width": 4},
                    "thickness": 0.8,
                    "value": top1_conf,
                },
            },
            title={"text": t["confidence"]},
        )
    )
    gauge_fig.update_layout(height=360)
    st.plotly_chart(gauge_fig, use_container_width=True)

    st.markdown("---")
    # Mostrar historial para comparar predicciones recientes
    st.subheader(t["history"])
    if st.button(t["clear_history"]):
        st.session_state.prediction_history = []
    if st.session_state.prediction_history:
        normalized_history = []
        for entry in st.session_state.prediction_history:
            if not isinstance(entry, dict):
                continue

            breed_value = (
                entry.get(HISTORY_BREED_KEY)
                or entry.get("Breed")
                or entry.get("Raza")
                or "-"
            )
            confidence_value = (
                entry.get(HISTORY_CONFIDENCE_KEY)
                if entry.get(HISTORY_CONFIDENCE_KEY) is not None
                else entry.get("Confidence", entry.get("Confianza", 0.0))
            )

            try:
                confidence_value = round(float(confidence_value), 2)
            except (TypeError, ValueError):
                confidence_value = 0.0

            normalized_history.append(
                {
                    t["col_breed"]: str(breed_value),
                    t["confidence"]: confidence_value,
                }
            )

        history_df = pd.DataFrame(normalized_history)
        st.dataframe(history_df.tail(10), use_container_width=True)
    else:
        st.caption(t["history_empty"])

# Pie de pagina

st.markdown("---")
st.caption(t["footer"])
