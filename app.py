"""
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║                  🐶 AI DOG BREED CLASSIFIER - WEB APPLICATION             ║
║                                                                            ║
║  Clasificador de razas de perros con Deep Learning usando EfficientNetB0  ║
║  Interfaz web construida con Streamlit para predicciones en tiempo real    ║
║                                                                            ║
║  Autor: Ingeniería de Sistemas                                            ║
║  Tecnologías: TensorFlow, Streamlit, EfficientNetB0, Transfer Learning   ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
"""

# ═════════════════════════════════════════════════════════════════════════════
# 📚 IMPORTACIONES
# ═════════════════════════════════════════════════════════════════════════════
# Estas librerías son esenciales para el funcionamiento de la aplicación web:

import streamlit as st
# Streamlit: Framework de Python para crear aplicaciones web interactivas 
# sin necesidad de HTML/CSS/JavaScript. Permite crear dashboards, interfaces
# de usuario y aplicaciones de Machine Learning de forma rápida y sencilla.

import tensorflow as tf
# TensorFlow: Librería de Google para Machine Learning y Deep Learning.
# Proporciona herramientas para crear, entrenar y usar redes neuronales.
# Es la base de nuestro modelo de clasificación de razas de perros.

import numpy as np
# NumPy: Librería fundamental para computación numérica en Python.
# Permite trabajar con arrays multidimensionales y operaciones matemáticas.
# Usamos NumPy para procesar las imágenes como arrays numéricos.

import pandas as pd
# Pandas: Librería para análisis y manipulación de datos estructurados.
# Permite crear DataFrames (tablas) para mostrar resultados organizados.

from PIL import Image
# PIL (Python Imaging Library): Librería para procesamiento de imágenes.
# Usamos PIL para cargar, convertir y redimensionar imágenes desde archivos.

from tensorflow.keras.preprocessing import image_dataset_from_directory
# image_dataset_from_directory: Función de TensorFlow que carga imágenes 
# desde directorios y las organiza en lotes (batches).
# En la app, la usamos para extraer los nombres de las clases (razas).

# ═════════════════════════════════════════════════════════════════════════════
# 🎨 CONFIGURACIÓN DE PÁGINA STREAMLIT
# ═════════════════════════════════════════════════════════════════════════════
# st.set_page_config() configura la apariencia y comportamiento de la página web.
# Aquí definimos el título, ícono y diseño de la aplicación.

st.set_page_config(
    page_title="🐶 AI Dog Breed Classifier",  # Título que aparece en la pestaña del navegador
    page_icon="🐾",                           # Ícono que aparece en la pestaña del navegador
    layout="wide"                             # Diseño: "centered" o "wide". Wide usa todo el ancho disponible
)

# ═════════════════════════════════════════════════════════════════════════════
# 🎨 ESTILOS PERSONALIZADOS CON CSS
# ═════════════════════════════════════════════════════════════════════════════
# Streamlit permite personalizar la apariencia usando HTML y CSS.
# Los estilos aplicados aquí mejoran la interfaz visual de la aplicación.

st.markdown("""
<style>

/* Fondo oscuro para toda la aplicación */
.main {
    background-color: #0E1117;  /* Color de fondo gris oscuro (tema oscuro profesional) */
}

/* Títulos en color blanco para mejor contraste */
h1, h2, h3 {
    color: white;  /* Los títulos se ven claramente sobre el fondo oscuro */
}

/* Botones personalizados */
.stButton>button {
    border-radius: 12px;    /* Esquinas redondeadas para un diseño moderno */
    height: 3em;            /* Altura del botón */
    width: 100%;            /* Ancho completo del contenedor */
    font-size: 18px;        /* Tamaño de fuente legible */
}

/* Caja de predicción personalizada */
.prediction-box {
    padding: 20px;                  /* Espacio interior de la caja */
    border-radius: 15px;            /* Esquinas muy redondeadas */
    background-color: #1E1E1E;      /* Fondo ligeramente más claro que el fondo principal */
    margin-top: 20px;               /* Espacio superior */
}

</style>
""", unsafe_allow_html=True)  # unsafe_allow_html=True permite ejecutar código HTML/CSS en Streamlit

# ═════════════════════════════════════════════════════════════════════════════
# ⚙️ CONFIGURACIÓN DEL MODELO Y DATASET
# ═════════════════════════════════════════════════════════════════════════════
# Estas variables definen los parámetros clave de la aplicación.
# Son constantes que afectan cómo procesamos las imágenes y el modelo.

IMG_SIZE = (224, 224)
# Tamaño de las imágenes: (ancho, alto) en píxeles.
# 224x224 es el tamaño estándar requerido por EfficientNetB0.
# EfficientNetB0 fue entrenado con imágenes de este tamaño específico,
# por lo que todas las imágenes de entrada deben redimensionarse a este tamaño.
# Imágenes más grandes = más detalles pero más lento.
# Imágenes más pequeñas = más rápido pero menos detalle.

BATCH_SIZE = 16
# Batch Size: número de imágenes que se procesan juntas en una pasada.
# Con batch_size=16, procesamos 16 imágenes a la vez desde el dataset.
# Esto es principalmente relevante cuando cargamos el dataset para extraer
# los nombres de las clases. En predicciones individuales usamos una sola imagen.

# ═════════════════════════════════════════════════════════════════════════════
# 🧠 CARGAR MODELO ENTRENADO
# ═════════════════════════════════════════════════════════════════════════════
# El modelo ya fue entrenado en train.py y guardado como "dog_classifier.keras".
# Aquí lo cargamos para usarlo en predicciones.

@st.cache_resource
# @st.cache_resource: Decorador que hace caché del modelo en memoria.
# Esto significa que el modelo se carga UNA SOLA VEZ, aunque se actualice 
# la página. Sin esto, se recargara cada vez que interactúes con la app,
# lo que sería lentísimo. Es una optimización crucial en Streamlit.
def load_model():
    """
    Carga el modelo entrenado de clasificación de razas de perros.
    
    El modelo fue entrenado usando:
    - Arquitectura: EfficientNetB0 (Transfer Learning)
    - Datos: 70 razas de perros diferentes
    - Accuracy: ~93% en validación
    
    Returns:
        tf.keras.Model: Modelo compilado y listo para hacer predicciones
    """

    model = tf.keras.models.load_model(
        "models/dog_classifier.keras"  # Ruta donde guardamos el modelo entrenado
    )

    return model

model = load_model()  # Cargamos el modelo una sola vez gracias al caché

# ═════════════════════════════════════════════════════════════════════════════
# 📂 CARGAR NOMBRES DE RAZAS (CLASES)
# ═════════════════════════════════════════════════════════════════════════════
# El modelo produce 70 números (uno por cada raza). Necesitamos saber
# cuál es el nombre de cada raza para mostrar resultados legibles.

@st.cache_resource
def load_class_names():
    """
    Extrae los nombres de las razas leyendo la estructura del dataset.
    
    Los nombres de las clases se obtienen de los directorios en "dataset/".
    Cada subdirectorio es una clase (raza), y su nombre se usa como etiqueta.
    
    Returns:
        list: Lista de nombres de razas, ej: ["beagle", "boxer", "bulldog", ...]
    """

    dataset = image_dataset_from_directory(
        "dataset",                          # Ruta del directorio principal del dataset
        image_size=IMG_SIZE,                # Redimensionar todas las imágenes a 224x224
        batch_size=BATCH_SIZE,              # Procesar en lotes de 16
        shuffle=False                       # No mezclar (no necesitamos entrenar, solo leer nombres)
    )

    return dataset.class_names  # Retorna los nombres de los directorios como lista

class_names = load_class_names()  # Cargamos los nombres de razas una sola vez

# ═════════════════════════════════════════════════════════════════════════════
# 📚 INFORMACIÓN ADICIONAL DE RAZAS
# ═════════════════════════════════════════════════════════════════════════════
# Diccionario con información educativa de algunas razas populares.
# Se muestra junto con la predicción para dar contexto al usuario.

breed_info = {
    # Cada entrada es: "nombre_clase": "Descripción/Información"
    
    "golden_retriever":
    "🐕 Friendly, intelligent and devoted dog.",  # Golden Retriever: Perro amigable e inteligente

    "poodle":
    "🐩 Extremely intelligent and elegant breed.",  # Poodle: Raza extremadamente inteligente y elegante

    "beagle":
    "🐶 Curious and energetic hunting dog.",  # Beagle: Perro de caza curioso y energético

    "rottweiler":
    "🛡️ Strong, loyal and protective dog.",  # Rottweiler: Perro fuerte, leal y protector

    "siberian_husky":
    "❄️ Energetic sled dog with wolf-like appearance.",  # Husky Siberiano: Perro de trineo con aspecto de lobo
}

# ═════════════════════════════════════════════════════════════════════════════
# 📋 BARRA LATERAL (SIDEBAR)
# ═════════════════════════════════════════════════════════════════════════════
# La barra lateral es un panel lateral en Streamlit que muestra información
# y opciones. Es perfecta para mostrar características de la aplicación.

st.sidebar.title("🐾 AI Dog Classifier")
# Título de la barra lateral

st.sidebar.markdown("""
### Features

✅ 70 Dog Breeds          # La aplicación puede identificar 70 razas diferentes
✅ Deep Learning          # Usa redes neuronales profundas (Deep Learning)
✅ EfficientNetB0         # Arquitectura de red neuronal eficiente
✅ 93% Accuracy           # Alcanza 93% de precisión en predicciones
✅ Top 5 Predictions      # Muestra las 5 predicciones más probables
✅ Transfer Learning      # Usa conocimiento pre-entrenado en ImageNet
""")
# Este markdown muestra las características principales de la app
# para que los usuarios sepan qué esperar

st.sidebar.info(
    "Upload a dog image and let the AI identify the breed."
    # Mensaje informativo que explica cómo usar la aplicación
)

# ═════════════════════════════════════════════════════════════════════════════
# 🏠 INTERFAZ PRINCIPAL
# ═════════════════════════════════════════════════════════════════════════════
# Aquí diseñamos la página principal de la aplicación

st.title("🐶 AI Dog Breed Classifier")
# Título principal de la página

st.markdown("""
Upload an image of a dog and the AI model will try to identify the breed.
# Descripción de la funcionalidad principal
""")

# ═════════════════════════════════════════════════════════════════════════════
# 📤 WIDGET DE CARGA DE ARCHIVOS
# ═════════════════════════════════════════════════════════════════════════════
# st.file_uploader() es un widget interactivo que permite al usuario
# seleccionar un archivo de su computadora y subirlo a la aplicación.

uploaded_file = st.file_uploader(
    "📤 Upload a dog image",              # Etiqueta del widget
    type=["jpg", "jpeg", "png"]           # Solo permite estos formatos de imagen
    # jpg/jpeg: Formato comprimido común para fotos
    # png: Formato sin pérdida que soporta transparencia
)

# ═════════════════════════════════════════════════════════════════════════════
# 🔍 PROCESAMIENTO Y PREDICCIÓN
# ═════════════════════════════════════════════════════════════════════════════
# Este bloque se ejecuta SOLO si el usuario ha subido una imagen.
# Es el corazón de la aplicación donde ocurre la predicción.

if uploaded_file is not None:
    # uploaded_file no es None significa que el usuario subió una imagen
    
    col1, col2 = st.columns([1, 1])
    # st.columns() divide la página en columnas para un diseño de dos lados
    # [1, 1] significa dos columnas de igual ancho
    
    # ====================================================================
    # 📸 COLUMNA 1: VISUALIZAR LA IMAGEN SUBIDA
    # ====================================================================
    # Esta columna muestra la imagen que el usuario subió para que pueda
    # verificar que se cargó correctamente.

    with col1:
        # 'with' abre un bloque de contexto para la columna 1

        image = Image.open(uploaded_file).convert("RGB")
        # Image.open(): Abre la imagen del archivo subido
        # .convert("RGB"): Convierte la imagen a modo RGB
        #   RGB = 3 canales: Rojo, Verde, Azul
        #   Si la imagen es escala de grises (1 canal) o RGBA (4 canales),
        #   la convierte a RGB para que el modelo pueda procesarla
        # EfficientNetB0 espera imágenes RGB de 3 canales

        st.image(
            image,                          # La imagen a mostrar
            caption="Uploaded Image",        # Título debajo de la imagen
            use_container_width=True        # Usar todo el ancho disponible
        )
        # st.image() es un widget de Streamlit para mostrar imágenes

    # ====================================================================
    # 🔄 PREPROCESAMIENTO DE LA IMAGEN
    # ====================================================================
    # Antes de enviar la imagen al modelo, debemos preprocesarla.
    # El modelo espera datos en un formato específico.

    image_resized = image.resize(IMG_SIZE)
    # resize(): Redimensiona la imagen al tamaño requerido (224, 224)
    # Independientemente del tamaño original, la reducimos/agrandamos
    # a 224x224, que es el tamaño que EfficientNetB0 espera
    # Esto es CRÍTICO porque si envías una imagen de tamaño diferente,
    # el modelo fallará.

    image_array = np.array(image_resized)
    # Convierte la imagen PIL en un array NumPy
    # En este momento, image_array tiene forma (224, 224, 3)
    # 224 x 224 píxeles, 3 canales de color (RGB)
    # Cada valor está entre 0-255

    image_array = np.expand_dims(image_array, axis=0)
    # expand_dims(): Agrega una dimensión al principio del array
    # Cambia de (224, 224, 3) a (1, 224, 224, 3)
    # El primer número es el BATCH SIZE
    # EfficientNetB0 espera un lote de imágenes, incluso si es solo una
    # Por eso necesitamos este extra 1 al principio

    # ====================================================================
    # 🧠 HACER LA PREDICCIÓN
    # ====================================================================
    # Ahora enviamos la imagen preprocesada al modelo para obtener
    # la predicción de qué raza es el perro.

    with st.spinner("🧠 AI is analyzing the dog..."):
        # st.spinner(): Muestra un mensaje de carga con animación
        # mientras se ejecuta el código dentro
        
        prediction = model.predict(image_array)
        # model.predict(): Envía la imagen al modelo y obtiene las predicciones
        # Retorna un array con 70 números (uno por cada raza)
        # Cada número es una probabilidad entre 0-1
        # Ejemplo: [0.001, 0.05, 0.92, 0.001, ...] significa:
        #   - Raza 0: 0.1% probabilidad
        #   - Raza 1: 5% probabilidad
        #   - Raza 2: 92% probabilidad (la más probable)
        #   - Raza 3: 0.1% probabilidad
        #   - ... etc
        # Todas las probabilidades suman 1.0 (100%)

    predicted_index = np.argmax(prediction)
    # np.argmax(): Encuentra el índice del valor más grande
    # Si prediction = [0.001, 0.05, 0.92, 0.001, ...]
    # argmax retorna 2 (el índice de 0.92, el valor más grande)

    predicted_class = class_names[predicted_index]
    # Usa el índice para obtener el nombre de la raza
    # class_names[2] = "bulldog" (por ejemplo)

    confidence = np.max(prediction)
    # np.max(): Obtiene el valor más grande del array de predicción
    # Si prediction = [0.001, 0.05, 0.92, 0.001, ...]
    # max retorna 0.92
    # Este es el nivel de confianza de la predicción

    # ====================================================================
    # 📊 COLUMNA 2: MOSTRAR RESULTADOS DE LA PREDICCIÓN
    # ====================================================================
    # Esta columna muestra los resultados: raza predicha, confianza, info

    with col2:
        # 'with' abre el bloque de contexto para la columna 2

        st.markdown("## 🎯 Prediction")
        # Encabezado de la sección de resultados

        st.success(
            f"### {predicted_class.replace('_', ' ').title()}"
            # Muestra la raza predicha en una caja verde de éxito
            # .replace('_', ' '): Cambia "german_shepherd" a "german shepherd"
            # .title(): Capitaliza cada palabra: "german shepherd" a "German Shepherd"
        )

        st.metric(
            "Confidence",
            f"{confidence*100:.2f}%"
            # Muestra la confianza en porcentaje con 2 decimales
            # confidence*100: Convierte 0.92 a 92
            # :.2f: Formato con 2 decimales, ej: 92.15%
        )

        # ============================================================
        # ⚠️ ALERTA DE BAJA CONFIANZA
        # ============================================================
        # Si el modelo no está muy seguro (< 50% confianza),
        # mostramos una advertencia al usuario

        if confidence < 0.50:
            # Si la confianza es menor al 50%
            
            st.warning(
                "⚠️ The model is not very confident about this prediction."
                # Advertencia en una caja amarilla
                # Indica que quizás la predicción no es confiable
                # Esto puede ocurrir si:
                #   - La imagen es de mala calidad
                #   - El perro no es claramente visible
                #   - Es una raza no común
                #   - Hay mucho fondo en la imagen
            )

        # ============================================================
        # ℹ️ INFORMACIÓN ADICIONAL DE LA RAZA
        # ============================================================
        # Si tenemos información adicional de la raza predicha,
        # la mostramos para darle contexto al usuario

        if predicted_class in breed_info:
            # Verifica si la raza predicha está en nuestro diccionario
            # de información
            
            st.info(
                breed_info[predicted_class]
                # Muestra información de la raza en una caja azul informativa
            )

    # ====================================================================
    # 🏆 TOP 5 PREDICCIONES
    # ====================================================================
    # Además de la predicción principal, mostramos las 5 razas más probables
    # Esto ayuda al usuario a entender la confianza del modelo
    # y ver otras alternativas que el modelo considera

    st.markdown("---")
    # Línea divisoria horizontal (markdown) para separar secciones

    st.subheader("🏆 Top 5 Predictions")
    # Encabezado de la sección "Top 5 Predicciones"

    top_5_indices = prediction[0].argsort()[-5:][::-1]
    # prediction[0]: Toma el primer (y único) elemento del lote
    #   Cambios de forma (1, 70) a (70,)
    # .argsort(): Ordena los índices de menor a mayor probabilidad
    #   Retorna algo como [65, 32, 45, 12, 2] (índices ordenados)
    # [-5:]: Toma los últimos 5 elementos (los más grandes/probables)
    # [::-1]: Invierte el orden (de mayor a menor)
    # Resultado: índices de las 5 razas más probables, de mayor a menor

    top_data = []
    # Lista para almacenar los datos de las top 5 predicciones

    for idx in top_5_indices:
        # Itera sobre cada uno de los 5 índices

        breed = class_names[idx]
        # Obtiene el nombre de la raza usando el índice

        prob = float(prediction[0][idx] * 100)
        # Obtiene la probabilidad de esa raza (convertida a porcentaje)
        # float(): Convierte de tipo TensorFlow a número Python

        top_data.append({
            "Breed": breed.replace("_", " ").title(),  # Nombre formateado
            "Confidence (%)": round(prob, 2)            # Porcentaje redondeado
        })
        # Agrega un diccionario a top_data con el nombre y confianza

        st.progress(prob / 100)
        # Muestra una barra de progreso que representa el porcentaje
        # Valores entre 0-1, por eso dividimos entre 100

        st.write(
            f"**{breed.replace('_', ' ').title()}** — {prob:.2f}%"
            # Muestra texto con el nombre de la raza y su porcentaje
            # :.2f format muestra 2 decimales
        )

    # ====================================================================
    # 📊 TABLA DE TOP 5 PREDICCIONES
    # ====================================================================
    # Organiza los datos en una tabla para una visualización clara

    df = pd.DataFrame(top_data)
    # Crea un DataFrame (tabla) a partir de la lista de diccionarios

    st.dataframe(
        df,
        use_container_width=True  # Usa todo el ancho disponible
    )
    # Muestra la tabla de forma interactiva en Streamlit

# ═════════════════════════════════════════════════════════════════════════════
# 🔚 PIE DE PÁGINA (FOOTER)
# ═════════════════════════════════════════════════════════════════════════════
# Información sobre las tecnologías utilizadas para construir la app

st.markdown("---")
# Línea divisoria horizontal

st.caption(
    "🚀 Built with TensorFlow, EfficientNetB0 and Streamlit"
    # Texto pequeño (caption) que muestra las tecnologías principales
)