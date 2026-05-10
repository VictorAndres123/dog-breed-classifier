"""
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║      🛠️  FUNCIONES AUXILIARES - DOG BREED CLASSIFIER                     ║
║                                                                            ║
║  Módulo con funciones de utilidad reutilizables en todo el proyecto.      ║
║  Incluye:                                                                  ║
║  - Funciones de visualización                                             ║
║  - Funciones de procesamiento de imágenes                                 ║
║  - Funciones de evaluación                                                ║
║  - Funciones de utilidad general                                          ║
║                                                                            ║
║  Autor: Ingeniería de Sistemas                                            ║
║  Tecnologías: NumPy, Matplotlib, PIL                                     ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf

# ═════════════════════════════════════════════════════════════════════════════
# 🎨 FUNCIONES DE VISUALIZACIÓN
# ═════════════════════════════════════════════════════════════════════════════

def plot_training_history(history, figsize=(15, 5)):
    """
    Visualiza el historial de entrenamiento (accuracy y loss).
    
    Args:
        history: Objeto History retornado por model.fit()
        figsize: Tamaño de la figura (default: (15, 5))
    
    Ejemplo:
        >>> history = model.fit(train_ds, validation_data=val_ds, epochs=25)
        >>> plot_training_history(history)
    """
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Gráfica de Accuracy
    axes[0].plot(history.history['accuracy'], label='Training Accuracy', marker='o')
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', marker='s')
    axes[0].set_title('Model Accuracy', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Gráfica de Loss
    axes[1].plot(history.history['loss'], label='Training Loss', marker='o')
    axes[1].plot(history.history['val_loss'], label='Validation Loss', marker='s')
    axes[1].set_title('Model Loss', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def display_prediction_results(image_array, prediction, class_names, top_k=5):
    """
    Muestra una imagen y sus top K predicciones.
    
    Args:
        image_array: Array de la imagen (altura, ancho, 3)
        prediction: Array de predicciones (70,)
        class_names: Lista de nombres de clases
        top_k: Número de top predicciones a mostrar (default: 5)
    
    Ejemplo:
        >>> display_prediction_results(img_array, predictions, class_names)
    """
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Mostrar imagen
    axes[0].imshow(image_array.astype(int))
    axes[0].axis('off')
    axes[0].set_title('Input Image')
    
    # Top K predicciones
    top_k_indices = prediction.argsort()[-top_k:][::-1]
    top_k_probs = prediction[top_k_indices]
    top_k_classes = [class_names[i] for i in top_k_indices]
    
    y_pos = np.arange(len(top_k_classes))
    
    axes[1].barh(y_pos, top_k_probs)
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(top_k_classes)
    axes[1].set_xlabel('Probability')
    axes[1].set_title(f'Top {top_k} Predictions')
    axes[1].invert_yaxis()
    
    plt.tight_layout()
    plt.show()


def plot_class_distribution(class_counts, figsize=(12, 6)):
    """
    Visualiza la distribución de clases en el dataset.
    
    Útil para detectar desbalance de clases.
    
    Args:
        class_counts: Diccionario {clase: count}
        figsize: Tamaño de la figura
    
    Ejemplo:
        >>> class_counts = {'beagle': 150, 'boxer': 145, ...}
        >>> plot_class_distribution(class_counts)
    """
    
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    
    plt.figure(figsize=figsize)
    plt.bar(range(len(classes)), counts)
    plt.xticks(range(len(classes)), classes, rotation=45, ha='right')
    plt.ylabel('Number of Images')
    plt.title('Class Distribution in Dataset')
    plt.tight_layout()
    plt.show()


# ═════════════════════════════════════════════════════════════════════════════
# 🖼️  FUNCIONES DE PROCESAMIENTO DE IMÁGENES
# ═════════════════════════════════════════════════════════════════════════════

def load_and_preprocess_image(image_path, target_size=(224, 224), normalize=True):
    """
    Carga y preprocesa una imagen para predicción.
    
    Args:
        image_path: Ruta a la imagen
        target_size: Tamaño de salida (default: 224x224)
        normalize: Si normalizar a 0-1 (default: True)
    
    Returns:
        Array de imagen preprocesada
    
    Ejemplo:
        >>> img = load_and_preprocess_image("dog.jpg")
        >>> predictions = model.predict(np.expand_dims(img, 0))
    """
    
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img, dtype=np.float32)
    
    if normalize:
        img_array = img_array / 255.0
    
    return img_array


def batch_normalize_images(image_batch):
    """
    Normaliza un lote de imágenes (0-255 a 0-1).
    
    Args:
        image_batch: Array de imágenes (batch_size, h, w, 3)
    
    Returns:
        Lote normalizado
    """
    
    return image_batch.astype(np.float32) / 255.0


def apply_image_augmentation(image, seed=None):
    """
    Aplica transformaciones de data augmentation a una imagen.
    
    Transformaciones:
    - Flip horizontal (50%)
    - Rotación ±20°
    - Zoom ±20%
    
    Args:
        image: Imagen como array
        seed: Para reproducibilidad (optional)
    
    Returns:
        Imagen aumentada
    """
    
    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal", seed=seed),
        tf.keras.layers.RandomRotation(0.2, seed=seed),
        tf.keras.layers.RandomZoom(0.2, seed=seed),
    ])
    
    return augmentation(image)


# ═════════════════════════════════════════════════════════════════════════════
# 📊 FUNCIONES DE EVALUACIÓN
# ═════════════════════════════════════════════════════════════════════════════

def get_top_k_predictions(prediction, k=5, class_names=None):
    """
    Obtiene las top K predicciones.
    
    Args:
        prediction: Array de predicciones (70,)
        k: Número de top predicciones
        class_names: Lista de nombres de clases (opcional)
    
    Returns:
        Lista de tuplas (índice, probabilidad, nombre_clase)
    
    Ejemplo:
        >>> top_5 = get_top_k_predictions(predictions, k=5, class_names)
    """
    
    top_k_indices = prediction.argsort()[-k:][::-1]
    results = []
    
    for idx in top_k_indices:
        prob = float(prediction[idx])
        name = class_names[idx] if class_names else f"Class {idx}"
        results.append((idx, prob, name))
    
    return results


def confidence_level_interpretation(confidence):
    """
    Interpreta el nivel de confianza de una predicción.
    
    Args:
        confidence: Probabilidad entre 0-1
    
    Returns:
        Descripción en texto
    """
    
    if confidence > 0.95:
        return "Very High Confidence - Prediction is very reliable"
    elif confidence > 0.85:
        return "High Confidence - Prediction is reliable"
    elif confidence > 0.70:
        return "Moderate Confidence - Prediction is likely correct"
    elif confidence > 0.50:
        return "Low Confidence - Prediction may be incorrect"
    else:
        return "Very Low Confidence - Prediction is unreliable"


def calculate_metrics(y_true, y_pred):
    """
    Calcula métricas básicas de clasificación.
    
    Args:
        y_true: Etiquetas verdaderas
        y_pred: Etiquetas predichas
    
    Returns:
        Diccionario con métricas
    
    Ejemplo:
        >>> metrics = calculate_metrics(y_true, y_pred)
        >>> print(metrics['accuracy'])
    """
    
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1': f1_score(y_true, y_pred, average='weighted')
    }


# ═════════════════════════════════════════════════════════════════════════════
# 🛠️  FUNCIONES GENERALES DE UTILIDAD
# ═════════════════════════════════════════════════════════════════════════════

def format_breed_name(breed_name):
    """
    Formatea un nombre de raza para presentación.
    
    Ejemplo: "german_shepherd" -> "German Shepherd"
    
    Args:
        breed_name: Nombre de raza con underscores
    
    Returns:
        Nombre formateado
    """
    
    return breed_name.replace('_', ' ').title()


def get_model_info(model):
    """
    Obtiene información detallada del modelo.
    
    Args:
        model: Modelo de Keras
    
    Returns:
        Diccionario con información
    """
    
    total_params = model.count_params()
    trainable_params = sum([np.prod(w.shape) for w in model.trainable_weights])
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'non_trainable_params': total_params - trainable_params,
        'layers': len(model.layers),
        'model_name': model.name if hasattr(model, 'name') else 'Unknown'
    }


def print_model_summary(model):
    """
    Imprime un resumen detallado del modelo.
    
    Args:
        model: Modelo de Keras
    """
    
    info = get_model_info(model)
    
    print(f"""
    ╔════════════════════════════════════════════╗
    ║         MODEL INFORMATION SUMMARY          ║
    ╠════════════════════════════════════════════╣
    ║  Total Parameters:        {info['total_params']:>15,}  ║
    ║  Trainable Parameters:    {info['trainable_params']:>15,}  ║
    ║  Non-Trainable Parameters:{info['non_trainable_params']:>15,}  ║
    ║  Total Layers:            {info['layers']:>15}  ║
    ╚════════════════════════════════════════════╝
    """)
