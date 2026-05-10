"""
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║    🐶 MÓDULO DE CARGA DE DATOS - DOG BREED CLASSIFIER                    ║
║                                                                            ║
║  Módulo destinado a funciones de carga y preprocesamiento de datos.       ║
║  Actualmente las funciones están integradas en train.py                   ║
║  Este archivo puede extenderse con:                                       ║
║  - Funciones de descarga de datasets                                      ║
║  - Aumento de data                                                        ║
║  - Validación de datos                                                    ║
║  - Funciones de preprocesamiento avanzado                                 ║
║                                                                            ║
║  Autor: Ingeniería de Sistemas                                            ║
║  Tecnologías: TensorFlow, Pandas, NumPy                                   ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
"""

import tensorflow as tf
import numpy as np
import pandas as pd
from pathlib import Path

# ═════════════════════════════════════════════════════════════════════════════
# 🔧 FUNCIONES AUXILIARES DE CARGA DE DATOS
# ═════════════════════════════════════════════════════════════════════════════

def load_dataset_from_directory(directory: str, image_size: tuple = (224, 224), 
                                batch_size: int = 16, validation_split: float = 0.2):
    """
    Carga imágenes desde directorios y las organiza en training/validation sets.
    
    Uso de Transfer Learning: Las imágenes se cargan desde una estructura de directorios
    donde cada subdirectorio es una clase (raza de perro).
    
    Args:
        directory (str): Ruta al directorio raíz con subdirectorios de clases
        image_size (tuple): Tamaño al que redimensionar las imágenes (default: 224x224)
        batch_size (int): Número de imágenes por lote (default: 16)
        validation_split (float): Proporción de datos para validación (default: 0.2 = 20%)
    
    Returns:
        tuple: (train_dataset, validation_dataset, class_names)
    
    Ejemplo:
        >>> train_ds, val_ds, classes = load_dataset_from_directory("dataset")
        >>> print(classes)  # ['beagle', 'boxer', 'bulldog', ...]
    """
    
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        directory,
        validation_split=validation_split,
        subset="training",
        seed=42,
        image_size=image_size,
        batch_size=batch_size
    )
    
    validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        directory,
        validation_split=validation_split,
        subset="validation",
        seed=42,
        image_size=image_size,
        batch_size=batch_size
    )
    
    class_names = train_dataset.class_names
    
    return train_dataset, validation_dataset, class_names


def apply_prefetch_optimization(dataset, autotune=tf.data.AUTOTUNE):
    """
    Optimiza la carga de datos usando prefetch.
    
    Prefetch: Carga el siguiente lote mientras se procesa el actual
    Beneficio: La GPU/CPU nunca espera a que se carguen los datos
    
    Args:
        dataset: Dataset de TensorFlow
        autotune: Configuración automática (default: tf.data.AUTOTUNE)
    
    Returns:
        Dataset optimizado
    
    Ejemplo:
        >>> train_ds = apply_prefetch_optimization(train_ds)
    """
    return dataset.prefetch(buffer_size=autotune)


def get_class_statistics(directory: str) -> pd.DataFrame:
    """
    Obtiene estadísticas sobre las clases en el dataset.
    
    Calcula para cada clase (raza):
    - Número de imágenes
    - Proporción respecto al total
    
    Útil para detectar desbalance de clases.
    
    Args:
        directory (str): Ruta al directorio del dataset
    
    Returns:
        pd.DataFrame: DataFrame con estadísticas
    
    Ejemplo:
        >>> stats = get_class_statistics("dataset")
        >>> print(stats)
    """
    
    class_counts = {}
    total_images = 0
    
    for breed_dir in Path(directory).iterdir():
        if breed_dir.is_dir():
            count = len(list(breed_dir.glob("*")))
            class_counts[breed_dir.name] = count
            total_images += count
    
    df = pd.DataFrame(list(class_counts.items()), columns=["Breed", "Count"])
    df["Proportion (%)"] = (df["Count"] / total_images * 100).round(2)
    df = df.sort_values("Count", ascending=False)
    
    return df


# ═════════════════════════════════════════════════════════════════════════════
# 💡 NOTAS SOBRE TRANSFER LEARNING Y DATA LOADING
# ═════════════════════════════════════════════════════════════════════════════

"""
📚 ¿POR QUÉ image_dataset_from_directory ES ÚTIL?

1. CARGA AUTOMÁTICA:
   - Lee imágenes automáticamente desde carpetas
   - No necesitas iterar manualmente
   
2. NORMALIZACIÓN AUTOMÁTICA:
   - Convierte píxeles de 0-255 a 0-1
   - Prepara datos para el modelo
   
3. ORGANIZACIÓN DE LOTES:
   - Agrupa imágenes en batches
   - Permite entrenamiento eficiente
   
4. BALANCEO:
   - Si hay desbalance, puede muestrear estratégicamente
   - Opcional, pero útil

🔧 OPTIMIZACIONES IMPLEMENTADAS:

1. PREFETCH:
   - Carga siguiente batch mientras procesa actual
   - Acelera entrenamiento significativamente

2. SEED:
   - Asegura reproducibilidad
   - Mismo split train/val cada vez

3. BATCH SIZE:
   - 16 es balance entre memoria y estabilidad
   - Mayor = más estable pero más memoria
   - Menor = menos memoria pero más ruidoso
"""
