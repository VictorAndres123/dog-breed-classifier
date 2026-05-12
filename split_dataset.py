import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def split_dataset(dataset_path, output_path, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Divide el dataset en train/val/test manteniendo la estructura de clases.
    
    Args:
        dataset_path: Ruta a la carpeta del dataset
        output_path: Ruta donde guardar los subconjuntos
        train_ratio: Proporción para entrenamiento (default: 0.7)
        val_ratio: Proporción para validación (default: 0.15)
        test_ratio: Proporción para prueba (default: 0.15)
    """
    
    assert train_ratio + val_ratio + test_ratio == 1.0, "Las proporciones deben sumar 1.0"
    
    # Crear directorio de salida
    os.makedirs(output_path, exist_ok=True)
    
    # Crear subdirectorios
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_path, split), exist_ok=True)
    
    # Obtener lista de razas
    breeds = [d for d in os.listdir(dataset_path) 
              if os.path.isdir(os.path.join(dataset_path, d))]
    
    print(f"Encontradas {len(breeds)} razas\n")
    
    total_images = 0
    
    for breed in tqdm(breeds, desc="Dividiendo dataset"):
        breed_path = os.path.join(dataset_path, breed)
        images = [f for f in os.listdir(breed_path) 
                 if os.path.isfile(os.path.join(breed_path, f))]
        
        total_images += len(images)
        
        # Dividir índices
        indices = list(range(len(images)))
        
        # Primera división: train + (val+test)
        train_idx, remaining_idx = train_test_split(
            indices, 
            test_size=1-train_ratio, 
            random_state=42
        )
        
        # Segunda división: val y test
        val_test_ratio = val_ratio / (val_ratio + test_ratio)
        val_idx, test_idx = train_test_split(
            remaining_idx,
            test_size=1-val_test_ratio,
            random_state=42
        )
        
        # Crear carpeta de raza en cada split
        for split in ['train', 'val', 'test']:
            breed_split_path = os.path.join(output_path, split, breed)
            os.makedirs(breed_split_path, exist_ok=True)
        
        # Copiar imágenes
        for idx in train_idx:
            src = os.path.join(breed_path, images[idx])
            dst = os.path.join(output_path, 'train', breed, images[idx])
            shutil.copy2(src, dst)
        
        for idx in val_idx:
            src = os.path.join(breed_path, images[idx])
            dst = os.path.join(output_path, 'val', breed, images[idx])
            shutil.copy2(src, dst)
        
        for idx in test_idx:
            src = os.path.join(breed_path, images[idx])
            dst = os.path.join(output_path, 'test', breed, images[idx])
            shutil.copy2(src, dst)
    
    # Mostrar estadísticas
    print(f"\n✓ Dataset dividido exitosamente\n")
    print(f"Total de imágenes: {total_images}\n")
    
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(output_path, split)
        count = sum(len(files) for _, _, files in os.walk(split_path))
        percentage = (count / total_images) * 100
        print(f"{split.upper():8s}: {count:6d} imágenes ({percentage:5.1f}%)")
    
    print(f"\n✓ Dataset guardado en: {output_path}")

if __name__ == "__main__":
    dataset_path = r"c:\Users\ESTEBAN\Desktop\dog-breed-classifier\dataset"
    output_path = r"c:\Users\ESTEBAN\Desktop\dog-breed-classifier\data"
    
    split_dataset(dataset_path, output_path)
