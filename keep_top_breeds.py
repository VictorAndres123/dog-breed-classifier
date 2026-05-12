import os
import shutil
from pathlib import Path
from collections import defaultdict

def count_images_by_breed(dataset_path):
    """Cuenta el número de imágenes por raza"""
    breed_counts = {}
    
    for breed_folder in os.listdir(dataset_path):
        breed_path = os.path.join(dataset_path, breed_folder)
        
        if os.path.isdir(breed_path):
            images = [f for f in os.listdir(breed_path) 
                     if os.path.isfile(os.path.join(breed_path, f))]
            breed_counts[breed_folder] = len(images)
    
    return breed_counts

def keep_top_breeds(dataset_path, num_breeds=55):
    """Mantiene solo las N razas con más imágenes"""
    
    breed_counts = count_images_by_breed(dataset_path)
    
    # Ordena por número de imágenes (descendente)
    sorted_breeds = sorted(breed_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Obtiene las top N razas
    top_breeds = [breed for breed, count in sorted_breeds[:num_breeds]]
    breeds_to_remove = [breed for breed, count in sorted_breeds[num_breeds:]]
    
    print(f"Total de razas actuales: {len(breed_counts)}")
    print(f"Se mantendran: {num_breeds} razas")
    print(f"Se eliminaran: {len(breeds_to_remove)} razas\n")
    
    print("TOP 55 RAZAS (por número de imágenes):\n")
    for i, (breed, count) in enumerate(sorted_breeds[:num_breeds], 1):
        print(f"{i:2d}. {breed:30s} - {count:4d} imágenes")
    
    print("\n\nRAZAS A ELIMINAR:\n")
    for i, (breed, count) in enumerate(sorted_breeds[num_breeds:], 1):
        print(f"{i:2d}. {breed:30s} - {count:4d} imágenes")
    
    response = input("\n¿Deseas proceder con la eliminación? (s/n): ").strip().lower()
    
    if response == 's':
        total_deleted = 0
        for breed in breeds_to_remove:
            breed_path = os.path.join(dataset_path, breed)
            try:
                shutil.rmtree(breed_path)
                total_deleted += breed_counts[breed]
                print(f"✓ Eliminada: {breed} ({breed_counts[breed]} imágenes)")
            except Exception as e:
                print(f"✗ Error al eliminar {breed}: {e}")
        
        print(f"\n✓ Completado:")
        print(f"  - Razas eliminadas: {len(breeds_to_remove)}")
        print(f"  - Imágenes eliminadas: {total_deleted}")
        print(f"  - Razas restantes: {num_breeds}")
    else:
        print("Operación cancelada.")

if __name__ == "__main__":
    dataset_path = r"c:\Users\ESTEBAN\Desktop\dog-breed-classifier\dataset"
    keep_top_breeds(dataset_path, num_breeds=55)
