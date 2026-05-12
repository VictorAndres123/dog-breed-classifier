import os
import shutil
from pathlib import Path
from difflib import SequenceMatcher

def similarity(a, b):
    """Calcula similitud entre dos strings (0-1)"""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def extract_breed_name(folder_name):
    """Extrae el nombre de la raza limpiando prefijos y caracteres especiales"""
    # Si tiene prefijo ImageNet (n02...), extrae la parte después del guion
    if folder_name.startswith('n0'):
        parts = folder_name.split('-', 1)
        if len(parts) > 1:
            return parts[1].lower().replace('_', ' ')
    
    # Para nombres normales, reemplaza guiones con espacios
    return folder_name.lower().replace('_', ' ')

def find_duplicate_groups(dataset_path):
    """Encuentra grupos de carpetas que representan la misma raza"""
    folders = [f for f in os.listdir(dataset_path) 
               if os.path.isdir(os.path.join(dataset_path, f))]
    
    breed_names = {folder: extract_breed_name(folder) for folder in folders}
    
    grouped = {}
    used = set()
    
    for folder1 in folders:
        if folder1 in used:
            continue
        
        group = [folder1]
        breed1 = breed_names[folder1]
        
        for folder2 in folders:
            if folder2 == folder1 or folder2 in used:
                continue
            
            breed2 = breed_names[folder2]
            
            # Si la similitud es alta o son exactamente iguales, son duplicadas
            if breed1 == breed2 or similarity(breed1, breed2) > 0.85:
                group.append(folder2)
                used.add(folder2)
        
        if len(group) > 1:
            used.add(folder1)
            grouped[breed_names[folder1]] = group
    
    return grouped

def consolidate_breeds(dataset_path):
    """Consolida razas duplicadas"""
    duplicates = find_duplicate_groups(dataset_path)
    
    if not duplicates:
        print("No se encontraron razas duplicadas.")
        return
    
    print(f"Se encontraron {len(duplicates)} razas duplicadas:\n")
    
    for breed_name, folders in sorted(duplicates.items()):
        print(f"\n{breed_name.upper()}:")
        print(f"  Carpetas encontradas: {folders}")
        
        # Usa la primera carpeta como destino (preferiblemente la más simple)
        # Ordena para que las carpetas sin prefijo ImageNet vengan primero
        folders_sorted = sorted(folders, key=lambda x: (x.startswith('n0'), len(x)))
        target_folder = folders_sorted[0]
        source_folders = folders_sorted[1:]
        
        target_path = os.path.join(dataset_path, target_folder)
        
        print(f"  → Consolidando en: {target_folder}")
        
        total_images = 0
        for source_folder in source_folders:
            source_path = os.path.join(dataset_path, source_folder)
            
            # Cuenta y mueve imágenes
            images = [f for f in os.listdir(source_path) 
                     if os.path.isfile(os.path.join(source_path, f))]
            
            print(f"    Moviendo {len(images)} imágenes desde {source_folder}")
            total_images += len(images)
            
            for image in images:
                src = os.path.join(source_path, image)
                dst = os.path.join(target_path, image)
                
                # Si el archivo ya existe, genera un nombre único
                if os.path.exists(dst):
                    name, ext = os.path.splitext(image)
                    counter = 1
                    while os.path.exists(dst):
                        dst = os.path.join(target_path, f"{name}_{counter}{ext}")
                        counter += 1
                
                shutil.move(src, dst)
            
            # Elimina la carpeta vacía
            try:
                os.rmdir(source_path)
                print(f"    Carpeta {source_folder} eliminada")
            except OSError:
                print(f"    ⚠️  No se pudo eliminar {source_folder} (puede contener subcarpetas)")
        
        print(f"  Total de imágenes consolidadas: {total_images}")
    
    print(f"\n✓ Consolidación completada. Se procesaron {len(duplicates)} razas.")

if __name__ == "__main__":
    dataset_path = r"c:\Users\ESTEBAN\Desktop\dog-breed-classifier\dataset"
    
    # Primero muestra lo que encontró
    duplicates = find_duplicate_groups(dataset_path)
    
    if duplicates:
        print("RAZAS DUPLICADAS ENCONTRADAS:\n")
        for breed_name, folders in sorted(duplicates.items()):
            print(f"  {breed_name}: {folders}")
        
        response = input("\n¿Deseas proceder con la consolidación? (s/n): ").strip().lower()
        if response == 's':
            consolidate_breeds(dataset_path)
        else:
            print("Operación cancelada.")
    else:
        print("No se encontraron razas duplicadas.")
