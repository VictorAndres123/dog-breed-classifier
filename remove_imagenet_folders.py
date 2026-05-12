import os
import shutil
from pathlib import Path

def remove_imagenet_folders(dataset_path):
    """Elimina automáticamente las carpetas con prefijo ImageNet (n020 y n021)"""
    
    folders = [f for f in os.listdir(dataset_path) 
               if os.path.isdir(os.path.join(dataset_path, f))]
    
    imagenet_folders = [f for f in folders if f.startswith('n020') or f.startswith('n021')]
    
    if not imagenet_folders:
        print("No se encontraron carpetas ImageNet (n020* o n021*) para eliminar.")
        return
    
    print(f"Se encontraron {len(imagenet_folders)} carpetas ImageNet:\n")
    for folder in sorted(imagenet_folders):
        print(f"  - {folder}")
    
    response = input("\n¿Deseas eliminar estas carpetas? (s/n): ").strip().lower()
    
    if response == 's':
        for folder in imagenet_folders:
            folder_path = os.path.join(dataset_path, folder)
            try:
                shutil.rmtree(folder_path)
                print(f"✓ Eliminada: {folder}")
            except Exception as e:
                print(f"✗ Error al eliminar {folder}: {e}")
        
        print(f"\n✓ Se eliminaron {len(imagenet_folders)} carpetas ImageNet.")
    else:
        print("Operación cancelada.")

if __name__ == "__main__":
    dataset_path = r"c:\Users\ESTEBAN\Desktop\dog-breed-classifier\dataset"
    remove_imagenet_folders(dataset_path)
