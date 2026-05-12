import os
import shutil

# Dataset principal
main_dataset = "dataset"

# Dataset nuevo
new_dataset = "new_dataset"

print("🚀 Uniendo datasets...")

# Recorrer razas
for breed in os.listdir(new_dataset):

    new_breed_path = os.path.join(new_dataset, breed)
    main_breed_path = os.path.join(main_dataset, breed)

    # Verificar si es carpeta
    if os.path.isdir(new_breed_path):

        # Crear carpeta si no existe
        os.makedirs(main_breed_path, exist_ok=True)

        # Recorrer imágenes
        for image in os.listdir(new_breed_path):

            src = os.path.join(new_breed_path, image)
            dst = os.path.join(main_breed_path, image)

            try:
                shutil.copy2(src, dst)

            except Exception as e:
                print(f"❌ Error copiando {image}: {e}")

print("✅ Datasets unidos correctamente")