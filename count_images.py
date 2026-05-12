import os

dataset_path = "dataset"

print("\n📊 IMÁGENES POR RAZA:\n")

for breed in os.listdir(dataset_path):

    breed_path = os.path.join(dataset_path, breed)

    if os.path.isdir(breed_path):

        total = len(os.listdir(breed_path))

        print(f"{breed}: {total} imágenes")