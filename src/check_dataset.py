from PIL import Image
import os

dataset_path = "dataset"

bad_images = []
total_images = 0

for breed in os.listdir(dataset_path):

    breed_path = os.path.join(dataset_path, breed)

    if os.path.isdir(breed_path):

        print(f"\nVerificando carpeta: {breed}")

        for file in os.listdir(breed_path):

            file_path = os.path.join(breed_path, file)

            try:

                img = Image.open(file_path)

                img.verify()

                total_images += 1

            except Exception as e:

                print(f"Imagen dañada: {file_path}")

                bad_images.append(file_path)

print("\n==============================")

print(f"Total imágenes válidas: {total_images}")

print(f"Total imágenes dañadas: {len(bad_images)}")