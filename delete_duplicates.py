from PIL import Image
import imagehash
import os

dataset_path = "dataset"

hashes = {}

deleted = 0

print("🔍 Searching and deleting duplicate images...\n")

for root, dirs, files in os.walk(dataset_path):

    for file in files:

        if file.lower().endswith((".jpg", ".jpeg", ".png")):

            path = os.path.join(root, file)

            try:
                img = Image.open(path)

                # Crear hash de la imagen
                img_hash = imagehash.average_hash(img)

                # Si ya existe -> eliminar duplicada
                if img_hash in hashes:

                    print(f"❌ Duplicate deleted:")
                    print(path)
                    print("-" * 70)

                    os.remove(path)

                    deleted += 1

                else:
                    hashes[img_hash] = path

            except Exception as e:
                print(f"⚠️ Error with {path}: {e}")

print(f"\n✅ Total duplicate images deleted: {deleted}")