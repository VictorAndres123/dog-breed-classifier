from PIL import Image
import imagehash
import os

dataset_path = "dataset"

hashes = {}

duplicates = []

print("🔍 Searching for duplicate images...\n")

for root, dirs, files in os.walk(dataset_path):

    for file in files:

        if file.lower().endswith((".jpg", ".jpeg", ".png")):

            path = os.path.join(root, file)

            try:
                img = Image.open(path)

                img_hash = imagehash.average_hash(img)

                # SI YA EXISTE
                if img_hash in hashes:

                    original = hashes[img_hash]

                    duplicates.append((original, path))

                else:
                    hashes[img_hash] = path

            except Exception as e:
                print(f"❌ Error with {path}: {e}")

print(f"\n📊 Duplicate images found: {len(duplicates)}\n")

# MOSTRAR DUPLICADOS
for original, duplicate in duplicates:

    print("🟢 ORIGINAL:")
    print(original)

    print("🔴 DUPLICATE:")
    print(duplicate)

    print("-" * 80)