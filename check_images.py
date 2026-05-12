from PIL import Image
import os

dataset_path = "dataset"

bad_images = []

for root, dirs, files in os.walk(dataset_path):

    for file in files:

        if file.lower().endswith((".jpg", ".jpeg", ".png")):

            file_path = os.path.join(root, file)

            try:
                img = Image.open(file_path)
                img.verify()

            except Exception:
                bad_images.append(file_path)

print(f"\n❌ Damaged images found: {len(bad_images)}\n")

for img in bad_images:
    print(img)