import os

dataset_path = "dataset"

print("\n📊 IMAGES PER BREED\n")

total_images = 0

for breed in sorted(os.listdir(dataset_path)):

    breed_path = os.path.join(dataset_path, breed)

    if os.path.isdir(breed_path):

        count = len([
            file for file in os.listdir(breed_path)
            if file.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

        total_images += count

        print(f"🐶 {breed}: {count} images")

print("\n" + "=" * 50)
print(f"📸 TOTAL IMAGES: {total_images}")