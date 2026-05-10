import os

dataset_path = "dataset"

for folder in os.listdir(dataset_path):

    old_path = os.path.join(dataset_path, folder)

    if os.path.isdir(old_path):

        new_name = folder.lower()

        new_name = new_name.replace(" ", "_")

        new_name = new_name.replace("-", "_")

        new_path = os.path.join(dataset_path, new_name)

        os.rename(old_path, new_path)

        print(f"{folder} -> {new_name}")

print("\nCarpetas renombradas correctamente")