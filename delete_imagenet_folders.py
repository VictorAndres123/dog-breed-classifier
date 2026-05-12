import os
import shutil

dataset_path = "dataset"

deleted = 0

print("\n🗑️ Removing ImageNet folders...\n")

for folder in os.listdir(dataset_path):

    if folder.startswith("n02"):

        folder_path = os.path.join(dataset_path, folder)

        try:
            shutil.rmtree(folder_path)

            print(f"❌ Deleted: {folder}")

            deleted += 1

        except Exception as e:
            print(f"⚠️ Error deleting {folder}: {e}")

print("\n" + "=" * 50)
print(f"✅ Total folders deleted: {deleted}")