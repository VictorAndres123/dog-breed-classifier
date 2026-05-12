from pathlib import Path

dataset_path = r"c:\Users\ESTEBAN\Desktop\dog-breed-classifier\dataset"

breeds = sorted([
    d.name for d in Path(dataset_path).iterdir()
    if d.is_dir()
])

print("\n🐶 RAZAS ENCONTRADAS:\n")
for i, breed in enumerate(breeds):
    print(f"{i}: {breed}")

print(f"\nTotal razas: {len(breeds)}")