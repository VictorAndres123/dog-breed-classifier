"""
check_classes.py
Verifica el orden exacto de clases que usó el DataLoader durante el entrenamiento.
Ejecutar con: python check_classes.py
"""
import os
from pathlib import Path

# Rutas posibles del dataset dividido
CANDIDATES = ["data/train", "data/training", "dataset", "data"]

for path in CANDIDATES:
    p = Path(path)
    if p.exists():
        subdirs = sorted([d.name for d in p.iterdir() if d.is_dir()])
        if subdirs:
            print(f"✅ Encontrado: '{path}' — {len(subdirs)} clases")
            print()
            print("Orden exacto de clases (índice → nombre):")
            for i, name in enumerate(subdirs):
                print(f"  {i:2d}: {name}")
            print()

            # Comparar con el orden actual del app.py
            app_classes = [
                "afghan", "african_wild_dog", "airedale", "american_hairless",
                "basenji", "basset", "beagle", "bermaise", "blenheim", "bloodhound",
                "bluetick", "border_collie", "borzoi", "boston_terrier", "boxer",
                "bull_mastiff", "bulldog", "cairn", "chihuahua", "chow", "clumber",
                "cocker", "collie", "corgi", "dhole", "dingo", "doberman", "elk_hound",
                "german_sheperd", "golden_retriever", "great_dane", "great_perenees",
                "groenendael", "irish_wolfhound", "japanese_spaniel", "komondor",
                "labrador", "lhasa", "malinois", "maltese", "newfoundland", "pekinese",
                "pomeranian", "poodle", "pug", "rhodesian", "rottweiler", "saint_bernard",
                "schnauzer", "scotch_terrier", "Scottish_deerhound", "shih_tzu",
                "siberian_husky", "vizsla", "yorkie",
            ]

            print("─" * 40)
            mismatches = []
            for i, (real, app) in enumerate(zip(subdirs, app_classes)):
                if real != app:
                    mismatches.append((i, real, app))

            if not mismatches and len(subdirs) == len(app_classes):
                print("✅ El orden de clases en app.py es CORRECTO")
            else:
                print(f"❌ HAY {len(mismatches)} DIFERENCIAS:")
                for i, real, app in mismatches:
                    print(f"  índice {i:2d}: dataset='{real}'  app.py='{app}'")
                if len(subdirs) != len(app_classes):
                    print(f"  Cantidad: dataset={len(subdirs)}  app.py={len(app_classes)}")

                print()
                print("Lista corregida para copiar en app.py:")
                print("class_names = [")
                for i, name in enumerate(subdirs):
                    comma = "," if i < len(subdirs) - 1 else ""
                    print(f'    "{name}"{comma}')
                print("]")
            break
else:
    print("❌ No se encontró el directorio de datos.")
    print("Ejecuta este script desde la raíz del proyecto.")