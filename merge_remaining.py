import os
import shutil

DATASET_PATH = "dataset"

breed_mapping = {

    "chihuahua": [
        "n02085620-Chihuahua"
    ],

    "boston_terrier": [
        "n02096585-Boston_bull"
    ],

    "pug": [
        "n02110958-pug"
    ],

    "mex_hairless": [
        "n02113978-Mexican_hairless"
    ],

    "great_perenees": [
        "n02111500-Great_Pyrenees"
    ]
}

for main_breed, duplicates in breed_mapping.items():

    main_path = os.path.join(DATASET_PATH, main_breed)

    os.makedirs(main_path, exist_ok=True)

    for duplicate in duplicates:

        duplicate_path = os.path.join(DATASET_PATH, duplicate)

        if not os.path.exists(duplicate_path):
            continue

        print(f"\n🔄 Fusionando {duplicate} -> {main_breed}")

        for file in os.listdir(duplicate_path):

            source = os.path.join(duplicate_path, file)
            destination = os.path.join(main_path, file)

            if not os.path.exists(destination):
                shutil.move(source, destination)

        shutil.rmtree(duplicate_path)

        print(f"✅ Eliminada: {duplicate}")

print("\n🎉 SEGUNDA LIMPIEZA COMPLETADA")