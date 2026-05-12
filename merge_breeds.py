import os
import shutil

# =========================================================
# DATASET PATH
# =========================================================

DATASET_PATH = "dataset"

# =========================================================
# EQUIVALENCIAS DE RAZAS
# izquierda = carpeta que quieres conservar
# derecha = carpetas que quieres fusionar
# =========================================================

breed_mapping = {

    "afghan": [
        "n02088094-Afghan_hound"
    ],

    "japanese_spaniel": [
        "n02085782-Japanese_spaniel"
    ],

    "maltese": [
        "n02085936-Maltese_dog"
    ],

    "pekinese": [
        "n02086079-Pekinese"
    ],

    "shih_tzu": [
        "n02086240-Shih-Tzu"
    ],

    "rhodesian": [
        "n02087394-Rhodesian_ridgeback"
    ],

    "basset": [
        "n02088238-basset"
    ],

    "beagle": [
        "n02088364-beagle"
    ],

    "bloodhound": [
        "n02088466-bloodhound"
    ],

    "bluetick": [
        "n02088632-bluetick"
    ],

    "borzoi": [
        "n02090622-borzoi"
    ],

    "irish_wolfhound": [
        "n02090721-Irish_wolfhound"
    ],

    "airedale": [
        "n02096051-Airedale"
    ],

    "cairn": [
        "n02096177-cairn"
    ],

    "scotch_terrier": [
        "n02097298-Scotch_terrier"
    ],

    "lhasa": [
        "n02098413-Lhasa"
    ],

    "golden_retriever": [
        "n02099601-golden_retriever"
    ],

    "labrador": [
        "n02099712-Labrador_retriever"
    ],

    "vizsla": [
        "n02100583-vizsla"
    ],

    "clumber": [
        "n02101556-clumber"
    ],

    "groenendael": [
        "n02105056-groenendael"
    ],

    "malinois": [
        "n02105162-malinois"
    ],

    "collie": [
        "n02106030-collie"
    ],

    "border_collie": [
        "n02106166-Border_collie"
    ],

    "rottweiler": [
        "n02106550-Rottweiler"
    ],

    "german_sheperd": [
        "n02106662-German_shepherd"
    ],

    "doberman": [
        "n02107142-Doberman"
    ],

    "boxer": [
        "n02108089-boxer"
    ],

    "bull_mastiff": [
        "n02108422-bull_mastiff"
    ],

    "french_bulldog": [
        "n02108915-French_bulldog"
    ],

    "great_dane": [
        "n02109047-Great_Dane"
    ],

    "saint_bernard": [
        "n02109525-Saint_Bernard"
    ],

    "siberian_husky": [
        "n02110185-Siberian_husky"
    ],

    "basenji": [
        "n02110806-basenji"
    ],

    "newfoundland": [
        "n02111277-Newfoundland"
    ],

    "pomeranian": [
        "n02112018-Pomeranian"
    ],

    "chow": [
        "n02112137-chow"
    ],

    "dingo": [
        "n02115641-dingo"
    ],

    "dhole": [
        "n02115913-dhole"
    ],

    "african_wild_dog": [
        "n02116738-African_hunting_dog"
    ],

    # Poodles
    "poodle": [
        "n02113624-toy_poodle",
        "n02113712-miniature_poodle",
        "n02113799-standard_poodle"
    ]
}

# =========================================================
# FUSIONAR CARPETAS
# =========================================================

for main_breed, duplicate_breeds in breed_mapping.items():

    main_path = os.path.join(DATASET_PATH, main_breed)

    # Crear carpeta principal si no existe
    os.makedirs(main_path, exist_ok=True)

    for duplicate in duplicate_breeds:

        duplicate_path = os.path.join(DATASET_PATH, duplicate)

        if not os.path.exists(duplicate_path):
            print(f"❌ No existe: {duplicate}")
            continue

        print(f"\n🔄 Fusionando {duplicate} -> {main_breed}")

        for image_name in os.listdir(duplicate_path):

            source = os.path.join(duplicate_path, image_name)
            destination = os.path.join(main_path, image_name)

            # evitar sobrescribir
            if not os.path.exists(destination):

                shutil.move(source, destination)

        # eliminar carpeta vacía
        shutil.rmtree(duplicate_path)

        print(f"✅ Fusionada y eliminada: {duplicate}")

print("\n🎉 DATASET LIMPIO Y UNIFICADO")