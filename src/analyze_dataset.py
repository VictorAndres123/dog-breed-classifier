import os
import pandas as pd
from PIL import Image
from ydata_profiling import ProfileReport

dataset_path = "dataset"

data = []

for breed in os.listdir(dataset_path):

    breed_path = os.path.join(dataset_path, breed)

    if os.path.isdir(breed_path):

        image_count = 0

        for file in os.listdir(breed_path):

            path = os.path.join(breed_path, file)

            try:

                with Image.open(path) as img:

                    width, height = img.size

                    data.append({
                        "breed": breed,
                        "width": width,
                        "height": height,
                        "format": img.format,
                        "size_kb": round(os.path.getsize(path)/1024, 2)
                    })

                    image_count += 1

            except:
                pass

        print(f"{breed}: {image_count} imágenes")

df = pd.DataFrame(data)

print("\nPrimeras filas:")
print(df.head())

print("\nResumen estadístico:")
print(df.describe())

# REPORTE HTML
profile = ProfileReport(
    df,
    title="Dog Breed Dataset Analysis"
)

profile.to_file("reports/dataset_report.html")

print("\nReporte generado correctamente")