"""
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║         🐶 VERIFICACIÓN DEL DATASET - DOG BREED CLASSIFIER               ║
║                                                                            ║
║  Script para verificar la integridad del dataset de imágenes.             ║
║  Detecta imágenes corruptas, dañadas o inválidas.                         ║
║                                                                            ║
║  Uso: python src/check_dataset.py                                        ║
║                                                                            ║
║  Autor: Ingeniería de Sistemas                                            ║
║  Tecnologías: PIL, Python                                                 ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝

██████████ ¿POR QUÉ VERIFICAR EL DATASET? ██████████

Los datasets pueden contener:
1. Imágenes corruptas: Archivos dañados, truncados o inválidos
2. Formatos inválidos: Archivos que no son realmente imágenes
3. Archivos huérfanos: Archivos sin extensión o con extensión incorrecta

Estos problemas causarían errores durante:
- Carga del dataset
- Entrenamiento del modelo
- Predicciones

Este script identifica estos problemas ANTES de entrenar.

██████████ QUÉ VERIFICA ██████████

✓ Itera sobre cada raza (subdirectorio)
✓ Para cada imagen:
  - Intenta abrirla con PIL
  - Verifica que sea un archivo válido
  - Valida la estructura interna
✓ Registra imágenes problemáticas
✓ Reporta estadísticas finales

██████████ CÓMO USAR ██████████

1. Ejecutar: python src/check_dataset.py
2. Revisar output para imágenes dañadas
3. Si hay imágenes dañadas:
   - Eliminarlas manualmente
   - O descargarlas nuevamente
4. Después, ya puedes entrenar el modelo con confianza
"""

# ═════════════════════════════════════════════════════════════════════════════
# 📚 IMPORTACIONES
# ═════════════════════════════════════════════════════════════════════════════

from PIL import Image
# PIL (Pillow): Librería de procesamiento de imágenes
# Usamos Image para:
# - Image.open(): Abre archivo de imagen
# - .verify(): Valida que sea una imagen válida

import os
# Módulo 'os': Para operaciones del sistema operativo
# Usamos os.listdir() para listar archivos
# Usamos os.path para manejar rutas

# ═════════════════════════════════════════════════════════════════════════════
# ⚙️ CONFIGURACIÓN
# ═════════════════════════════════════════════════════════════════════════════

print("""
╔═══════════════════════════════════════════════════════════════════════╗
║  🐶 VERIFICACIÓN DE DATASET - DOG BREED CLASSIFIER                   ║
║  Analizando integridad de archivos...                                ║
╚═══════════════════════════════════════════════════════════════════════╝
""")

dataset_path = "dataset"
# Ruta del directorio principal del dataset
# Estructura esperada:
# dataset/
#   ├── breed_1/
#   │   ├── image1.jpg
#   │   ├── image2.jpg
#   │   └── ...
#   ├── breed_2/
#   │   ├── image1.jpg
#   │   └── ...
#   └── ...

bad_images = []
# Lista para almacenar rutas de imágenes problemáticas

total_images = 0
# Contador total de imágenes válidas

# ═════════════════════════════════════════════════════════════════════════════
# 🔍 VERIFICAR CADA RAZA Y SUS IMÁGENES
# ═════════════════════════════════════════════════════════════════════════════

print(f"\n📂 Analizando directorio: {dataset_path}\n")

for breed in os.listdir(dataset_path):
    # Itera sobre cada ítem en la carpeta dataset
    
    breed_path = os.path.join(dataset_path, breed)
    # Construye la ruta completa: dataset/breed_name
    # os.path.join() asegura que el separador sea el correcto (/ó\)

    if os.path.isdir(breed_path):
        # Verifica que sea un directorio (no un archivo)
        
        print(f"✓ Verificando carpeta: {breed}")
        # Encabezado de la raza que está analizando

        for file in os.listdir(breed_path):
            # Itera sobre cada archivo en el directorio de la raza
            
            file_path = os.path.join(breed_path, file)
            # Ruta completa del archivo: dataset/breed/filename

            try:
                # INTENTA abrir y verificar la imagen
                
                img = Image.open(file_path)
                # Image.open(): Intenta abrir el archivo como imagen
                # Si falla aquí, el archivo está corrupto o no es imagen
                
                img.verify()
                # .verify(): Verifica que la imagen sea válida
                # Valida su estructura interna
                # Si falla, el archivo está dañado
                
                total_images += 1
                # Incrementa contador de imágenes válidas

            except Exception as e:
                # CAPTURA cualquier error durante la verificación
                # Esto incluye:
                # - Archivos corruptos
                # - Archivos que no son imágenes
                # - Imágenes con formato inválido
                # - Errores de lectura
                
                print(f"  ⚠️  Imagen dañada: {file_path}")
                # Muestra la ruta del archivo problemático
                
                bad_images.append(file_path)
                # Agrega a la lista de imágenes problemáticas

# ═════════════════════════════════════════════════════════════════════════════
# 📊 REPORTE FINAL
# ═════════════════════════════════════════════════════════════════════════════

print(f"""
╔═══════════════════════════════════════════════════════════════════════╗
║                      📊 REPORTE DE VERIFICACIÓN                      ║
╠═══════════════════════════════════════════════════════════════════════╣
║
║  Total de imágenes válidas: {total_images}
║  Total de imágenes dañadas: {len(bad_images)}
║
""")

if len(bad_images) == 0:
    print("""║  ✅ ¡Excelente! No hay imágenes dañadas.
║     El dataset está en perfecto estado para entrenamiento.
║
╚═══════════════════════════════════════════════════════════════════════╝
""")
else:
    print(f"""║  ⚠️  Se encontraron {len(bad_images)} imágenes problemáticas.
║     Estas se listarán abajo.
║
║  💡 RECOMENDACIÓN:
║     - Elimina estos archivos
║     - O descarga el dataset nuevamente
║
╚═══════════════════════════════════════════════════════════════════════╝
""")
    
    # Mostrar detalle de imágenes dañadas
    print("\n📝 Detalle de imágenes dañadas:\n")
    for i, bad_img in enumerate(bad_images, 1):
        print(f"   {i}. {bad_img}")
    
    # Sugerencia para eliminar
    print(f"\n💻 Para eliminar estas imágenes, puedes ejecutar:")
    print("\n   # En Python:")
    print("   import os")
    for bad_img in bad_images:
        print(f'   os.remove("{bad_img}")')

print(f"""
╔═══════════════════════════════════════════════════════════════════════╗
║                    📈 ESTADÍSTICAS FINALES                          ║
╠═══════════════════════════════════════════════════════════════════════╣
║
║  Imágenes totales procesadas: {total_images + len(bad_images)}
║  Imágenes válidas: {total_images}
║  Imágenes dañadas: {len(bad_images)}
║  Tasa de integridad: {(total_images / (total_images + len(bad_images)) * 100):.2f}% ✓
║
║  Estado: {'✅ LISTO PARA ENTRENAR' if len(bad_images) == 0 else '⚠️  NECESITA LIMPIEZA'}
║
╚═══════════════════════════════════════════════════════════════════════╝
""")