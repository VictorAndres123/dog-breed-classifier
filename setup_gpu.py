#!/usr/bin/env python3
"""
Script para instalar PyTorch con soporte CUDA para RTX 3050
Ejecutar este script ANTES de entrenar
"""

import subprocess
import sys
import os

def install_pytorch_cuda():
    """Instala PyTorch con soporte CUDA"""
    
    print("\n" + "="*70)
    print("PYTORCH CUDA INSTALLATION FOR RTX 3050")
    print("="*70)
    
    print("""
PyTorch necesita ser instalado CON soporte CUDA para usar la GPU.

Selecciona tu versión de CUDA:
1. CUDA 12.1 (Recomendado para RTX 3050)
2. CUDA 11.8
3. CPU only (Sin GPU - MÁS LENTO)

¿Cuál quieres instalar? (1-3):
""")
    
    choice = input().strip()
    
    if choice == "1":
        print("\n🔧 Instalando PyTorch 2.0.1 con CUDA 12.1...")
        cmd = [
            sys.executable, "-m", "pip", "install", "-q",
            "torch==2.0.1", "torchvision==0.15.2", "torchaudio==2.0.2",
            "--index-url", "https://download.pytorch.org/whl/cu121"
        ]
    elif choice == "2":
        print("\n🔧 Instalando PyTorch 2.0.1 con CUDA 11.8...")
        cmd = [
            sys.executable, "-m", "pip", "install", "-q",
            "torch==2.0.1", "torchvision==0.15.2", "torchaudio==2.0.2",
            "--index-url", "https://download.pytorch.org/whl/cu118"
        ]
    elif choice == "3":
        print("\n🔧 Instalando PyTorch 2.0.1 CPU only...")
        cmd = [
            sys.executable, "-m", "pip", "install", "-q",
            "torch==2.0.1", "torchvision==0.15.2", "torchaudio==2.0.2"
        ]
    else:
        print("❌ Opción inválida")
        return False
    
    try:
        subprocess.run(cmd, check=True)
        print("✓ PyTorch instalado exitosamente")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error durante la instalación: {e}")
        return False

def install_dependencies():
    """Instala dependencias restantes"""
    
    print("\n🔧 Instalando dependencias adicionales...")
    
    packages = [
        "numpy>=1.24.3",
        "Pillow>=10.0.0",
        "opencv-python>=4.8.0",
        "albumentations>=1.3.1",
        "scikit-learn>=1.3.0",
        "tqdm>=4.66.1",
        "matplotlib>=3.7.2",
        "psutil>=5.9.0"
    ]
    
    cmd = [sys.executable, "-m", "pip", "install", "-q"] + packages
    
    try:
        subprocess.run(cmd, check=True)
        print("✓ Todas las dependencias instaladas")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        return False

def verify_installation():
    """Verifica que todo esté correctamente instalado"""
    
    print("\n" + "="*70)
    print("VERIFYING INSTALLATION")
    print("="*70)
    
    try:
        import torch
        print(f"\n✓ PyTorch {torch.__version__} instalado")
        
        cuda_available = torch.cuda.is_available()
        print(f"✓ CUDA disponible: {cuda_available}")
        
        if cuda_available:
            print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
            print(f"✓ CUDA Version: {torch.version.cuda}")
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✓ GPU Memory: {total_memory:.2f} GB")
        
        import torchvision
        print(f"✓ torchvision {torchvision.__version__} instalado")
        
        import albumentations
        print(f"✓ albumentations {albumentations.__version__} instalado")
        
        import cv2
        print(f"✓ opencv-python {cv2.__version__} instalado")
        
        import numpy
        print(f"✓ numpy {numpy.__version__} instalado")
        
        print("\n✓ ¡Todas las librerías instaladas correctamente!")
        
        if not cuda_available:
            print("\n⚠️  ADVERTENCIA: CUDA no está disponible")
            print("   El entrenamiento será MUCHO más lento (solo CPU)")
            print("   Reinstala PyTorch con soporte CUDA si es posible")
        
        return cuda_available
    
    except Exception as e:
        print(f"❌ Error durante verificación: {e}")
        return False

def main():
    print("\n" + "="*70)
    print("SETUP WIZARD - PyTorch + Dependencies for RTX 3050")
    print("="*70)
    
    print("""
Este wizard te ayudará a:
1. Instalar PyTorch con soporte CUDA
2. Instalar todas las dependencias necesarias
3. Verificar que todo funcione correctamente

¿Deseas continuar? (s/n):
""")
    
    if input().strip().lower() != 's':
        print("Cancelado")
        return
    
    # Instalar PyTorch
    if not install_pytorch_cuda():
        print("❌ Error al instalar PyTorch")
        return
    
    # Instalar dependencias
    if not install_dependencies():
        print("❌ Error al instalar dependencias")
        return
    
    # Verificar
    cuda_ok = verify_installation()
    
    print("\n" + "="*70)
    if cuda_ok:
        print("✓ ¡Sistema listo para entrenar!")
        print("\nPróximos pasos:")
        print("  1. python split_dataset.py     # Dividir dataset")
        print("  2. python training.py          # Entrenar modelo")
    else:
        print("⚠️  Sistema parcialmente listo (solo CPU)")
        print("\nPróximos pasos:")
        print("  1. Reinstala PyTorch con soporte CUDA")
        print("  2. O continúa con CPU (será muy lento)")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
