#!/usr/bin/env python3
"""
Script de optimización para RTX 3050 (4GB VRAM)
Verifica memoria y proporciona recomendaciones
"""

import torch
import os
import psutil

def check_gpu_memory():
    """Verifica memoria disponible en GPU"""
    print("\n" + "="*70)
    print("GPU MEMORY ANALYSIS - RTX 3050")
    print("="*70)
    
    if not torch.cuda.is_available():
        print("❌ GPU no disponible")
        return False
    
    # Información de GPU
    print(f"\n✓ GPU: {torch.cuda.get_device_name(0)}")
    
    # Memoria total
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"✓ VRAM Total: {total_memory:.2f} GB")
    
    # Memoria disponible
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    free = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved()) / 1e9
    
    print(f"\nMemoria Actual:")
    print(f"  - Asignada: {allocated:.2f} GB")
    print(f"  - Reservada: {reserved:.2f} GB")
    print(f"  - Disponible: {free:.2f} GB")
    
    # Capacidad de compute
    props = torch.cuda.get_device_properties(0)
    print(f"\nCapacidades:")
    print(f"  - Compute Capability: {props.major}.{props.minor}")
    print(f"  - SM Count: {props.multi_processor_count}")
    
    return True

def check_system_memory():
    """Verifica RAM del sistema"""
    print("\n" + "="*70)
    print("SYSTEM MEMORY ANALYSIS")
    print("="*70)
    
    # Memoria total
    virtual_memory = psutil.virtual_memory()
    total_ram = virtual_memory.total / 1e9
    available_ram = virtual_memory.available / 1e9
    used_ram = virtual_memory.used / 1e9
    percent = virtual_memory.percent
    
    print(f"\n✓ RAM Total: {total_ram:.2f} GB")
    print(f"✓ RAM Disponible: {available_ram:.2f} GB")
    print(f"✓ RAM Usada: {used_ram:.2f} GB ({percent}%)")
    
    if available_ram < 4:
        print("\n⚠️  ADVERTENCIA: Menos de 4GB de RAM disponible")
        print("   Puede haber problemas de rendimiento")
    
    return available_ram >= 4

def check_cuda_memory():
    """Verifica cuánta memoria CUDA está siendo usada"""
    print("\n" + "="*70)
    print("CUDA MEMORY STATS")
    print("="*70)
    
    try:
        # Reset
        if hasattr(torch.cuda, 'reset_peak_memory_stats'):
            torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
        print("\n✓ Cache limpiado")
        print(f"✓ Memoria reservada antes: {torch.cuda.memory_reserved() / 1e6:.2f} MB")
    except Exception as e:
        print(f"\n⚠ Error al limpiar cache: {e}")

def estimate_batch_size():
    """Estima el tamaño de batch máximo recomendado"""
    print("\n" + "="*70)
    print("BATCH SIZE ESTIMATION")
    print("="*70)
    
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    # EfficientNetB0 típicamente ocupa ~2GB por batch de 32
    # Regla empírica: 4GB / 32 ≈ 125MB por imagen
    memory_per_batch_32 = 2.0  # GB
    
    if total_memory == 4:
        recommended_batch = 16
        print(f"\n✓ VRAM: {total_memory:.1f} GB")
        print(f"✓ Batch Size Recomendado: {recommended_batch}")
        print(f"✓ Con Gradient Accumulation (2x): Simula batch {recommended_batch * 2}")
        print(f"\nExplicación:")
        print(f"  - Batch 16: ~1.5-2GB VRAM")
        print(f"  - Gradient Accumulation 2x: Batch efectivo de 32")
        print(f"  - Mejor para estabilidad del training")
    
    return recommended_batch

def recommendations():
    """Imprime recomendaciones específicas"""
    print("\n" + "="*70)
    print("RECOMENDACIONES PARA RTX 3050 (4GB VRAM)")
    print("="*70)
    
    print("""
✓ CONFIGURACIÓN ACTUAL EN config.py:
  - BATCH_SIZE: 16
  - GRADIENT_ACCUMULATION_STEPS: 2
  - NUM_WORKERS: 2
  - USE_AMP: True (Mixed Precision - CRÍTICO)
  - EARLY_STOPPING_PATIENCE: 20

✓ TIPS PARA OPTIMIZAR:

1. ANTES DE ENTRENAR:
   - Cierra otros programas pesados
   - Reinicia el notebook si fue usado antes
   - Ejecuta: torch.cuda.empty_cache()

2. DURANTE EL ENTRENAMIENTO:
   - Monitorea memoria con: nvidia-smi
   - Si hay out-of-memory: reduce BATCH_SIZE a 8
   - Si es muy lento: aumenta BATCH_SIZE a 20 (si tienes suerte)

3. SI HAY OUT-OF-MEMORY:
   Reduce en config.py:
   BATCH_SIZE = 8
   GRADIENT_ACCUMULATION_STEPS = 4
   NUM_WORKERS = 1

4. SI ES MUY LENTO:
   Aumenta en config.py:
   BATCH_SIZE = 20
   GRADIENT_ACCUMULATION_STEPS = 2
   NUM_WORKERS = 2

5. CONGELAR BACKBONE (Ahorra memoria, entrena más rápido):
   En training.py antes de trainer.train():
   trainer.model.freeze_backbone(True)
   (Desventaja: menos precisión, pero entrena en ~15 minutos)

6. MONITOREO DE MEMORIA:
   - GPU Memory: nvidia-smi (actualización cada 2s)
   - RAM: htop (en Linux) o Task Manager (Windows)

✓ TIEMPO ESTIMADO:
  - Con config actual: 60-90 minutos por epoch
  - Con backbone congelado: 15-20 minutos por epoch
  - Total para 100 epochs: 100-150 horas (o 25-30 horas congelado)

✓ RENDIMIENTO ESPERADO:
  - Sin congelado: 88-92% validation accuracy
  - Con backbone congelado: 80-85% validation accuracy
    """)

def main():
    print("\n" + "="*70)
    print("RTX 3050 (4GB VRAM) - OPTIMIZATION GUIDE")
    print("="*70)
    
    # Verificaciones
    gpu_ok = check_gpu_memory()
    ram_ok = check_system_memory()
    check_cuda_memory()
    batch_size = estimate_batch_size()
    
    if gpu_ok and ram_ok:
        recommendations()
        print("\n✓ Sistema listo para entrenar")
        print("✓ Ejecuta: python split_dataset.py")
        print("✓ Luego: python training.py\n")
    else:
        print("\n❌ Hay problemas con tu sistema")
        if not gpu_ok:
            print("   - GPU no disponible o no tiene suficiente memoria")
        if not ram_ok:
            print("   - No hay suficiente RAM disponible")

if __name__ == "__main__":
    main()
