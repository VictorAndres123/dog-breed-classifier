#!/usr/bin/env python3
"""
Monitor de GPU durante el entrenamiento
Muestra en tiempo real: memoria, temperatura, utilización
"""

import torch
import threading
import time
import os

class GPUMonitor:
    def __init__(self, interval=2):
        self.interval = interval
        self.running = False
        self.thread = None
    
    def start(self):
        """Inicia monitoreo en background"""
        if not torch.cuda.is_available():
            print("GPU no disponible")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        print("✓ GPU Monitor iniciado")
    
    def stop(self):
        """Detiene monitoreo"""
        self.running = False
        if self.thread:
            self.thread.join()
        print("✓ GPU Monitor detenido")
    
    def _monitor_loop(self):
        """Loop de monitoreo"""
        while self.running:
            self._print_stats()
            time.sleep(self.interval)
    
    def _print_stats(self):
        """Imprime estadísticas"""
        try:
            # Limpiar pantalla (opcional)
            # os.system('clear' if os.name == 'posix' else 'cls')
            
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            max_allocated = torch.cuda.max_memory_allocated() / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            # Barra de progreso
            usage_percent = (allocated / total) * 100
            bar_length = 30
            filled = int(bar_length * usage_percent / 100)
            bar = '█' * filled + '░' * (bar_length - filled)
            
            # Mostrar
            print(f"\r[{bar}] {usage_percent:.1f}% | {allocated:.2f}GB / {total:.2f}GB | " +
                  f"Reserved: {reserved:.2f}GB | Max: {max_allocated:.2f}GB", end='')
            
        except Exception as e:
            print(f"Error en monitor: {e}")

def print_memory_summary():
    """Resumen de memoria"""
    print("\n" + "="*70)
    print("MEMORIA GPU SUMMARY")
    print("="*70)
    
    if not torch.cuda.is_available():
        print("GPU no disponible")
        return
    
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    max_allocated = torch.cuda.max_memory_allocated() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"\nMemoria Actual:")
    print(f"  Asignada: {allocated:.2f} GB")
    print(f"  Reservada: {reserved:.2f} GB")
    print(f"  Máxima: {max_allocated:.2f} GB")
    print(f"  Total: {total:.2f} GB")
    print(f"\nUtilización: {(allocated/total)*100:.1f}%")
    print("="*70)

if __name__ == "__main__":
    print("GPU Monitor v1.0")
    print("Presiona Ctrl+C para salir\n")
    
    monitor = GPUMonitor(interval=1)
    monitor.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        monitor.stop()
        print_memory_summary()
        print("\n✓ Monitor cerrado")
