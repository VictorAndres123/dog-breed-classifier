import torch
import sys
from pathlib import Path

def check_setup():
    """Verifica que todo esté configurado correctamente"""
    
    print("\n" + "="*70)
    print("DOG BREED CLASSIFIER - SETUP VERIFICATION")
    print("="*70)
    
    checks_passed = 0
    checks_total = 0
    
    # 1. Python version
    print("\n[1] Python Version")
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(f"    Version: {py_version}")
    if sys.version_info >= (3, 8):
        print("    ✓ PASSED")
        checks_passed += 1
    else:
        print("    ✗ FAILED: Python 3.8+ required")
    checks_total += 1
    
    # 2. PyTorch
    print("\n[2] PyTorch")
    try:
        print(f"    Version: {torch.__version__}")
        print("    ✓ PASSED")
        checks_passed += 1
    except:
        print("    ✗ FAILED: PyTorch not installed")
    checks_total += 1
    
    # 3. GPU/CUDA
    print("\n[3] GPU/CUDA Support")
    cuda_available = torch.cuda.is_available()
    print(f"    CUDA Available: {cuda_available}")
    if cuda_available:
        print(f"    GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"    GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"    CUDA Version: {torch.version.cuda}")
        print("    ✓ PASSED")
        checks_passed += 1
    else:
        print("    ⚠ WARNING: GPU not available (CPU will be used)")
        checks_passed += 1  # No es crítico
    checks_total += 1
    
    # 4. Required packages
    print("\n[4] Required Packages")
    packages = [
        'torchvision',
        'numpy',
        'PIL',
        'cv2',
        'albumentations',
        'sklearn',
        'tqdm'
    ]
    
    all_packages_ok = True
    for package in packages:
        try:
            if package == 'PIL':
                import PIL
                print(f"    ✓ {package}: {PIL.__version__}")
            elif package == 'cv2':
                import cv2
                print(f"    ✓ {package}: {cv2.__version__}")
            else:
                mod = __import__(package)
                version = getattr(mod, '__version__', 'installed')
                print(f"    ✓ {package}: {version}")
        except ImportError:
            print(f"    ✗ {package}: NOT INSTALLED")
            all_packages_ok = False
    
    if all_packages_ok:
        checks_passed += 1
    checks_total += 1
    
    # 5. Project structure
    print("\n[5] Project Structure")
    project_root = Path(__file__).parent
    required_files = [
        'config.py',
        'model.py',
        'augmentation.py',
        'training.py',
        'split_dataset.py',
        'predict.py'
    ]
    
    all_files_ok = True
    for file in required_files:
        file_path = project_root / file
        if file_path.exists():
            print(f"    ✓ {file}")
        else:
            print(f"    ✗ {file}: NOT FOUND")
            all_files_ok = False
    
    if all_files_ok:
        print("    ✓ PASSED")
        checks_passed += 1
    checks_total += 1
    
    # 6. Directories
    print("\n[6] Required Directories")
    from config import Config
    
    directories = [
        ('Dataset', Config.DATASET_PATH),
        ('Models', Config.MODELS_PATH),
        ('Logs', Config.LOGS_PATH)
    ]
    
    all_dirs_ok = True
    for name, path in directories:
        if Path(path).exists():
            print(f"    ✓ {name}: {path}")
        else:
            print(f"    ⚠ {name}: {path} (will be created)")
    
    print("    ✓ PASSED")
    checks_passed += 1
    checks_total += 1
    
    # 7. Config validation
    print("\n[7] Configuration")
    try:
        from config import Config
        
        required_attrs = [
            'MODEL_NAME',
            'NUM_CLASSES',
            'BATCH_SIZE',
            'EPOCHS',
            'INITIAL_LR',
            'DEVICE'
        ]
        
        all_attrs_ok = True
        for attr in required_attrs:
            if hasattr(Config, attr):
                value = getattr(Config, attr)
                print(f"    ✓ {attr}: {value}")
            else:
                print(f"    ✗ {attr}: NOT FOUND")
                all_attrs_ok = False
        
        if all_attrs_ok:
            checks_passed += 1
        checks_total += 1
    except Exception as e:
        print(f"    ✗ FAILED: {e}")
        checks_total += 1
    
    # Summary
    print("\n" + "="*70)
    print(f"RESULTS: {checks_passed}/{checks_total} checks passed")
    print("="*70)
    
    if checks_passed == checks_total:
        print("\n✓ System is ready! You can start training now.")
        print("\nQuick start:")
        print("  1. python split_dataset.py     # Divide train/val/test")
        print("  2. python training.py          # Start training")
        print("  3. python predict.py           # Make predictions\n")
        return True
    else:
        print(f"\n✗ {checks_total - checks_passed} issue(s) found. Please fix them before training.\n")
        return False

if __name__ == "__main__":
    success = check_setup()
    sys.exit(0 if success else 1)
