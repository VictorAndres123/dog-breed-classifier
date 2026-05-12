import os
import torch

class Config:
    """Configuración centralizada del proyecto"""

    # =========================
    # RUTAS DEL PROYECTO
    # =========================
    PROJECT_ROOT = r"c:\Users\ESTEBAN\Desktop\dog-breed-classifier"
    DATASET_PATH = os.path.join(PROJECT_ROOT, "dataset")
    DATA_PATH = os.path.join(PROJECT_ROOT, "data")
    MODELS_PATH = os.path.join(PROJECT_ROOT, "models")
    LOGS_PATH = os.path.join(PROJECT_ROOT, "logs")

    # Crear directorios si no existen
    os.makedirs(MODELS_PATH, exist_ok=True)
    os.makedirs(LOGS_PATH, exist_ok=True)

    # =========================
    # MODELO
    # =========================
    MODEL_NAME = "efficientnet_b0"
    NUM_CLASSES = 55
    IMAGE_SIZE = 224
    PRETRAINED = True

    # =========================
    # ENTRENAMIENTO (RTX 3050 OPTIMIZADO)
    # =========================
    BATCH_SIZE = 16
    VAL_BATCH_SIZE = 32
    TEST_BATCH_SIZE = 32
    GRADIENT_ACCUMULATION_STEPS = 2
    EPOCHS = 100
    INITIAL_LR = 1e-3
    WEIGHT_DECAY = 1e-5
    MOMENTUM = 0.9

    # =========================
    # SCHEDULER
    # =========================
    LR_SCHEDULER = "cosine"
    SCHEDULER_T_MAX = 100
    SCHEDULER_GAMMA = 0.1
    SCHEDULER_STEP_SIZE = 30

    # =========================
    # EARLY STOPPING
    # =========================
    EARLY_STOPPING = True
    EARLY_STOPPING_PATIENCE = 20
    EARLY_STOPPING_MIN_DELTA = 1e-4

    # =========================
    # GPU / DEVICE
    # =========================
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 2 if torch.cuda.is_available() else 0
    PIN_MEMORY = torch.cuda.is_available()

    # =========================
    # OPTIMIZACIÓN GPU
    # =========================
    USE_AMP = True  # Mixed Precision (OBLIGATORIO RTX 3050)

    # =========================
    # OPTIMIZER
    # =========================
    OPTIMIZER = "adamw"

    # =========================
    # CHECKPOINT (TU MODELO REAL)
    # =========================
    CHECKPOINT_PATH = os.path.join(MODELS_PATH, "best_efficientnet_b0.pth")

    # =========================
    # LOGGING
    # =========================
    LOG_INTERVAL = 10
    EVAL_INTERVAL = 1

    # =========================
    # REPRODUCIBILIDAD
    # =========================
    SEED = 42

    # =========================
    # DATASET INFO
    # =========================
    NUM_BREEDS = 55

    @classmethod
    def print_config(cls):
        """Imprime la configuración"""
        print("\n" + "=" * 60)
        print("CONFIGURACIÓN DEL PROYECTO")
        print("=" * 60)
        print(f"Device: {cls.DEVICE}")
        print(f"CUDA disponible: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"GPU Name: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

        print(f"\nModelo: {cls.MODEL_NAME}")
        print(f"Clases: {cls.NUM_CLASSES}")
        print(f"Imagen: {cls.IMAGE_SIZE}x{cls.IMAGE_SIZE}")

        print(f"\nBatch Train: {cls.BATCH_SIZE}")
        print(f"Batch Val: {cls.VAL_BATCH_SIZE}")
        print(f"Epochs: {cls.EPOCHS}")
        print(f"LR: {cls.INITIAL_LR}")
        print(f"Optimizer: {cls.OPTIMIZER}")

        print(f"\nAMP (GPU): {cls.USE_AMP}")
        print(f"Early Stopping: {cls.EARLY_STOPPING}")

        print(f"\nCheckpoint:")
        print(cls.CHECKPOINT_PATH)

        print("=" * 60 + "\n")