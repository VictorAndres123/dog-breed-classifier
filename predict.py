import torch
import torch.nn.functional as F
from pathlib import Path
import numpy as np
from PIL import Image
from config import Config
from model import create_model
from augmentation import AugmentationPipeline

class Predictor:
    """Clase para hacer predicciones con el modelo entrenado"""
    
    def __init__(self, model_path=Config.CHECKPOINT_PATH):
        self.device = Config.DEVICE
        self.model = create_model(num_classes=Config.NUM_CLASSES, pretrained=False)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        # Obtener nombres de clases
        data_path = Path(Config.DATA_PATH) / "train"
        self.classes = sorted([d.name for d in data_path.iterdir() if d.is_dir()])
        
        print(f"✓ Modelo cargado desde: {model_path}")
        print(f"✓ Device: {self.device}")
        print(f"✓ Clases disponibles: {len(self.classes)}\n")
    
    def predict_single(self, image_path, top_k=5):
        """
        Realiza predicción en una imagen individual.
        
        Args:
            image_path: Ruta a la imagen
            top_k: Top K predicciones a retornar
            
        Returns:
            Lista de tuplas (clase, probabilidad)
        """
        image = Image.open(image_path).convert('RGB')
        transform = AugmentationPipeline.get_val_transforms(Config.IMAGE_SIZE)
        image_tensor = transform(image=np.array(image))['image'].unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1).cpu()
        
        # Obtener top K
        top_probs, top_indices = torch.topk(probabilities, min(top_k, len(self.classes)))
        
        results = []
        for prob, idx in zip(top_probs[0], top_indices[0]):
            results.append((self.classes[idx], prob.item()))
        
        return results
    
    def predict_directory(self, directory_path, output_file=None):
        """
        Realiza predicciones en todas las imágenes de un directorio.
        
        Args:
            directory_path: Ruta al directorio con imágenes
            output_file: Archivo donde guardar resultados (opcional)
        """
        image_files = list(Path(directory_path).glob("*.*"))
        image_files = [f for f in image_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
        
        results = {}
        
        print(f"Realizando predicciones en {len(image_files)} imágenes...\n")
        
        for image_path in image_files:
            predictions = self.predict_single(image_path, top_k=3)
            results[image_path.name] = predictions
            
            print(f"{image_path.name}:")
            for rank, (class_name, prob) in enumerate(predictions, 1):
                print(f"  {rank}. {class_name}: {prob*100:.2f}%")
            print()
        
        if output_file:
            import json
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=4)
            print(f"✓ Resultados guardados en: {output_file}")
        
        return results
    
    def predict_batch(self, image_paths, top_k=5):
        """
        Realiza predicciones en un lote de imágenes.
        
        Args:
            image_paths: Lista de rutas a imágenes
            top_k: Top K predicciones
            
        Returns:
            Lista de resultados
        """
        results = []
        for image_path in image_paths:
            predictions = self.predict_single(image_path, top_k=top_k)
            results.append({
                'image': image_path,
                'predictions': predictions
            })
        return results

def main():
    """Función principal para pruebas"""
    
    print("\n" + "="*60)
    print("DOG BREED CLASSIFIER - PREDICTION")
    print("="*60)
    
    # Crear predictor
    predictor = Predictor()
    
    # Ejemplo: predicción en una imagen de test
    test_data_path = Path(Config.DATA_PATH) / "test"
    
    if test_data_path.exists():
        # Obtener primera imagen de cada clase
        sample_images = []
        for class_dir in sorted(test_data_path.iterdir()):
            if class_dir.is_dir():
                images = list(class_dir.glob("*"))
                if images:
                    sample_images.append(images[0])
        
        if sample_images:
            print(f"\nRealizando predicciones en {len(sample_images)} imágenes de prueba...\n")
            
            for image_path in sample_images[:10]:  # Mostrar primeras 10
                predictions = predictor.predict_single(image_path, top_k=3)
                print(f"Imagen: {image_path.parent.name}/{image_path.name}")
                for rank, (class_name, prob) in enumerate(predictions, 1):
                    print(f"  {rank}. {class_name}: {prob*100:.2f}%")
                print()

if __name__ == "__main__":
    main()
