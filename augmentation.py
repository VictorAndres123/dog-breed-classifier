import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

class AugmentationPipeline:
    """Pipeline profesional de augmentación de imágenes"""
    
    @staticmethod
    def get_train_transforms(image_size=224):
        """
        Transformaciones agresivas para entrenamiento.
        
        Args:
            image_size: Tamaño de la imagen de entrada
            
        Returns:
            Composición de transformaciones de Albumentations
        """
        return A.Compose([
            # Redimensionamiento
            A.Resize(image_size, image_size, interpolation=cv2.INTER_CUBIC),
            
            # Rotaciones y perspectiva
            A.Rotate(limit=30, p=0.7, border_mode=cv2.BORDER_REFLECT_101),
            A.Perspective(scale=(0.05, 0.1), p=0.5),
            
            # Traslaciones
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, 
                             border_mode=cv2.BORDER_REFLECT_101, p=0.7),
            
            # Flips
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            
            # Distorsiones
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
            A.GridDistortion(p=0.3),
            
            # Cambios de iluminación
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
            A.RandomGamma(gamma_limit=(70, 130), p=0.5),
            A.CLAHE(p=0.3),
            
            # Cambios de color
            A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=15, p=0.7),
            
            # Ruido
            A.GaussNoise(p=0.2),
            A.MultiplicativeNoise(p=0.2),
            
            # Blur
            A.Blur(blur_limit=3, p=0.2),
            A.MotionBlur(p=0.1),
            
            # Normalización
            A.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225]),
            
            ToTensorV2()
        ])
    
    @staticmethod
    def get_val_transforms(image_size=224):
        """
        Transformaciones mínimas para validación.
        
        Args:
            image_size: Tamaño de la imagen de entrada
            
        Returns:
            Composición de transformaciones de Albumentations
        """
        return A.Compose([
            A.Resize(image_size, image_size, interpolation=cv2.INTER_CUBIC),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    @staticmethod
    def get_test_transforms(image_size=224):
        """
        Transformaciones idénticas a validación para prueba.
        
        Args:
            image_size: Tamaño de la imagen de entrada
            
        Returns:
            Composición de transformaciones de Albumentations
        """
        return AugmentationPipeline.get_val_transforms(image_size)
