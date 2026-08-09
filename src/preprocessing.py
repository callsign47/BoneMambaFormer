import numpy as np
from PIL import Image, ImageOps, ImageEnhance
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

class MedicalImagePreprocessor:
    """
    Deterministic image preprocessor for bone cancer X-ray / Radiographs.
    Includes controlled contrast enhancement (Adaptive/Auto Contrast) and standard RGB normalization.
    """
    def __init__(self, target_size=(224, 224), mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), enable_clahe=True):
        self.target_size = tuple(target_size)
        self.mean = mean
        self.std = std
        self.enable_clahe = enable_clahe
        
        # Standard normalization transform
        self.normalize_tf = T.Normalize(mean=self.mean, std=self.std)

    def apply_contrast_enhancement(self, img_pil):
        """
        Applies controlled contrast enhancement safely using PIL ImageOps autocontrast.
        Preserves diagnostic morphology without introducing noise artifacts.
        """
        enhanced = ImageOps.autocontrast(img_pil, cutoff=1)
        enhancer = ImageEnhance.Contrast(enhanced)
        return enhancer.enhance(1.1)

    def process(self, img_pil):
        """
        Full deterministic preprocessing pipeline:
        1. Resize
        2. Controlled Contrast Enhancement if enabled
        3. Convert to Tensor [0, 1]
        4. Normalize
        """
        # 1. Resize
        img_resized = img_pil.resize(self.target_size, Image.BILINEAR)
        
        # 2. Contrast enhancement
        if self.enable_clahe:
            img_processed = self.apply_contrast_enhancement(img_resized)
        else:
            img_processed = img_resized
            
        # 3. Convert to Tensor
        tensor = TF.to_tensor(img_processed) # [C, H, W] in range [0, 1]
        
        # 4. Normalize
        tensor_norm = self.normalize_tf(tensor)
        
        return tensor_norm

if __name__ == '__main__':
    test_img = Image.new('RGB', (640, 640), color=(128, 128, 128))
    prep = MedicalImagePreprocessor(target_size=(224, 224), enable_clahe=True)
    out_tensor = prep.process(test_img)
    print("Preprocessed Tensor Shape:", out_tensor.shape)
    print("Tensor Mean:", out_tensor.mean().item(), "Std:", out_tensor.std().item())
