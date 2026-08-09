import random
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
from scipy.ndimage import gaussian_filter, map_coordinates

class ElasticTransform:
    """
    Elastic deformation of images using scipy.ndimage as described in [Simard2003].
    """
    def __init__(self, alpha=10.0, sigma=3.0, prob=0.3):
        self.alpha = alpha
        self.sigma = sigma
        self.prob = prob

    def __call__(self, img_pil):
        if random.random() > self.prob:
            return img_pil
            
        img_np = np.array(img_pil)
        shape = img_np.shape
        
        dx = gaussian_filter((np.random.rand(*shape[:2]) * 2 - 1), self.sigma, mode="constant", cval=0) * self.alpha
        dy = gaussian_filter((np.random.rand(*shape[:2]) * 2 - 1), self.sigma, mode="constant", cval=0) * self.alpha

        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

        if len(shape) == 3:
            distorted = np.zeros_like(img_np)
            for i in range(shape[2]):
                distorted[:, :, i] = map_coordinates(img_np[:, :, i], indices, order=1, mode='reflect').reshape(shape[:2])
            return Image.fromarray(distorted.astype(np.uint8))
        else:
            distorted = map_coordinates(img_np, indices, order=1, mode='reflect').reshape(shape[:2])
            return Image.fromarray(distorted.astype(np.uint8))

class MedicalImageAugmentor:
    """
    Training-only augmentation pipeline for Bone Cancer Radiographs.
    """
    def __init__(self, rotation_degrees=15, flip_prob=0.5, zoom_scale=(0.9, 1.1), brightness=0.1, contrast=0.1, elastic_alpha=10.0, elastic_sigma=3.0, elastic_prob=0.3):
        self.rotation_degrees = rotation_degrees
        self.flip_prob = flip_prob
        self.zoom_scale = zoom_scale
        self.brightness = brightness
        self.contrast = contrast
        self.elastic_transform = ElasticTransform(alpha=elastic_alpha, sigma=elastic_sigma, prob=elastic_prob)

    def augment(self, img_pil):
        """
        Applies controlled medical image augmentations on PIL Image:
        1. Elastic Transform
        2. Random Rotation
        3. Random Flips (Horizontal / Vertical)
        4. Random Zoom / Rescale
        5. Intensity Variation (Brightness / Contrast Jitter)
        """
        # 1. Elastic Transform
        img_aug = self.elastic_transform(img_pil)

        # 2. Rotation
        angle = random.uniform(-self.rotation_degrees, self.rotation_degrees)
        img_aug = TF.rotate(img_aug, angle)

        # 3. Flips (Horizontal flip only, vertical flip disabled)
        if random.random() < self.flip_prob:
            img_aug = TF.hflip(img_aug)

        # 4. Zoom / Rescale
        scale = random.uniform(self.zoom_scale[0], self.zoom_scale[1])
        w, h = img_aug.size
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        img_scaled = img_aug.resize((new_w, new_h), Image.BILINEAR)
        
        # Center crop or pad to original size
        if scale > 1.0:
            left = (new_w - w) // 2
            top = (new_h - h) // 2
            img_aug = img_scaled.crop((left, top, left + w, top + h))
        else:
            pad_left = (w - new_w) // 2
            pad_top = (h - new_h) // 2
            img_padded = Image.new(img_scaled.mode, (w, h), color=(0, 0, 0))
            img_padded.paste(img_scaled, (pad_left, pad_top))
            img_aug = img_padded

        # 5. Brightness & Contrast
        if self.brightness > 0 or self.contrast > 0:
            b_factor = random.uniform(1.0 - self.brightness, 1.0 + self.brightness)
            c_factor = random.uniform(1.0 - self.contrast, 1.0 + self.contrast)
            img_aug = TF.adjust_brightness(img_aug, b_factor)
            img_aug = TF.adjust_contrast(img_aug, c_factor)

        return img_aug

if __name__ == '__main__':
    test_img = Image.new('RGB', (224, 224), color=(100, 150, 200))
    aug = MedicalImageAugmentor()
    res = aug.augment(test_img)
    print("Augmentation test successful. Output size:", res.size)
