import os
import sys
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import ProjectConfig
from src.preprocessing import MedicalImagePreprocessor
from src.augmentation import MedicalImageAugmentor
from src.dataset import load_original_split_samples

def generate_visualizations():
    workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = ProjectConfig()
    dataset_root = cfg.get('dataset.root')
    
    samples = load_original_split_samples(dataset_root, 'train')
    
    # Pick 2 normal and 2 cancer samples
    normal_samples = [s['path'] for s in samples if s['class'] == 'normal'][:2]
    cancer_samples = [s['path'] for s in samples if s['class'] == 'cancer'][:2]
    
    selected_paths = [('NORMAL', p) for p in normal_samples] + [('CANCER', p) for p in cancer_samples]
    
    preprocessor = MedicalImagePreprocessor(target_size=(224, 224), enable_clahe=True)
    augmentor = MedicalImageAugmentor()

    fig, axes = plt.subplots(4, 3, figsize=(10, 12))
    plt.suptitle("Phase 2 Data Pipeline Visualizations: Original vs Preprocessed vs Augmented", fontsize=13)
    
    row_titles = ["Original (640x640)", "Preprocessed (CLAHE 224x224)", "Augmented (Elastic/Rot/Flip/Zoom)"]
    
    for row_idx, (label_str, fpath) in enumerate(selected_paths):
        img_orig = Image.open(fpath).convert('RGB')
        
        # Preprocessed (Tensor -> PIL for visualization)
        tensor_prep = preprocessor.process(img_orig)
        # Denormalize for viewing
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        tensor_denorm = tensor_prep.clone()
        for t, m, s in zip(tensor_denorm, mean, std):
            t.mul_(s).add_(m)
        tensor_denorm.clamp_(0, 1)
        img_prep = Image.fromarray((tensor_denorm.permute(1, 2, 0).numpy() * 255).astype('uint8'))

        # Augmented
        img_aug_pil = augmentor.augment(img_orig.resize((224, 224)))
        
        # Render Original
        axes[row_idx, 0].imshow(img_orig)
        axes[row_idx, 0].set_title(f"{label_str} - Original", fontsize=10)
        axes[row_idx, 0].axis('off')
        
        # Render Preprocessed
        axes[row_idx, 1].imshow(img_prep)
        axes[row_idx, 1].set_title(f"{label_str} - Preprocessed (CLAHE)", fontsize=10)
        axes[row_idx, 1].axis('off')

        # Render Augmented
        axes[row_idx, 2].imshow(img_aug_pil)
        axes[row_idx, 2].set_title(f"{label_str} - Augmented", fontsize=10)
        axes[row_idx, 2].axis('off')

    plt.tight_layout()
    fig_out = os.path.join(workspace_root, 'figures', 'augmentation_samples.png')
    os.makedirs(os.path.dirname(fig_out), exist_ok=True)
    plt.savefig(fig_out, dpi=300)
    plt.close()
    print(f"Augmentation & pipeline visualization figure saved to: {fig_out}")

if __name__ == '__main__':
    generate_visualizations()
