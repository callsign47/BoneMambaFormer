import os
import sys
import json
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import MedicalImagePreprocessor
from src.augmentation import MedicalImageAugmentor
from src.config import ProjectConfig

class BoneCancerDataset(Dataset):
    """
    PyTorch Dataset for Bone Cancer Classification.
    Class Label Mapping:
      cancer -> 0
      normal -> 1
    """
    def __init__(self, samples, is_train=False, preprocessor=None, augmentor=None, class_mapping=None):
        """
        samples: list of dicts {'path': abs_path, 'class': label_str}
        """
        self.samples = samples
        self.is_train = is_train
        self.preprocessor = preprocessor or MedicalImagePreprocessor()
        self.augmentor = augmentor if is_train else None
        self.class_mapping = class_mapping or {'cancer': 0, 'normal': 1}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = sample['path']
        label_str = sample['class']
        label = self.class_mapping[label_str]

        # Load image
        img_pil = Image.open(img_path).convert('RGB')

        # Apply augmentation if in training mode
        if self.is_train and self.augmentor is not None:
            img_pil = self.augmentor.augment(img_pil)

        # Apply deterministic preprocessor (resize, CLAHE, tensor, normalization)
        img_tensor = self.preprocessor.process(img_pil)

        return img_tensor, torch.tensor(label, dtype=torch.long)

def load_original_split_samples(dataset_root, split_name):
    splits_folders = {
        'train': 'train_sorted-20251125T062947Z-1-001',
        'val': 'valid_sorted-20251125T062929Z-1-001',
        'test': 'test_sorted-20251125T062917Z-1-001'
    }
    folder_name = splits_folders[split_name]
    split_dir = os.path.join(dataset_root, folder_name)
    
    samples = []
    for c in ['cancer', 'normal']:
        cdir = os.path.join(split_dir, c)
        if os.path.exists(cdir):
            for fname in os.listdir(cdir):
                fpath = os.path.join(cdir, fname)
                if os.path.isfile(fpath):
                    samples.append({'path': fpath, 'class': c})
    return samples

def load_derived_clean_samples(clean_json_path, split_name):
    with open(clean_json_path, 'r') as f:
        clean_split_data = json.load(f)
    samples = []
    for item in clean_split_data[split_name]:
        samples.append({'path': item['abs_path'], 'class': item['class']})
    return samples

def get_dataloaders(config=None):
    """
    DataLoader factory returning (train_loader, val_loader, test_loader).
    """
    if config is None:
        config = ProjectConfig()
        
    dataset_root = config.get('dataset.root')
    split_type = config.get('dataset.split_type', 'original')
    clean_json = config.get('dataset.clean_split_json')
    class_mapping = config.get('dataset.class_mapping')
    batch_size = config.get('dataset.batch_size', 16)
    num_workers = config.get('dataset.num_workers', 2)
    target_size = config.get('dataset.input_size', [224, 224])
    enable_clahe = config.get('preprocessing.enable_clahe', True)
    
    # Preprocessor & Augmentor
    preprocessor = MedicalImagePreprocessor(target_size=target_size, enable_clahe=enable_clahe)
    augmentor = MedicalImageAugmentor()

    # Load samples per split
    if split_type == 'derived_clean':
        train_samples = load_derived_clean_samples(clean_json, 'train')
        val_samples = load_derived_clean_samples(clean_json, 'val')
        test_samples = load_derived_clean_samples(clean_json, 'test')
    else:
        train_samples = load_original_split_samples(dataset_root, 'train')
        val_samples = load_original_split_samples(dataset_root, 'val')
        test_samples = load_original_split_samples(dataset_root, 'test')

    train_dataset = BoneCancerDataset(train_samples, is_train=True, preprocessor=preprocessor, augmentor=augmentor, class_mapping=class_mapping)
    val_dataset = BoneCancerDataset(val_samples, is_train=False, preprocessor=preprocessor, class_mapping=class_mapping)
    test_dataset = BoneCancerDataset(test_samples, is_train=False, preprocessor=preprocessor, class_mapping=class_mapping)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader

if __name__ == '__main__':
    train_ld, val_ld, test_ld = get_dataloaders()
    print(f"DataLoaders created successfully. Train batches: {len(train_ld)}, Val batches: {len(val_ld)}, Test batches: {len(test_ld)}")
