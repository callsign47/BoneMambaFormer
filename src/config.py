import os
import json

class ProjectConfig:
    def __init__(self, config_dict=None):
        if config_dict is None:
            config_dict = self.get_default_config()
        self._config = config_dict
        
    @staticmethod
    def get_default_config():
        workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return {
            'project': {
                'name': 'ICIMCPS_2026_Bone_Cancer',
                'workspace_root': workspace_root,
                'seed': 42
            },
            'dataset': {
                'root': os.path.join(workspace_root, 'DATASET'),
                'split_type': 'original', # 'original' or 'derived_clean'
                'clean_split_json': os.path.join(workspace_root, 'results', 'derived_leakage_clean_split.json'),
                'class_mapping': {
                    'cancer': 0,
                    'normal': 1
                },
                'num_classes': 2,
                'input_size': [224, 224], # Target input resolution for backbone models (or [640, 640] if full scale)
                'batch_size': 16,
                'num_workers': 2
            },
            'preprocessing': {
                'resize': [224, 224],
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225],
                'enable_denoising': False, # Disabled by default to preserve diagnostic morphology
                'enable_clahe': True,      # Mild contrast enhancement
                'clahe_clip_limit': 2.0,
                'clahe_tile_grid': [8, 8]
            },
            'augmentation': {
                'rotation_degrees': 15,
                'flip_prob': 0.5,
                'zoom_scale': [0.9, 1.1],
                'brightness': 0.1,
                'contrast': 0.1,
                'elastic_transform': {
                    'alpha': 10.0,
                    'sigma': 3.0,
                    'prob': 0.3
                }
            }
        }

    def get(self, key, default=None):
        keys = key.split('.')
        val = self._config
        for k in keys:
            if isinstance(val, dict) and k in val:
                val = val[k]
            else:
                return default
        return val

    def to_dict(self):
        return self._config

    def save_json(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self._config, f, indent=2)

if __name__ == '__main__':
    cfg = ProjectConfig()
    print("Default config loaded successfully.")
