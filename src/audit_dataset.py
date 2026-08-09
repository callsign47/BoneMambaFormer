import os
import json
import hashlib
from collections import Counter
from PIL import Image
import matplotlib.pyplot as plt

def compute_hash(filepath, block_size=65536):
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read(block_size)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(block_size)
    return hasher.hexdigest()

def audit_dataset(dataset_root, output_json, output_grid):
    splits = {
        'train': 'train_sorted-20251125T062947Z-1-001',
        'val': 'valid_sorted-20251125T062929Z-1-001',
        'test': 'test_sorted-20251125T062917Z-1-001'
    }
    classes = ['cancer', 'normal']
    
    stats = {
        'total_images': 0,
        'splits': {},
        'overall_class_counts': {'cancer': 0, 'normal': 0},
        'dimensions': Counter(),
        'color_modes': Counter(),
        'formats': Counter(),
        'corrupted_files': [],
        'hash_duplicates': {}, # hash: list of filepaths
        'duplicate_hash_count': 0
    }

    hash_to_paths = {}
    sample_images = {} # split -> class -> list of sample paths

    for split_name, folder_name in splits.items():
        split_path = os.path.join(dataset_root, folder_name)
        stats['splits'][split_name] = {'total': 0, 'cancer': 0, 'normal': 0}
        sample_images[split_name] = {}
        
        for c in classes:
            class_path = os.path.join(split_path, c)
            sample_images[split_name][c] = []
            if not os.path.exists(class_path):
                print(f"Warning: Directory missing {class_path}")
                continue
                
            files = os.listdir(class_path)
            for fname in files:
                fpath = os.path.join(class_path, fname)
                if not os.path.isfile(fpath):
                    continue
                
                # Check corruption & properties
                try:
                    with Image.open(fpath) as img:
                        img.verify() # verify integrity
                    with Image.open(fpath) as img:
                        stats['dimensions'][f"{img.size[0]}x{img.size[1]}"] += 1
                        stats['color_modes'][img.mode] += 1
                        stats['formats'][img.format] += 1
                        
                        if len(sample_images[split_name][c]) < 4:
                            sample_images[split_name][c].append(fpath)
                except Exception as e:
                    stats['corrupted_files'].append({'path': fpath, 'error': str(e)})
                    continue

                # Hash duplicate check
                fhash = compute_hash(fpath)
                if fhash in hash_to_paths:
                    hash_to_paths[fhash].append(fpath)
                else:
                    hash_to_paths[fhash] = [fpath]

                stats['total_images'] += 1
                stats['splits'][split_name]['total'] += 1
                stats['splits'][split_name][c] += 1
                stats['overall_class_counts'][c] += 1

    # Filter hash duplicates
    duplicates = {h: paths for h, paths in hash_to_paths.items() if len(paths) > 1}
    stats['hash_duplicates_count'] = len(duplicates)
    stats['total_duplicate_files'] = sum(len(paths) - 1 for paths in duplicates.values())

    # Convert Counters for JSON serialization
    stats['dimensions'] = dict(stats['dimensions'])
    stats['color_modes'] = dict(stats['color_modes'])
    stats['formats'] = dict(stats['formats'])

    # Save JSON report
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Audit JSON saved to {output_json}")

    # Generate Image Grid
    os.makedirs(os.path.dirname(output_grid), exist_ok=True)
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    plt.suptitle("Dataset Sample Grid (Train / Val / Test - Cancer & Normal)", fontsize=14)
    
    row_idx = 0
    for split_name in ['train', 'val', 'test']:
        col_idx = 0
        for c in ['normal', 'cancer']:
            for i in range(2):
                ax = axes[row_idx, col_idx]
                if i < len(sample_images[split_name][c]):
                    imgPath = sample_images[split_name][c][i]
                    img = Image.open(imgPath)
                    ax.imshow(img)
                    ax.set_title(f"{split_name.upper()} - {c.upper()} ({i+1})", fontsize=10)
                ax.axis('off')
                col_idx += 1
        row_idx += 1

    plt.tight_layout()
    plt.savefig(output_grid, dpi=300)
    plt.close()
    print(f"Sample grid figure saved to {output_grid}")

    return stats

if __name__ == '__main__':
    workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(workspace_root, 'DATASET')
    json_path = os.path.join(workspace_root, 'results', 'dataset_audit.json')
    grid_path = os.path.join(workspace_root, 'figures', 'dataset_sample_grid.png')
    
    print(f"Starting audit on dataset directory: {dataset_dir}")
    res = audit_dataset(dataset_dir, json_path, grid_path)
    print("Audit Complete!")
    print(f"Total images audited: {res['total_images']}")
    print(f"Splits breakdown: {res['splits']}")
    print(f"Overall class counts: {res['overall_class_counts']}")
    print(f"Dimensions: {res['dimensions']}")
    print(f"Color modes: {res['color_modes']}")
    print(f"Formats: {res['formats']}")
    print(f"Corrupted files count: {len(res['corrupted_files'])}")
    print(f"Duplicate image hash groups: {res['hash_duplicates_count']}")
