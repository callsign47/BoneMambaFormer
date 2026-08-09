import os
import json
import hashlib
from collections import defaultdict

def compute_md5(filepath, block_size=65536):
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read(block_size)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(block_size)
    return hasher.hexdigest()

def analyze_leakage():
    workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(workspace_root, 'DATASET')
    
    splits = {
        'train': 'train_sorted-20251125T062947Z-1-001',
        'val': 'valid_sorted-20251125T062929Z-1-001',
        'test': 'test_sorted-20251125T062917Z-1-001'
    }
    classes = ['cancer', 'normal']
    
    # Hash -> list of records: {'split': ..., 'class': ..., 'filename': ..., 'rel_path': ...}
    hash_to_records = defaultdict(list)
    
    for split_name, folder_name in splits.items():
        split_path = os.path.join(dataset_dir, folder_name)
        for c in classes:
            class_path = os.path.join(split_path, c)
            if not os.path.exists(class_path):
                continue
            for fname in os.listdir(class_path):
                fpath = os.path.join(class_path, fname)
                if not os.path.isfile(fpath):
                    continue
                h = compute_md5(fpath)
                rel_p = os.path.join(folder_name, c, fname)
                hash_to_records[h].append({
                    'split': split_name,
                    'class': c,
                    'filename': fname,
                    'rel_path': rel_p,
                    'abs_path': fpath
                })
                
    # Overlap metrics
    cross_split_groups = []
    intra_split_groups = []
    single_image_groups = []
    
    # Overlap pair counts
    overlap_counts = {
        'train_val': 0,
        'train_test': 0,
        'val_test': 0,
        'train_val_test': 0
    }
    
    label_disagreements = []
    
    for h, records in hash_to_records.items():
        if len(records) == 1:
            single_image_groups.append(h)
            continue
            
        splits_in_group = set(r['split'] for r in records)
        labels_in_group = set(r['class'] for r in records)
        
        # Check label agreement
        if len(labels_in_group) > 1:
            label_disagreements.append({
                'hash': h,
                'labels': list(labels_in_group),
                'records': records
            })
            
        if len(splits_in_group) > 1:
            cross_split_groups.append({
                'hash': h,
                'splits': list(splits_in_group),
                'labels': list(labels_in_group),
                'count': len(records),
                'records': records
            })
            
            if splits_in_group == {'train', 'val'}:
                overlap_counts['train_val'] += 1
            elif splits_in_group == {'train', 'test'}:
                overlap_counts['train_test'] += 1
            elif splits_in_group == {'val', 'test'}:
                overlap_counts['val_test'] += 1
            elif splits_in_group == {'train', 'val', 'test'}:
                overlap_counts['train_val_test'] += 1
        else:
            intra_split_groups.append({
                'hash': h,
                'split': list(splits_in_group)[0],
                'count': len(records),
                'records': records
            })
            
    print(f"Total Unique MD5 Hashes: {len(hash_to_records)}")
    print(f"Single Instance Hashes: {len(single_image_groups)}")
    print(f"Intra-Split Duplicate Hash Groups: {len(intra_split_groups)}")
    print(f"Cross-Split Duplicate Hash Groups: {len(cross_split_groups)}")
    print("\nCross-Split Breakdown:")
    print(f"  Train <-> Val overlap groups: {overlap_counts['train_val']}")
    print(f"  Train <-> Test overlap groups: {overlap_counts['train_test']}")
    print(f"  Val <-> Test overlap groups: {overlap_counts['val_test']}")
    print(f"  Train <-> Val <-> Test overlap groups: {overlap_counts['train_val_test']}")
    print(f"\nLabel Disagreements across duplicate hashes: {len(label_disagreements)}")
    
    # Generate Derived Leakage-Clean Split Mapping
    # Rule: Keep all images of an MD5 hash in ONE split together.
    # Group hashes: assign whole hash groups to train / val / test aiming for 80/10/10 split on unique groups or image counts.
    # If a hash group originally contains any 'train' image, prioritize train unless needed for val/test balancing.
    # Actually, let's sort unique hash groups by original split preference or random seed 42.
    
    clean_split_mapping = {'train': [], 'val': [], 'test': []}
    
    # We sort hashes deterministically
    sorted_hashes = sorted(hash_to_records.keys())
    
    # Count images per hash
    # To keep original split as close as possible:
    # If a group was purely train -> keep in train
    # If a group was purely val -> keep in val
    # If a group was purely test -> keep in test
    # If a group was cross-split -> reassign the entire group to 'train' to guarantee zero leakage into val & test!
    
    reassigned_cross_split_count = 0
    for h in sorted_hashes:
        recs = hash_to_records[h]
        splits_in_group = set(r['split'] for r in recs)
        
        if len(splits_in_group) == 1:
            target_split = list(splits_in_group)[0]
        else:
            # Reassign cross-split group entirely to 'train' so val/test remain 100% leak-free!
            target_split = 'train'
            reassigned_cross_split_count += 1
            
        for r in recs:
            clean_split_mapping[target_split].append({
                'rel_path': r['rel_path'],
                'abs_path': r['abs_path'],
                'class': r['class'],
                'hash': h,
                'original_split': r['split']
            })
            
    clean_counts = {
        'train': len(clean_split_mapping['train']),
        'val': len(clean_split_mapping['val']),
        'test': len(clean_split_mapping['test']),
        'total': sum(len(v) for v in clean_split_mapping.values())
    }
    
    clean_class_counts = {
        'train': {'cancer': sum(1 for x in clean_split_mapping['train'] if x['class']=='cancer'),
                  'normal': sum(1 for x in clean_split_mapping['train'] if x['class']=='normal')},
        'val': {'cancer': sum(1 for x in clean_split_mapping['val'] if x['class']=='cancer'),
                'normal': sum(1 for x in clean_split_mapping['val'] if x['class']=='normal')},
        'test': {'cancer': sum(1 for x in clean_split_mapping['test'] if x['class']=='cancer'),
                 'normal': sum(1 for x in clean_split_mapping['test'] if x['class']=='normal')}
    }
    
    print("\nDerived Leakage-Clean Split Image Counts:")
    print(f"  Train: {clean_counts['train']} ({clean_class_counts['train']})")
    print(f"  Val:   {clean_counts['val']} ({clean_class_counts['val']})")
    print(f"  Test:  {clean_counts['test']} ({clean_class_counts['test']})")
    
    # Save detailed JSON report
    report = {
        'total_images': 8810,
        'unique_md5_hashes': len(hash_to_records),
        'cross_split_groups_count': len(cross_split_groups),
        'intra_split_groups_count': len(intra_split_groups),
        'single_image_groups_count': len(single_image_groups),
        'overlap_breakdown': overlap_counts,
        'label_disagreements_count': len(label_disagreements),
        'label_disagreements_details': label_disagreements,
        'cross_split_groups': cross_split_groups,
        'derived_clean_split_summary': {
            'image_counts': clean_counts,
            'class_counts': clean_class_counts,
            'reassigned_cross_split_groups': reassigned_cross_split_count
        }
    }
    
    out_json = os.path.join(workspace_root, 'results', 'cross_split_leakage_detailed.json')
    with open(out_json, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nDetailed leakage analysis saved to {out_json}")
    
    out_clean_json = os.path.join(workspace_root, 'results', 'derived_leakage_clean_split.json')
    with open(out_clean_json, 'w') as f:
        json.dump(clean_split_mapping, f, indent=2)
    print(f"Derived clean split mapping saved to {out_clean_json}")

if __name__ == '__main__':
    analyze_leakage()
