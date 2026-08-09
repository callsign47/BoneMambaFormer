import os
import json
import hashlib

def compute_hash(filepath, block_size=65536):
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read(block_size)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(block_size)
    return hasher.hexdigest()

def check_cross_leak():
    workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(workspace_root, 'DATASET')
    
    splits = {
        'train': 'train_sorted-20251125T062947Z-1-001',
        'val': 'valid_sorted-20251125T062929Z-1-001',
        'test': 'test_sorted-20251125T062917Z-1-001'
    }
    classes = ['cancer', 'normal']
    
    hash_map = {} # hash -> list of (split, class, filename)
    
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
                fhash = compute_hash(fpath)
                if fhash not in hash_map:
                    hash_map[fhash] = []
                hash_map[fhash].append((split_name, c, fname))
                
    cross_split_duplicates = []
    intra_split_duplicates = []
    
    for fhash, entries in hash_map.items():
        if len(entries) > 1:
            splits_present = set(e[0] for e in entries)
            if len(splits_present) > 1:
                cross_split_duplicates.append((fhash, entries))
            else:
                intra_split_duplicates.append((fhash, entries))
                
    print(f"Total unique hash groups with duplicates: {len(cross_split_duplicates) + len(intra_split_duplicates)}")
    print(f"Intra-split duplicates (within same split): {len(intra_split_duplicates)}")
    print(f"Cross-split duplicates (leakage across splits): {len(cross_split_duplicates)}")
    
    if len(cross_split_duplicates) > 0:
        print("\nSample Cross-Split Duplicates:")
        for fhash, entries in cross_split_duplicates[:5]:
            print(f"Hash {fhash}: {entries}")
            
    # Save detailed analysis
    leak_report = {
        'total_unique_hashes': len(hash_map),
        'cross_split_leakage_groups': len(cross_split_duplicates),
        'intra_split_duplicate_groups': len(intra_split_duplicates),
        'cross_split_examples': cross_split_duplicates[:10]
    }
    
    output_path = os.path.join(workspace_root, 'results', 'hash_leakage_audit.json')
    with open(output_path, 'w') as f:
        json.dump(leak_report, f, indent=2)
    print(f"\nDetailed hash leakage audit saved to {output_path}")

if __name__ == '__main__':
    check_cross_leak()
