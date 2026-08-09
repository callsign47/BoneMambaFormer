"""
CNN PyTorch to HDF5 (.h5) Converter and Verifier
Converts PyTorch state_dict from checkpoints/cnn_best.pth into checkpoints/cnn_best.h5
and verifies structural integrity and numerical equality of saved weights.
"""

import os
import sys
import time
import json
import torch
import numpy as np
import h5py

# Add project root to sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.cnn import BoneCancerCNN

def export_pth_to_h5(pth_path, h5_path):
    print(f"Loading PyTorch checkpoint from: {pth_path}")
    checkpoint = torch.load(pth_path, map_location="cpu")
    
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    epoch = checkpoint.get("epoch", None)
    val_acc = checkpoint.get("val_acc", None)
    val_loss = checkpoint.get("val_loss", None)

    print(f"Exporting state_dict ({len(state_dict)} tensor keys) to HDF5 container: {h5_path}")
    
    os.makedirs(os.path.dirname(h5_path), exist_ok=True)
    
    with h5py.File(h5_path, "w") as h5f:
        # Save metadata attributes
        h5f.attrs["architecture"] = "BoneCancerCNN (ResNet18 backbone + 256D projection + 2-class head)"
        h5f.attrs["input_shape"] = json.dumps([3, 224, 224])
        h5f.attrs["num_classes"] = 2
        h5f.attrs["class_mapping"] = json.dumps({"cancer": 0, "normal": 1})
        if epoch is not None:
            h5f.attrs["best_epoch"] = epoch
        if val_acc is not None:
            h5f.attrs["val_acc"] = float(val_acc)
        if val_loss is not None:
            h5f.attrs["val_loss"] = float(val_loss)
            
        weights_grp = h5f.create_group("weights")
        for key, tensor in state_dict.items():
            param_np = tensor.cpu().numpy()
            if param_np.ndim > 0:
                weights_grp.create_dataset(key, data=param_np, compression="gzip", compression_opts=4)
            else:
                weights_grp.create_dataset(key, data=param_np)

    print(f"Successfully saved {h5_path} (Size: {os.path.getsize(h5_path) / (1024*1024):.2f} MB)")

def verify_h5_export(h5_path, pth_path):
    print("\n--- VERIFYING HDF5 EXPORT ---")
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"Export file {h5_path} does not exist!")
        
    with h5py.File(h5_path, "r") as h5f:
        print("HDF5 Root Attributes:")
        for attr_k, attr_v in h5f.attrs.items():
            print(f"  - {attr_k}: {attr_v}")
            
        weights_grp = h5f["weights"]
        keys = list(weights_grp.keys())
        print(f"Total layer datasets stored in HDF5: {len(keys)}")
        
        # Verify against PyTorch pth file
        checkpoint = torch.load(pth_path, map_location="cpu")
        original_sd = checkpoint.get("model_state_dict", checkpoint)
        
        mismatches = 0
        total_params = 0
        reconstructed_sd = {}
        
        for key in original_sd.keys():
            if key not in weights_grp:
                print(f"ERROR: Key '{key}' missing from HDF5 export!")
                mismatches += 1
                continue
            
            orig_np = original_sd[key].cpu().numpy()
            h5_np = np.asarray(weights_grp[key])
            
            total_params += orig_np.size
            if not np.array_equal(orig_np, h5_np):
                print(f"ERROR: Mismatch in weights for key '{key}'")
                mismatches += 1
            else:
                reconstructed_sd[key] = torch.from_numpy(h5_np)

        if mismatches == 0:
            print(f"VERIFICATION PASSED: All {len(keys)} datasets ({total_params:,} parameters) match PyTorch checkpoint exactly (0 errors).")
        else:
            raise ValueError(f"VERIFICATION FAILED with {mismatches} mismatched tensors!")

        # Verify loading into BoneCancerCNN PyTorch model instance
        model = BoneCancerCNN(pretrained=False, feature_dim=256, num_classes=2)
        model.load_state_dict(reconstructed_sd)
        model.eval()
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy_input)
        print(f"PyTorch Model Load Test: Output shape = {output.shape}")
        print("HDF5 Export and Reconstruction Test Completed Successfully!\n")

if __name__ == "__main__":
    pth_file = os.path.join(PROJECT_ROOT, "checkpoints", "cnn_best.pth")
    h5_file = os.path.join(PROJECT_ROOT, "checkpoints", "cnn_best.h5")
    
    export_pth_to_h5(pth_file, h5_file)
    verify_h5_export(h5_file, pth_file)
