import torch
import sys
import os

# Add the directory containing the model files to python path
sys.path.append("/home/hjahn/HJA_nas/lensless-depth/WS-rawgen/test/code/")

from forHJ_light_modulated import MWDNet_CPSF_depth_light

MODEL_PATH = "/home/hjahn/HJA_nas/lensless-depth/WS-rawgen/pth_512_uv-light-newobj/model_51ch/20260214-090327_model_aiobio.pth"
DEVICE = torch.device("cpu") # Use CPU for check

def check_compatibility():
    print(f"Loading checkpoint from: {MODEL_PATH}")
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return

    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    # Clean 'module.' prefix if present (DataParallel)
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    print(f"Checkpoint loaded. Number of keys: {len(state_dict)}")
    
    # Instantiate the current model (Light version)
    # Need dummy PSF
    dummy_psf = torch.randn(51, 1, 512, 512)
    model = MWDNet_CPSF_depth_light(n_channels=3, n_classes=51, psf=dummy_psf, height=256, width=256)
    
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state_dict.keys())
    
    print(f"Current Model keys: {len(model_keys)}")
    
    intersection = model_keys.intersection(ckpt_keys)
    missing_in_model = ckpt_keys - model_keys
    missing_in_ckpt = model_keys - ckpt_keys
    
    print(f"Keys matching: {len(intersection)}")
    print(f"Keys in Checkpoint but NOT in Model: {len(missing_in_model)}")
    print(f"Keys in Model but NOT in Checkpoint: {len(missing_in_ckpt)}")
    
    if len(missing_in_model) > 0:
        print("\nExample/First 5 matching failures (Checkpoint -> Model):")
        for k in list(missing_in_model)[:5]:
            print(f"  {k}")

    if len(missing_in_ckpt) > 0:
        print("\nExample/First 5 matching failures (Model -> Checkpoint):")
        for k in list(missing_in_ckpt)[:5]:
            print(f"  {k}")
            
    # Try strict loading to see the error
    print("\nAttempting strict load_state_dict:")
    try:
        model.load_state_dict(state_dict, strict=True)
        print("SUCCESS: load_state_dict with strict=True worked!")
    except RuntimeError as e:
        print("FAILURE: load_state_dict failed as expected.")
        print(f"Error message snippet: {str(e)[:500]}...")

if __name__ == "__main__":
    check_compatibility()
