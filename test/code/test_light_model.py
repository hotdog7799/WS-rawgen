
import torch
from forHJ import MWDNet_CPSF_depth
from forHJ_light import MWDNet_CPSF_depth_light
from ptflops import get_model_complexity_info
import sys

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    print("Initializing models...")
    # Dummy PSF: 51 depth levels, 1 channel, 512x512
    psf = torch.randn(51, 1, 256, 256) # 256 for testing as in code
    
    try:
        model_orig = MWDNet_CPSF_depth(n_channels=3, n_classes=51, psf=psf)
        params_orig = count_parameters(model_orig)
        print(f"Original Model Parameters: {params_orig:,}")
    except Exception as e:
        print(f"Original model init failed: {e}")
        params_orig = 0

    try:
        model_light = MWDNet_CPSF_depth_light(n_channels=3, n_classes=51, psf=psf, dim=16)
        params_light = count_parameters(model_light)
        print(f"Light Model Parameters:    {params_light:,}")
    except Exception as e:
        print(f"Light model init failed: {e}")
        params_light = 0

    if params_orig > 0:
        reduction = (1 - params_light / params_orig) * 100
        print(f"Parameter Reduction:       {reduction:.2f}%")

    # Forward pass test
    x = torch.randn(1, 3, 256, 256).cuda()
    psf = psf.cuda()
    
    if torch.cuda.is_available():
        print("\nTesting Forward Pass on CUDA...")
        try:
            model_light = model_light.cuda()
            # Need to update model.psf to cuda as it was cleared in init? No, it's stored in module.
            # But the forward pass takes 'current_psf = self.psf.to(x.device)'
            
            with torch.no_grad():
                intensity, depth = model_light(x)
            print(f"Light Model Output Shape: intensity={intensity.shape}, depth={depth.shape}")
            print("Forward pass successful!")
        except Exception as e:
            print(f"Forward pass failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("CUDA not available, skipping forward pass.")

if __name__ == "__main__":
    main()
