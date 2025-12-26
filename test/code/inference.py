import os
import torch
import numpy as np
from PIL import Image
from torchvision.transforms import v2
import matplotlib.pyplot as plt

# Import your project's custom modules
from utils import *
from forHJ import MWDNet_CPSF_depth

# --- 1. CONFIGURE YOUR SETTINGS HERE ---

# ⬇️ Set this to the path of the .pth file you want to use (e.g., epoch 8)
CHECKPOINT_PATH = "/home/hotdog/20251029-023147_nnDP_20K_v14_MWDN3D_increase_params_w_UpG_ConvT_softmax_single_decoder_DepthRefine/model_20251029-023147_epoch_8.pth" 

# ⬇️ Set this to the path of your 1500x1500 raw image
INPUT_IMAGE_PATH = "/home/hotdog/test/images/1500_08_lm1.png"

# ⬇️ Set this to where you want the final depth map saved
OUTPUT_DEPTH_MAP_PATH = "/home/hotdog/inference_result_depth_08_lm1.png"
OUTPUT_AIF_PATH = "/home/hotdog/inference_result_aif_08_lm1.png"

# These HPARAMS must match your training script
PSF_PATH = "/home/hotdog/files_251026/image_stack_selected_resized_512_060_080.mat"
IN_CHANNEL = 3
OUT_CHANNEL = 21 # This is your depth-stack size
# --- End of Configuration ---


def main_inference():
    print("Starting inference...")
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")

    # 1. Load PSF (same as training)
    print(f"Loading PSF from {PSF_PATH}...")
    psf = psf_Load(PSF_PATH, 'psf_stack').permute(0, 2, 1, 3).to(DEVICE)
    
    # 2. Load Depth Range (same as training)
    depth_range_tensor = get_depth_range().to(DEVICE)

    # 3. Initialize Model (same as training)
    print("Initializing model structure...")
    model = MWDNet_CPSF_depth(
        n_channels=IN_CHANNEL, 
        n_classes=OUT_CHANNEL, 
        psf=psf, 
        height=512, 
        width=512
    )
    model = torch.nn.DataParallel(model).to(DEVICE)

    # 4. Load Saved Weights
    print(f"Loading weights from {CHECKPOINT_PATH}...")
    checkpoint = torch.load(CHECKPOINT_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])

    # 5. [CRITICAL] Set model to evaluation mode
    model.eval()

    # 6. Load and Preprocess Input Image
    print(f"Loading and preprocessing input image: {INPUT_IMAGE_PATH}")
    # Define the transform to resize the 1500x1500 image to 512x512
    inference_transform = v2.Compose([
        v2.ToTensor(),
        v2.Resize((512, 512), interpolation=v2.InterpolationMode.BICUBIC)
    ])
    
    raw_img = Image.open(INPUT_IMAGE_PATH).convert('RGB')
    image_tensor = inference_transform(raw_img).unsqueeze(0).to(DEVICE) # [1, 3, 512, 512]

    # 7. Run Inference
    print("Running model inference...")
    with torch.no_grad(): # Disable gradient calculation
        
        # Run the model
        aif_normalized, soft_max_depth_stack = model(image_tensor)

        # 8. Post-process to get Depth Map (same as training)
        depth_range_expanded = depth_range_tensor.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        depth_range_expanded = depth_range_expanded.expand(
            soft_max_depth_stack.size(0), -1, 
            soft_max_depth_stack.size(2), soft_max_depth_stack.size(3)
        )
        
        # Calculate the depth map (values 0-1)
        output_depth_map_normalized = torch.sum(depth_range_expanded * soft_max_depth_stack, dim=1, keepdim=True)
        
        # Invert it (same as training)
        final_depth_map_normalized = (1.0 - output_depth_map_normalized)

        # 9. Convert to CPU/Numpy for saving
        # Squeeze removes batch and channel dims: [1, 1, 512, 512] -> [512, 512]
        depth_map_np = final_depth_map_normalized.squeeze().cpu().numpy()
        aif_np = aif_normalized.squeeze().cpu().numpy().transpose(1, 2, 0) # [C, H, W] -> [H, W, C]

    # 10. Save Results
    # Save the All-in-Focus (AIF) image
    plt.imsave(OUTPUT_AIF_PATH, aif_np.clip(0, 1))
    print(f"All-in-focus image saved to: {OUTPUT_AIF_PATH}")
    
    # Save the depth map using a colormap
    plt.imsave(OUTPUT_DEPTH_MAP_PATH, depth_map_np, cmap='jet')
    print(f"Depth map saved to: {OUTPUT_DEPTH_MAP_PATH}")
    print("Inference complete.")

if __name__ == "__main__":
    main_inference()