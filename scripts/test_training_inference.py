from pathlib import Path

import torch
import cv2
import numpy as np

from src.shared.tennis_dataset import TennisSkeletonDataset
from scripts.train_pix2pix import UNetGenerator
from src.shared.spade4tennis_arch import SPADEResNet9Generator

OUTPUT_DIR = Path("outputs/inference_smoke")

def tensor_to_image(tensor):
    # tensor: [C, H, W] in range [-1, 1]
    img = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    img = (img * 0.5 + 0.5) * 255.0
    img = img.clip(0, 255).astype(np.uint8)
    # BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def test_inference():
    print("Testing inference on Pix2Pix and SPADE...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = TennisSkeletonDataset(
        root_dir="assets/dataset",
        target_size=256,
        include_reference=True
    )
    if len(dataset) == 0:
        print("Dataset empty!")
        return
        
    skel, ref, _ = dataset[0]
    skel = skel.unsqueeze(0).to(device)
    ref = ref.unsqueeze(0).to(device)
    
    # Pix2Pix
    print("Loading Pix2Pix...")
    try:
        p2p_ckpt = "assets/weights/pix2pix_generator.pt"
        p2p_net = UNetGenerator(in_channels=6, out_channels=3).to(device)
        p2p_net.load_state_dict(torch.load(p2p_ckpt, map_location=device))
        p2p_net.eval()
        
        with torch.no_grad():
            out_p2p = p2p_net(torch.cat((skel, ref), 1))
        
        out_path = OUTPUT_DIR / "pix2pix.jpg"
        cv2.imwrite(str(out_path), tensor_to_image(out_p2p[0]))
        print(f"Saved Pix2Pix inference to {out_path}")
    except Exception as e:
        print(f"Pix2Pix inference failed: {e}")
        
    # SPADE
    print("Loading SPADE...")
    try:
        spade_ckpt = "assets/weights/spade4tennis_lite_generator.pt"
        spade_net = SPADEResNet9Generator(in_nc=3, out_nc=3).to(device)
        # Handle DDP state dict prefix if it exists
        state_dict = torch.load(spade_ckpt, map_location=device)
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        spade_net.load_state_dict(new_state_dict)
        spade_net.eval()
        
        with torch.no_grad():
            out_spade = spade_net(skel, ref)
            
        out_path = OUTPUT_DIR / "spade.jpg"
        cv2.imwrite(str(out_path), tensor_to_image(out_spade[0]))
        print(f"Saved SPADE inference to {out_path}")
    except Exception as e:
        print(f"SPADE inference failed: {e}")

if __name__ == "__main__":
    test_inference()
