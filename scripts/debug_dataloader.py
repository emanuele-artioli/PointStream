import os
import torch
import torchvision.utils as vutils
from src.shared.tennis_dataset import TennisSkeletonDataset

def main():
    print("Testing TennisSkeletonDataset...")
    dataset = TennisSkeletonDataset(
        root_dir="assets/dataset",
        target_size=512,
        include_reference=True
    )
    
    print(f"Found {len(dataset)} items in the dataset.")
    if len(dataset) == 0:
        print("No items found! Check directory parsing logic.")
        return
        
    os.makedirs("assets/debug_dataloader", exist_ok=True)
    
    # Let's save the first 4 items
    skeletons = []
    refs = []
    gts = []
    for i in range(min(4, len(dataset))):
        skeleton, ref, gt = dataset[i]
        print(f"Item {i} shapes - Skeleton: {skeleton.shape}, Ref: {ref.shape}, GT: {gt.shape}")
        skeletons.append(skeleton)
        refs.append(ref)
        gts.append(gt)
        
    # Stack them
    skeletons_tensor = torch.stack(skeletons)
    refs_tensor = torch.stack(refs)
    gts_tensor = torch.stack(gts)
    
    # Save a grid: column 1 = Skeleton, column 2 = Reference, column 3 = GT
    # We can concatenate horizontally per image, then use vutils.make_grid
    combined = torch.cat((skeletons_tensor, refs_tensor, gts_tensor), dim=3) # concat along width
    
    # Normalise from [-1, 1] to [0, 1] for saving
    vutils.save_image(combined, "assets/debug_dataloader/batch_preview.png", normalize=True, value_range=(-1, 1))
    print("Saved preview to assets/debug_dataloader/batch_preview.png")

if __name__ == "__main__":
    main()
