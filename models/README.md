# PointStream Generative Models

This directory should contain the trained generative models for object reconstruction:

- `human_cgan.pth` - Conditional GAN model for human object generation
- `animal_cgan.pth` - Conditional GAN model for animal object generation  
- `other_cgan.pth` - Conditional GAN model for other object generation

These models are trained using the PointStream training pipeline and are required for full object reconstruction functionality.

## Training the Models

To train these models, you need:
1. A dataset of object patches with their corresponding segmentation masks
2. Run the training phase with sufficient training data
3. The models will be automatically saved to this directory

## Current Status

The models are currently missing, which means:
- Object generation will be skipped
- Only background reconstruction will be performed
- Quality will be lower as foreground objects won't be reconstructed

To enable full reconstruction, you need to train or obtain these model files.
