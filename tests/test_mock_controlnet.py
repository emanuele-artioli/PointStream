import torch
from types import SimpleNamespace
from src.decoder.genai_compositor import DiffusersCompositor

def test_mock_controlnet():
    print("Testing MockCaptionControlNetStrategy...")
    
    # Setup mock inputs
    # Shape: [C, H, W]
    reference_crop = torch.zeros((3, 256, 128), dtype=torch.uint8)
    
    # Shape: [Frames, 18, 3] or [18, 3] for pose
    dense_dwpose = torch.zeros((18, 3), dtype=torch.float32)
    
    # Warped background
    warped_background = torch.zeros((3, 1080, 1920), dtype=torch.uint8)
    
    config = SimpleNamespace(
        genai_backend="mock-caption-controlnet",
        gpu_dtype=None,
        animate_anyone_transparent_threshold=8,
        genai_resize_mode="aspect-recovery",
        animate_anyone_adaptive_threshold=True,
        animate_anyone_alpha_smoothing=0.25,
        compositing_mask_mode="postgen-seg-client",
        postgen_segmenter_backend="heuristic",  # Use heuristic to avoid loading YOLO in basic test
        postgen_segmenter_model=None,
        allow_auto_model_download=False,
    )
    
    compositor = DiffusersCompositor(
        confidence_threshold=0.2,
        backend="mock-caption-controlnet",
        seed=42,
        device="cpu",
        config=config,
    )
    
    print("Compositor initialized. Running process...")
    output = compositor.process(
        reference_crop_tensor=reference_crop,
        dense_dwpose_tensor=dense_dwpose,
        warped_background_frame=warped_background,
    )
    
    print(f"Output shape: {output.shape}")
    print(f"Output dtype: {output.dtype}")
    assert output.ndim == 3 and output.shape[0] == 3
    print("Test passed successfully!")

if __name__ == "__main__":
    test_mock_controlnet()
