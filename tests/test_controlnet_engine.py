import torch
from types import SimpleNamespace

from src.decoder.controlnet_engine import MockCaptionControlNetStrategy, CaptionControlNetStrategy

def test_mock_caption_controlnet_strategy() -> None:
    strategy = MockCaptionControlNetStrategy()
    
    reference = torch.zeros((3, 64, 64), dtype=torch.float32)
    pose = torch.zeros((18, 3), dtype=torch.float32)
    
    out = strategy.generate(reference, pose, seed=42, device=torch.device("cpu"))
    assert out.shape == (3, 64, 64)
    assert out.dtype == torch.uint8

    # Check with metadata_bbox included
    out2 = strategy.generate(reference, pose, seed=42, device=torch.device("cpu"), metadata_bbox=(0, 0, 64, 64))
    assert out2.shape == (3, 64, 64)

def test_caption_controlnet_strategy_init() -> None:
    class DummyConfig:
        controlnet_steps = 15
        controlnet_strength = 0.8
        controlnet_cfg = 8.0
        controlnet_width = 256
        controlnet_height = 256
        
    strategy = CaptionControlNetStrategy(
        model_id="runwayml/stable-diffusion-v1-5",
        controlnet_id="lllyasviel/control_v11p_sd15_openpose",
        config=DummyConfig(),
    )
    assert strategy._steps == 15
    assert strategy._strength == 0.8

def test_caption_controlnet_strategy_generate(monkeypatch) -> None:
    class DummyConfig:
        controlnet_steps = 15
        controlnet_strength = 0.8
        controlnet_cfg = 8.0
        controlnet_width = 256
        controlnet_height = 256
        
    strategy = CaptionControlNetStrategy(config=DummyConfig())
    
    class _StubPipe:
        def set_progress_bar_config(self, **kwargs):
            pass
        def to(self, device):
            return self
        def __call__(self, *args, **kwargs):
            import numpy as np
            from PIL import Image
            img = Image.fromarray(np.zeros((kwargs.get("height", 512), kwargs.get("width", 512), 3), dtype=np.uint8))
            return SimpleNamespace(images=[img])
            
    class _StubModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return _StubPipe()
            
    import diffusers
    monkeypatch.setattr(diffusers, "ControlNetModel", _StubModel)
    monkeypatch.setattr(diffusers, "StableDiffusionControlNetImg2ImgPipeline", _StubModel)
    
    # Mock VLM
    class _StubProcessor:
        def __call__(self, *args, **kwargs):
            return SimpleNamespace(to=lambda x: {"input_ids": torch.zeros((1,1))})
        def decode(self, *args, **kwargs):
            return "A fake caption"
            
    class _StubVLMModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()
        def to(self, *args, **kwargs):
            return self
        def eval(self):
            return self
        def generate(self, *args, **kwargs):
            return [torch.zeros((1,))]
            
    class _StubVLMProcessorModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return _StubProcessor()
            
    # Mock Transformers
    import transformers
    monkeypatch.setattr(transformers, "BlipProcessor", _StubVLMProcessorModel)
    monkeypatch.setattr(transformers, "BlipForConditionalGeneration", _StubVLMModel)
    
    import src.decoder.controlnet_engine as engine
    monkeypatch.setattr(engine, "BlipProcessor", transformers.BlipProcessor, raising=False)
    monkeypatch.setattr(engine, "BlipForConditionalGeneration", transformers.BlipForConditionalGeneration, raising=False)
    
    reference = torch.zeros((3, 64, 64), dtype=torch.float32)
    pose = torch.zeros((18, 3), dtype=torch.float32)
    
    out = strategy.generate(reference, pose, seed=42, device=torch.device("cpu"), metadata_bbox=(10, 10, 50, 50))
    assert out.shape == (3, 256, 256)
