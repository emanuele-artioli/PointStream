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
            
    import sys
    from types import ModuleType
    mock_diffusers = ModuleType("diffusers")
    setattr(mock_diffusers, "ControlNetModel", _StubModel)
    setattr(mock_diffusers, "StableDiffusionControlNetImg2ImgPipeline", _StubModel)
    monkeypatch.setitem(sys.modules, "diffusers", mock_diffusers)
    
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
    mock_transformers = ModuleType("transformers")
    setattr(mock_transformers, "BlipProcessor", _StubVLMProcessorModel)
    setattr(mock_transformers, "BlipForConditionalGeneration", _StubVLMModel)
    monkeypatch.setitem(sys.modules, "transformers", mock_transformers)
    
    import src.decoder.controlnet_engine as engine
    monkeypatch.setattr(engine, "BlipProcessor", _StubVLMProcessorModel, raising=False)
    monkeypatch.setattr(engine, "BlipForConditionalGeneration", _StubVLMModel, raising=False)
    
    reference = torch.zeros((3, 64, 64), dtype=torch.float32)
    pose = torch.zeros((18, 3), dtype=torch.float32)
    
    out = strategy.generate(reference, pose, seed=42, device=torch.device("cpu"), metadata_bbox=(10, 10, 50, 50))
    assert out.shape == (3, 256, 256)


def test_ipadapter_controlnet_strategy_init() -> None:
    from src.decoder.controlnet_engine import IPAdapterControlNetStrategy
    class DummyConfig:
        controlnet_steps = 15
        controlnet_cfg = 8.0
        controlnet_width = 256
        controlnet_height = 256
        ip_adapter_scale = 0.7
        ip_adapter_weight = "custom.bin"
        
    strategy = IPAdapterControlNetStrategy(
        model_id="runwayml/stable-diffusion-v1-5",
        controlnet_id="lllyasviel/control_v11p_sd15_openpose",
        config=DummyConfig(),
    )
    assert strategy._steps == 15
    assert strategy._ip_adapter_scale == 0.7
    assert strategy._ip_adapter_weight == "custom.bin"


def test_ipadapter_controlnet_strategy_generate(monkeypatch) -> None:
    from src.decoder.controlnet_engine import IPAdapterControlNetStrategy
    class DummyConfig:
        controlnet_steps = 15
        controlnet_cfg = 8.0
        controlnet_width = 256
        controlnet_height = 256
        ip_adapter_scale = 0.7
        
    strategy = IPAdapterControlNetStrategy(config=DummyConfig())
    
    class _StubPipe:
        def set_progress_bar_config(self, **kwargs):
            pass
        def to(self, device):
            return self
        def load_ip_adapter(self, *args, **kwargs):
            self.loaded_ip_adapter = True
        def set_ip_adapter_scale(self, scale):
            self.ip_adapter_scale = scale
        def __call__(self, *args, **kwargs):
            import numpy as np
            from PIL import Image
            assert "ip_adapter_image" in kwargs
            img = Image.fromarray(np.zeros((kwargs.get("height", 512), kwargs.get("width", 512), 3), dtype=np.uint8))
            return SimpleNamespace(images=[img])
            
    class _StubModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return _StubPipe()
            
    import sys
    from types import ModuleType
    mock_diffusers = ModuleType("diffusers")
    setattr(mock_diffusers, "ControlNetModel", _StubModel)
    setattr(mock_diffusers, "StableDiffusionControlNetPipeline", _StubModel)
    monkeypatch.setitem(sys.modules, "diffusers", mock_diffusers)
    
    reference = torch.zeros((3, 64, 64), dtype=torch.float32)
    pose = torch.zeros((18, 3), dtype=torch.float32)
    
    out = strategy.generate(reference, pose, seed=42, device=torch.device("cpu"), metadata_bbox=(10, 10, 50, 50))
    assert out.shape == (3, 256, 256)
