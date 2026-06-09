import json
from pathlib import Path
from src.shared.config import PointstreamConfig, load_config

def test_pointstream_config_post_init():
    config = PointstreamConfig(log_level="info")
    assert config.disable_debug_artifacts is True
    
    config2 = PointstreamConfig(log_level="debug")
    assert config2.disable_debug_artifacts is False

def test_pointstream_config_from_dict():
    data = {
        "input": "test.mp4",
        "detector-caption": "tennis player 1",
        "segmenter_caption": "tennis player 2",
        "evaluation_mode": "psnr, ssim",
    }
    config = PointstreamConfig.from_dict(data)
    assert config.source_uri == "test.mp4"
    assert config.target_class_caption == "tennis player 1"
    assert config.evaluation_mode == ["psnr", "ssim"]
    
    data2 = {
        "evaluation_mode": "none",
        "segmenter_caption": "player",
    }
    config2 = PointstreamConfig.from_dict(data2)
    assert config2.evaluation_mode == []
    assert config2.target_class_caption == "player"

def test_load_config(tmp_path: Path):
    json_path = tmp_path / "test_config.json"
    json_path.write_text(json.dumps({"input": "json.mp4"}))
    
    config = load_config(json_path, cli_overrides={"log_level": "info"})
    assert config.source_uri == "json.mp4"
    assert config.log_level == "info"
    
    yaml_path = tmp_path / "test_config.yaml"
    yaml_path.write_text("input: yaml.mp4\nevaluation_mode: psnr")
    
    config2 = load_config(yaml_path)
    assert config2.source_uri == "yaml.mp4"
    assert config2.evaluation_mode == ["psnr"]
