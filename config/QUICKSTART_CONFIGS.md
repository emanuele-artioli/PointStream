# PointStream Config Quick Reference

## What Was Created

A new configuration system in `config/` with 5 pre-built profiles and comprehensive documentation.

### Files Created:
- `config/default.yaml` - Complete default with all CLI arguments documented
- `config/mock_fast.yaml` - Fast testing with lightweight mock extractors
- `config/baseline_residual_only.yaml` - Ablation baseline (residual-only, no GenAI)
- `config/genai_animate_anyone.yaml` - Full GenAI synthesis pipeline
- `config/low_bandwidth_optimization.yaml` - Extreme compression optimization
- `config/README.md` - Comprehensive documentation and usage guide

## Quick Start

### Run with default config:
```bash
python -m src.main --config config/default.yaml --input video.mp4
```

### Run with mock extractors (fast testing):
```bash
python -m src.main --config config/mock_fast.yaml --input video.mp4
```

### Run with GenAI enabled:
```bash
python -m src.main --config config/genai_animate_anyone.yaml --input video.mp4
```

### Run ablation baseline:
```bash
python -m src.main --config config/baseline_residual_only.yaml --input video.mp4
```

### Override config values from command line:
```bash
python -m src.main --config config/default.yaml --input video.mp4 --num-frames 100 --debug
```

## Config File Support Status

✅ **Already Implemented** - The config file functionality was already present in `src/main.py`:
- `--config` flag supports JSON and YAML files
- Config files provide defaults that CLI arguments can override
- Sensible parameter normalization (dashes ↔ underscores)

## Recommended Workflow for Ablation Testing

1. **Establish Baseline:**
   ```bash
   python -m src.main --config config/baseline_residual_only.yaml \
     --input test_video.mp4 --summary-file
   ```

2. **Test GenAI Variant:**
   ```bash
   python -m src.main --config config/genai_animate_anyone.yaml \
     --input test_video.mp4 --summary-file
   ```

3. **Compare Results:**
   - Compare `outputs/*/run_summary.json` files
   - Measure residual bitrate savings with GenAI
   - Evaluate PSNR/quality metrics

4. **Create Custom Ablations:**
   - Copy `default.yaml` to `ablation_<feature_name>.yaml`
   - Modify the ONE parameter you're testing
   - Document your changes in comments
   - Run and compare against baseline

## Key Features by Config

| Config | Use Case | Actor Mode | Ball Extractor | GenAI | Residuals |
|--------|----------|-----------|-----------------|-------|-----------|
| `default.yaml` | Reference/documentation | real | difference | ❌ | full_video |
| `mock_fast.yaml` | Quick testing | **mock** | **mock** | ❌ | players_only |
| `baseline_residual_only.yaml` | Ablation baseline | real | difference | ❌ | full_video |
| `genai_animate_anyone.yaml` | Full synthesis | real | segmentation | ✅ AnimateAnyone | full_video |
| `low_bandwidth_optimization.yaml` | Extreme compression | real | cascade | ✅ AnimateAnyone | players_only |

## For Custom Domain Adaptation

1. Copy `default.yaml` to your domain config:
   ```bash
   cp config/default.yaml config/soccer_domain.yaml
   ```

2. Update domain-specific parameters:
   ```yaml
   detector-caption: "soccer player"
   segmenter-caption: "soccer player"
   ball-extractor: cascade  # Works better for soccer balls
   ```

3. Test and iterate:
   ```bash
   python -m src.main --config config/soccer_domain.yaml --input soccer_video.mp4
   ```

## Performance Tuning Checklist

When optimizing for your use case:

- [ ] **Bitrate reduction?** Use `low_bandwidth_optimization.yaml` as starting point
- [ ] **Quality vs speed?** Adjust `animate-anyone-steps` (10-30 range)
- [ ] **GPU memory?** Set `gpu-dtype: fp16` and `panorama-warp-batch-size: 4`
- [ ] **CPU heavy?** Use `execution-pool: tagged` with `--cpu-workers N`
- [ ] **Background artifacts?** Increase `residual-background-downscale` (2-4)
- [ ] **Actor quality?** Increase `reference-jpeg-quality` (75-90)
- [ ] **Pose temporal smoothing?** Adjust `animate-anyone-alpha-smoothing` (0.0-1.0)

## Reproducibility Tips

- Always use `--summary-file` to generate metrics
- Keep config files in version control
- Document config changes with comments
- Use descriptive filenames for ablation tests
- Validate with `--dry-run` before batch runs

## Next Steps

1. Read `config/README.md` for comprehensive documentation
2. Run a quick test: `python -m src.main --config config/mock_fast.yaml --input test.mp4`
3. Create your first ablation test config
4. Compare baseline vs optimized performance
