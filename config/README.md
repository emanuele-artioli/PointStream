# PointStream Configurations

This directory contains YAML configuration profiles for running PointStream with reproducible defaults and ablation-friendly presets.

## Usage

Pass any profile with `--config`:

```bash
python -m src.main --config config/default.yaml --input /path/to/video.mp4
```

Command-line arguments still override config values, so you can start from a profile and tweak only the parameters you want to test.

## Profiles

- `default.yaml`: documented baseline with sensible defaults for every CLI option
- `mock_fast.yaml`: lightweight mock setup for quick validation and debugging
- `baseline_residual_only.yaml`: full-video residual baseline for ablation comparisons
- `genai_animate_anyone.yaml`: AnimateAnyone-enabled synthesis pipeline
- `low_bandwidth_optimization.yaml`: aggressive bitrate-saving profile

## Notes

- Use `null` for options you want the CLI or runtime to auto-detect.
- Keep one change per ablation config when measuring bitrate or quality impact.
- Use `--dry-run` to validate a config before a longer run.
