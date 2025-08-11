# Video Scene Splitter Configuration Guide

This document explains all configuration options available in the `config.ini` file for the Video Scene Splitter system.

## Configuration Files

- **`config.ini`**: Main configuration file with all default settings
- **`config_test.ini`**: Test configuration with relaxed quality controls
- **Custom configs**: You can create your own config files and specify them with `--config path/to/config.ini`

## Configuration Sections

### [general]
Controls overall behavior of the scene splitter.

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `default_output_pattern` | String | `{input_stem}_scenes` | Output directory naming pattern. `{input_stem}` = filename without extension |
| `default_batch_size` | Integer | `100` | Default number of frames per batch for batch processing |
| `default_max_frames` | Integer | `1000` | Default maximum frames per scene before forced split |
| `default_threshold` | Float | `30.0` | Default scene detection sensitivity (lower = more scenes) |
| `create_scene_list` | Boolean | `true` | Whether to create a text file listing all scenes |
| `merge_adjacent_scenes` | Boolean | `true` | Whether to merge scenes that are very close together |
| `merge_tolerance` | Float | `0.1` | Maximum gap in seconds to merge adjacent scenes |

### [scene_detection]
Controls the scene detection algorithm parameters.

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `threshold` | Float | `30.0` | Scene change detection threshold (0-100+, lower = more sensitive) |
| `min_scene_len` | Integer | `15` | Minimum scene length in frames |
| `fade_bias` | Float | `0.0` | Bias for fade detection (-1.0 to 1.0) |
| `luma_only` | Boolean | `false` | Use only brightness for detection (faster but less accurate) |
| `kernel_size` | Integer | `3` | Gaussian kernel size for content detection |

### [video_processing]
Controls video input processing parameters.

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `skip_frames` | Integer | `0` | Number of frames to skip at the beginning |
| `max_duration` | Integer | `0` | Maximum duration to process in seconds (0 = entire video) |
| `frame_skip` | Integer | `1` | Skip every N frames during processing (1 = process all) |
| `downscale_factor` | Float | `1.0` | Factor to downscale video for processing (1.0 = no scaling) |

### [encoding]
Controls video and audio encoding parameters.

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `video_codec` | String | `libsvtav1` | Video codec (libsvtav1, libx264, libx265, etc.) |
| `audio_codec` | String | `libopus` | Audio codec (libopus, aac, mp3, etc.) |
| `crf` | Integer | `30` | Constant Rate Factor (0-51, lower = better quality) |
| `preset` | String/Integer | `6` | Encoding speed preset (codec-dependent) |
| `pixel_format` | String | `yuv420p` | Pixel format for output video |
| `color_range` | String | `tv` | Color range (tv, pc) |
| `color_space` | String | `bt709` | Color space (bt709, bt2020, etc.) |
| `audio_bitrate` | String | `128k` | Audio bitrate |
| `audio_sample_rate` | Integer | `48000` | Audio sample rate in Hz |
| `audio_channels` | Integer | `2` | Number of audio channels |
| `two_pass` | Boolean | `false` | Use two-pass encoding (slower but better quality) |
| `tune` | String | `` | Encoding tune parameter (codec-dependent) |
| `profile` | String | `` | Encoding profile (codec-dependent) |
| `container_format` | String | `mp4` | Output container format |
| `faststart` | Boolean | `true` | Enable fast start for web streaming |

### [ffmpeg]
Controls FFmpeg execution and behavior.

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `ffmpeg_binary` | String | `ffmpeg` | Path to FFmpeg executable |
| `ffmpeg_timeout` | Integer | `300` | Timeout for FFmpeg operations in seconds |
| `ffmpeg_threads` | Integer | `0` | Number of encoding threads (0 = auto) |
| `continue_on_error` | Boolean | `false` | Continue processing if one scene fails |
| `retry_failed` | Integer | `1` | Number of retries for failed encodes |
| `extra_input_args` | Array | `[]` | Additional FFmpeg input arguments |
| `extra_output_args` | Array | `[]` | Additional FFmpeg output arguments |

### [output]
Controls output file organization and metadata.

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `scene_filename_pattern` | String | `scene_{number:04d}.{extension}` | Scene file naming pattern |
| `scene_list_filename` | String | `scene_list.txt` | Scene list file name |
| `metadata_filename` | String | `metadata.json` | Metadata file name |
| `organize_by_duration` | Boolean | `false` | Organize scenes into duration-based subdirectories |
| `organize_by_size` | Boolean | `false` | Organize scenes into size-based subdirectories |
| `duration_thresholds` | Array | `[5, 15, 60]` | Duration thresholds in seconds for organization |
| `size_thresholds` | Array | `[1, 10, 50]` | Size thresholds in MB for organization |
| `export_metadata` | Boolean | `true` | Export scene metadata to JSON |
| `include_timestamps` | Boolean | `true` | Include timestamps in metadata |
| `include_file_info` | Boolean | `true` | Include file size/duration info |
| `include_encoding_info` | Boolean | `true` | Include encoding parameters in metadata |

### [quality_control]
Controls output validation and quality requirements.

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `min_scene_duration` | Float | `0.5` | Minimum scene duration in seconds |
| `max_scene_duration` | Float | `300` | Maximum scene duration in seconds (0 = no limit) |
| `min_file_size_kb` | Integer | `10` | Minimum output file size in KB |
| `max_file_size_mb` | Integer | `1000` | Maximum output file size in MB (0 = no limit) |
| `verify_output_files` | Boolean | `true` | Verify output files after creation |
| `check_video_integrity` | Boolean | `true` | Check video integrity with ffprobe |
| `remove_invalid_files` | Boolean | `false` | Remove files that fail validation |

### [batch_processing]
Controls batch processing behavior (for the main script).

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `batch_size` | Integer | `100` | Frames per batch |
| `overlap_frames` | Integer | `10` | Overlap between batches |
| `memory_limit_mb` | Integer | `1024` | Memory limit for batch processing |
| `parallel_batches` | Integer | `1` | Number of batches to process in parallel |
| `adaptive_batch_size` | Boolean | `true` | Automatically adjust batch size |
| `min_batch_size` | Integer | `50` | Minimum batch size when adaptive |
| `max_batch_size` | Integer | `500` | Maximum batch size when adaptive |

### [logging]
Controls logging and progress reporting.

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `log_level` | String | `INFO` | Log level (DEBUG, INFO, WARNING, ERROR) |
| `log_to_file` | Boolean | `true` | Save logs to file |
| `log_filename_pattern` | String | `{input_stem}_processing.log` | Log file naming pattern |
| `progress_bar` | Boolean | `true` | Show progress bar during processing |
| `verbose_ffmpeg` | Boolean | `false` | Show detailed FFmpeg output |
| `collect_statistics` | Boolean | `true` | Collect processing statistics |
| `statistics_filename` | String | `processing_stats.json` | Statistics file name |

### [performance]
Controls performance optimization settings.

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `use_gpu_acceleration` | Boolean | `false` | Use GPU acceleration if available |
| `gpu_device` | Integer | `0` | GPU device index |
| `memory_optimization` | Boolean | `true` | Enable memory optimizations |
| `disk_cache_size_mb` | Integer | `512` | Disk cache size for temporary files |
| `max_workers` | Integer | `4` | Maximum number of worker threads |
| `io_threads` | Integer | `2` | Number of I/O threads |
| `encoding_threads` | Integer | `0` | Number of encoding threads per job |

### [advanced]
Advanced and experimental features.

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `custom_scene_detectors` | Array | `[]` | Custom scene detector configurations |
| `post_processing_filters` | Array | `[]` | Post-processing video filters |
| `scene_transition_effects` | Array | `[]` | Transition effects between scenes |
| `experimental_features` | Boolean | `false` | Enable experimental features |
| `debug_mode` | Boolean | `false` | Enable debug mode |
| `dry_run` | Boolean | `false` | Perform dry run without actual encoding |
| `webhook_url` | String | `` | Webhook URL for completion notifications |
| `email_notifications` | Boolean | `false` | Send email notifications |
| `notification_events` | Array | `["completion", "error"]` | Events to notify about |

## Usage Examples

### 1. Basic Usage with Default Config
```bash
python simple_scene_splitter.py input.mp4
```

### 2. Show Current Configuration
```bash
python simple_scene_splitter.py --show-config
```

### 3. Use Custom Configuration
```bash
python simple_scene_splitter.py input.mp4 --config my_config.ini
```

### 4. Override Specific Settings
```bash
python simple_scene_splitter.py input.mp4 --threshold 20.0 --max-frames 500
```

### 5. Test Configuration (No Quality Checks)
```bash
python simple_scene_splitter.py input.mp4 --config config_test.ini
```

## Common Configuration Scenarios

### High Quality Output
```ini
[encoding]
video_codec = libx265
crf = 18
preset = slow

[quality_control]
verify_output_files = true
check_video_integrity = true
```

### Fast Processing
```ini
[encoding]
video_codec = libx264
crf = 23
preset = ultrafast

[quality_control]
verify_output_files = false
check_video_integrity = false
```

### High Sensitivity Scene Detection
```ini
[scene_detection]
threshold = 15.0
min_scene_len = 10
```

### Large File Processing
```ini
[batch_processing]
batch_size = 200
memory_limit_mb = 2048

[performance]
max_workers = 8
```

### Organized Output
```ini
[output]
organize_by_duration = true
duration_thresholds = [3, 10, 30]
export_metadata = true
```

## Configuration Validation

The system automatically validates configuration values and will warn about:
- Invalid CRF values (must be 0-51)
- Negative durations or thresholds
- Inconsistent min/max values
- Missing required files or executables

Use `python config_manager.py` to test configuration validation independently.
