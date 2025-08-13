# Background Inpainting Configuration Guide

## Overview

The PointStream pipeline now includes a configurable background inpainting step. This allows you to choose whether to:

1. **Enable inpainting (default)**: Remove detected objects from frames and create clean background panoramas
2. **Disable inpainting**: Keep original frames with objects intact and create panoramas from them

## Configuration

### Enable/Disable Inpainting

In your `config.ini` file, modify the `[inpainting]` section:

```ini
[inpainting]
# Background inpainting configuration
# enable_inpainting: Whether to perform background inpainting (True) or skip it (False)
enable_inpainting = true    # Set to false to disable inpainting
method = telea
radius = 3
dilate_mask = true
dilate_kernel_size = 3
```

### Configuration Options

- **`enable_inpainting = true`** (default)
  - Objects are detected and segmented
  - Object masks are created
  - Backgrounds are inpainted to remove objects
  - Clean panoramas are created from inpainted backgrounds
  - Best for creating clean environment maps

- **`enable_inpainting = false`**
  - Objects are still detected and segmented (for metadata)
  - No inpainting is performed
  - Original frames (with objects) are used for panorama creation
  - Faster processing (skips inpainting step)
  - Best for creating panoramas that include moving objects

## Use Cases

### Inpainting Enabled (Default)
- **Clean environment mapping**: Create panoramas without people, vehicles, or other moving objects
- **Background analysis**: Study the static environment without distractions
- **Architectural documentation**: Capture buildings/spaces without temporary objects

### Inpainting Disabled
- **Activity panoramas**: Create panoramas that show the activity in a scene
- **Performance optimization**: Skip computationally expensive inpainting for faster processing
- **Sports analysis**: Capture fields/courts with players visible
- **Crowd analysis**: Study areas with people included

## Performance Impact

- **Inpainting enabled**: Higher processing time due to inpainting computation
- **Inpainting disabled**: Faster processing, only frame copying overhead

The pipeline will log which mode is being used:
```
Enable inpainting: True/False
Purpose: Create clean panoramas from inpainted backgrounds/original frames
```

## Example Usage

### With Custom Config File
```bash
# Use a config file with inpainting disabled
python server_pipeline.py input.mp4 --config config_no_inpainting.ini
```

### Default Behavior
```bash
# Uses default config.ini (inpainting enabled by default)
python server_pipeline.py input.mp4
```

## Output Data Structure

The processed data structure remains the same regardless of inpainting setting:

```python
processed_data = {
    'frames_data': [
        {
            'inpainted_background': frame,      # Always present (may be original if inpainting disabled)
            'background_frame': frame,          # The actual frame used for panorama
            'objects': [...],                   # Detected objects (always computed)
            'combined_mask': mask,              # Object masks (always computed)
            # ...
        }
    ],
    'scene_panorama': panorama,  # Panorama from processed frames
    # ...
}
```

## Migration

Existing configurations will default to `enable_inpainting = true` if the parameter is not explicitly set, maintaining backward compatibility.
