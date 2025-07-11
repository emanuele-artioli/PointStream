from dataclasses import dataclass, field
from typing import List, Dict, Any
from .representation import Pose, AppearanceEmbedding

@dataclass
class DetectedObject:
    """Represents a single tracked object within a scene."""
    instance_id: int
    class_label: str
    bbox_per_frame: Dict[int, List[float]] = field(default_factory=dict)
    segmentation_mask_per_frame: Dict[int, Any] = field(default_factory=dict) # e.g., RLE
    pose: Pose = field(default_factory=Pose)
    appearance: AppearanceEmbedding = field(default_factory=AppearanceEmbedding)

@dataclass
class Scene:
    """Represents a single, coherent scene segment."""
    scene_id: str
    frame_count: int
    detected_objects: List[DetectedObject] = field(default_factory=list)
    background_image_path: str = ""
    camera_motion: List[Any] = field(default_factory=list) # List of transformation matrices