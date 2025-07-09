# pointstream/core/scene.py
#
# This file defines the core data structures for representing video content.

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np

# Import the representation classes
from .representation import Pose, AppearanceEmbedding

# A type hint for bounding boxes for clarity
BoundingBox = Tuple[int, int, int, int]  # (x_min, y_min, x_max, y_max)

@dataclass
class DetectedObject:
    """
    Represents a single object detected within a frame.
    """
    track_id: int                 # A unique ID for the object across frames
    label: str                    # The object's class label (e.g., 'person', 'tennis_racket')
    bbox: BoundingBox             # The bounding box coordinates
    confidence: float             # The detection confidence score from the model
    
    # Use the specific types for semantic representations
    pose: Optional[Pose] = None
    appearance_embedding: Optional[AppearanceEmbedding] = None


@dataclass
class Frame:
    """
    Represents a single frame in a video, containing the image and detected objects.
    This object is intended for temporary, in-memory use.
    """
    frame_number: int
    image: np.ndarray             # The actual frame image as a NumPy array
    objects: List[DetectedObject] = field(default_factory=list)

    def __post_init__(self):
        # Ensure image is a numpy array
        if not isinstance(self.image, np.ndarray):
            raise TypeError("Frame image must be a NumPy array.")

    def get_object_by_id(self, track_id: int) -> Optional[DetectedObject]:
        """Finds a detected object in this frame by its tracking ID."""
        for obj in self.objects:
            if obj.track_id == track_id:
                return obj
        return None


@dataclass
class Scene:
    """
    Represents a continuous segment of a video with consistent characteristics.
    This object is memory-efficient and does NOT store frame data directly.
    """
    scene_id: int
    start_frame: int
    end_frame: int
    is_static_background: bool = False
    background_image: Optional[np.ndarray] = None
    
    # This field can be populated on-demand with frame objects if needed
    frames: List[Frame] = field(default_factory=list, repr=False, compare=False)

    @property
    def duration(self) -> int:
        """Returns the number of frames in the scene."""
        return self.end_frame - self.start_frame + 1

