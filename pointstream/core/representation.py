from dataclasses import dataclass, field
from typing import Dict, Any
import numpy as np

@dataclass
class Pose:
    """Container for pose keypoints for a single object over a scene."""
    # Maps frame_index -> np.array of (x, y, confidence)
    keypoints_per_frame: Dict[int, np.ndarray] = field(default_factory=dict)

@dataclass
class AppearanceEmbedding:
    """Container for the appearance embedding of a single object."""
    vector: np.ndarray = None

    def __post_init__(self):
        if self.vector is None:
            self.vector = np.array([])