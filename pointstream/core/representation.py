# pointstream/core/representation.py
#
# This file defines the core data structures for semantic representations
# like pose and appearance.

from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class Pose:
    """
    Represents the pose of an object, typically a person.
    """
    keypoints: np.ndarray  # Shape: (N, 2) or (N, 3) for (x, y, [confidence])
    
    # Optional metadata about the pose estimation
    source_model: Optional[str] = None # e.g., 'MediaPipe', 'OpenPose'

    def __post_init__(self):
        if not isinstance(self.keypoints, np.ndarray):
            raise TypeError("Keypoints must be a NumPy array.")
        if self.keypoints.ndim != 2:
            raise ValueError("Keypoints must be a 2D array.")

    @property
    def num_keypoints(self) -> int:
        """Returns the number of keypoints."""
        return self.keypoints.shape[0]


@dataclass
class AppearanceEmbedding:
    """
    Represents the visual appearance of an object as a compact vector.
    """
    vector: np.ndarray # A 1D feature vector
    
    # Optional metadata about the embedding
    source_model: Optional[str] = None # e.g., 'ResNet50', 'CustomEncoder'

    def __post_init__(self):
        if not isinstance(self.vector, np.ndarray):
            raise TypeError("Embedding vector must be a NumPy array.")
        if self.vector.ndim != 1:
            raise ValueError("Embedding must be a 1D vector.")
    
    @property
    def dimensions(self) -> int:
        """Returns the dimensionality of the embedding vector."""
        return self.vector.shape[0]

