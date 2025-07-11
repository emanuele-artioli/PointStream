from abc import ABC, abstractmethod
from typing import List, Any

class SceneDetector(ABC):
    @abstractmethod
    def detect(self, frame_buffer: List[Any]) -> List[int]:
        """Returns a list of frame indices where scene cuts occur."""
        pass

class ObjectDetector(ABC):
    @abstractmethod
    def detect(self, frame: Any) -> List[Any]:
        """Detects and tracks objects in a single frame."""
        pass

class PoseEstimator(ABC):
    @abstractmethod
    def estimate(self, frame: Any, objects: List[Any]) -> List[Any]:
        """Estimates pose for detected objects."""
        pass

class Inpainter(ABC):
    @abstractmethod
    def inpaint(self, frame: Any, mask: Any) -> Any:
        """Fills in the masked area of a frame."""
        pass
# ... other base classes for AppearanceEncoder, GenerativeModel, etc.