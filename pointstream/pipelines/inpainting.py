from .base import Inpainter
from typing import Any
import logging

class LaMaInpainter(Inpainter):
    def inpaint(self, frame: Any, mask: Any) -> Any:
        logging.info("Inpainting with LaMa...")
        # Placeholder: Return the original frame
        return frame

class ZITSInpainter(Inpainter):
    def inpaint(self, frame: Any, mask: Any) -> Any:
        raise NotImplementedError("ZITS inpainter has not been implemented.")