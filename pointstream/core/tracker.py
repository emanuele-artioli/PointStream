# pointstream/core/tracker.py
#
# Contains a simple tracker to maintain object identities across frames.

from typing import List, Dict
from scipy.optimize import linear_sum_assignment
import numpy as np

from .scene import DetectedObject, BoundingBox

def _calculate_iou(box1: BoundingBox, box2: BoundingBox) -> float:
    """Calculates the Intersection over Union (IoU) of two bounding boxes."""
    x1, y1, x2, y2 = box1
    x1_b, y1_b, x2_b, y2_b = box2

    inter_x1 = max(x1, x1_b)
    inter_y1 = max(y1, y1_b)
    inter_x2 = min(x2, x2_b)
    inter_y2 = min(y2, y2_b)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_b - x1_b) * (y2_b - y1_b)
    
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0

class SimpleTracker:
    """
    A simple tracker using IoU and the Hungarian algorithm for assignment.
    """
    def __init__(self, iou_threshold=0.3):
        self.next_track_id = 0
        self.active_tracks: Dict[int, DetectedObject] = {}
        self.iou_threshold = iou_threshold

    def update(self, new_detections: List[DetectedObject]) -> List[DetectedObject]:
        """Updates the tracker with new detections for the current frame."""
        if not self.active_tracks:
            # First frame, initialize all detections as new tracks
            for det in new_detections:
                det.track_id = self.next_track_id
                self.active_tracks[self.next_track_id] = det
                self.next_track_id += 1
            return new_detections

        # Build cost matrix based on IoU
        track_ids = list(self.active_tracks.keys())
        cost_matrix = np.zeros((len(track_ids), len(new_detections)))

        for i, track_id in enumerate(track_ids):
            for j, det in enumerate(new_detections):
                iou = _calculate_iou(self.active_tracks[track_id].bbox, det.bbox)
                cost_matrix[i, j] = 1 - iou # Use 1 - IoU as cost

        # Use Hungarian algorithm to find optimal assignments
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Update matched tracks
        matched_track_ids = set()
        for r, c in zip(row_ind, col_ind):
            track_id = track_ids[r]
            if cost_matrix[r, c] < (1 - self.iou_threshold):
                new_det = new_detections[c]
                new_det.track_id = track_id
                self.active_tracks[track_id] = new_det
                matched_track_ids.add(track_id)

        # Handle new, unmatched detections
        for i, det in enumerate(new_detections):
            if i not in col_ind:
                det.track_id = self.next_track_id
                self.active_tracks[self.next_track_id] = det
                self.next_track_id += 1
        
        # Remove lost tracks (can be refined later)
        # For simplicity, we keep all tracks active for now.

        return new_detections
