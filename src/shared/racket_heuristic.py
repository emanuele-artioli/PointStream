import math
from typing import Tuple, List, Optional
import numpy as np
import cv2

def get_dominant_wrist(person_kpts: np.ndarray, racket_bbox: Optional[Tuple[float, float, float, float]], dominant_hand: Optional[int] = None) -> Optional[Tuple[float, float, int]]:
    """
    Finds the wrist (index 4 or 7) closest to the racket center, or uses the forced dominant_hand.
    person_kpts: [18, 3] array of (x, y, conf)
    racket_bbox: (x1, y1, x2, y2)
    Returns: (wrist_x, wrist_y, wrist_index) or None if neither wrist is valid.
    """
    if racket_bbox is None:
        return None
        
    rx1, ry1, rx2, ry2 = racket_bbox
    cx = (rx1 + rx2) / 2.0
    cy = (ry1 + ry2) / 2.0

    r_wrist = person_kpts[4]
    l_wrist = person_kpts[7]

    r_valid = r_wrist[2] > 0.1
    l_valid = l_wrist[2] > 0.1

    if dominant_hand == 4 and r_valid:
        return (float(r_wrist[0]), float(r_wrist[1]), 4)
    elif dominant_hand == 7 and l_valid:
        return (float(l_wrist[0]), float(l_wrist[1]), 7)

    if not r_valid and not l_valid:
        return None

    r_dist = math.hypot(r_wrist[0] - cx, r_wrist[1] - cy) if r_valid else float('inf')
    l_dist = math.hypot(l_wrist[0] - cx, l_wrist[1] - cy) if l_valid else float('inf')

    if r_dist < l_dist:
        return (float(r_wrist[0]), float(r_wrist[1]), 4)
    else:
        return (float(l_wrist[0]), float(l_wrist[1]), 7)

def get_racket_heuristic_skeleton(person_kpts: np.ndarray, racket_bbox: Optional[Tuple[float, float, float, float]], dominant_hand: Optional[int] = None) -> Optional[dict]:
    """
    Computes the heuristic racket skeleton.
    Returns a dict with:
      'handle_base': (x, y)
      'center': (x, y)
      'tip': (x, y)
      'head_left': (x, y)
      'head_right': (x, y)
    """
    if racket_bbox is None:
        return None
        
    wrist_info = get_dominant_wrist(person_kpts, racket_bbox, dominant_hand)
    if wrist_info is None:
        return None
    wx, wy, _ = wrist_info

    rx1, ry1, rx2, ry2 = racket_bbox
    cx = (rx1 + rx2) / 2.0
    cy = (ry1 + ry2) / 2.0

    vx = cx - wx
    vy = cy - wy

    if vx == 0 and vy == 0:
        return None

    # Find tip (intersection of ray from wrist through center with bbox)
    t_x = float('inf')
    if vx > 0:
        t_x = (rx2 - wx) / vx
    elif vx < 0:
        t_x = (rx1 - wx) / vx

    t_y = float('inf')
    if vy > 0:
        t_y = (ry2 - wy) / vy
    elif vy < 0:
        t_y = (ry1 - wy) / vy

    t = min(t_x, t_y)
    tx = wx + t * vx
    ty = wy + t * vy

    # Find head width points (perpendicular line passing through center, bounded by bbox)
    length = math.hypot(vx, vy)
    nx = -vy / length
    ny = vx / length

    s_x = float('inf')
    if nx != 0:
        s_x = min(abs((rx2 - cx) / nx), abs((rx1 - cx) / nx))
    s_y = float('inf')
    if ny != 0:
        s_y = min(abs((ry2 - cy) / ny), abs((ry1 - cy) / ny))

    s = min(s_x, s_y)
    hx1 = cx + s * nx
    hy1 = cy + s * ny
    hx2 = cx - s * nx
    hy2 = cy - s * ny

    return {
        'handle_base': (wx, wy),
        'center': (cx, cy),
        'tip': (tx, ty),
        'head_left': (hx1, hy1),
        'head_right': (hx2, hy2)
    }

def render_pose_with_racket(
    person_kpts: np.ndarray, 
    racket_bbox: Optional[Tuple[float, float, float, float]], 
    height: int, 
    width: int,
    dominant_hand: Optional[int] = None
) -> np.ndarray:
    """
    Renders DWPose and the racket heuristic skeleton onto a canvas.
    Note: Coordinates in person_kpts and racket_bbox are assumed to be in the original image coordinate space
    (NOT normalized to [0,1]). If they are relative to the bounding box crop, width and height should be the crop size.
    """
    from src.shared.dwpose_draw import draw_dwpose_canvas
    
    # dwpose_draw expects normalized coordinates for the drawing fallback, or absolute for draw_poses
    # wait, draw_dwpose_canvas expects absolute coordinates if values > 1.5, otherwise assumes normalized.
    # We will pass the array directly. 
    
    # We need to make sure the array has shape [1, 18, 3]
    people_dw = person_kpts[np.newaxis, ...] if person_kpts.ndim == 2 else person_kpts
    
    # We call draw_dwpose_canvas which will draw the body.
    canvas = draw_dwpose_canvas(height=height, width=width, people_dw=people_dw, confidence_threshold=0.2)
    
    # Draw racket
    racket_skel = get_racket_heuristic_skeleton(person_kpts, racket_bbox, dominant_hand)
    if racket_skel is not None:
        wx, wy = int(round(racket_skel['handle_base'][0])), int(round(racket_skel['handle_base'][1]))
        cx, cy = int(round(racket_skel['center'][0])), int(round(racket_skel['center'][1]))
        tx, ty = int(round(racket_skel['tip'][0])), int(round(racket_skel['tip'][1]))
        hx1, hy1 = int(round(racket_skel['head_left'][0])), int(round(racket_skel['head_left'][1]))
        hx2, hy2 = int(round(racket_skel['head_right'][0])), int(round(racket_skel['head_right'][1]))
        
        # Draw racket shaft (wrist to tip)
        cv2.line(canvas, (wx, wy), (tx, ty), (255, 255, 255), 4, cv2.LINE_AA)
        
        # Draw racket head width
        cv2.line(canvas, (hx1, hy1), (hx2, hy2), (200, 200, 200), 3, cv2.LINE_AA)
        
        # Draw points
        cv2.circle(canvas, (wx, wy), 5, (0, 0, 255), -1, cv2.LINE_AA) # wrist (red)
        cv2.circle(canvas, (cx, cy), 5, (0, 255, 0), -1, cv2.LINE_AA) # center (green)
        cv2.circle(canvas, (tx, ty), 5, (255, 0, 0), -1, cv2.LINE_AA) # tip (blue)

    return canvas
