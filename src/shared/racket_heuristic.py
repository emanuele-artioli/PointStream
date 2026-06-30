import math
from typing import Tuple, Optional
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

def get_racket_heuristic_skeleton(person_kpts: np.ndarray, racket_bbox: Optional[Tuple[float, float, float, float]], dominant_hand: Optional[int] = None, racket_mask_points: Optional[dict] = None) -> Optional[dict]:
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

    # If we have the exact points from the mask's convex hull, use them!
    if racket_mask_points is not None:
        p1 = racket_mask_points['p1']
        p2 = racket_mask_points['p2']
        p3 = racket_mask_points['p3']
        p4 = racket_mask_points['p4']
        
        # Which point is the handle? (closest to wrist)
        dist_p1 = math.hypot(wx - p1[0], wy - p1[1])
        dist_p2 = math.hypot(wx - p2[0], wy - p2[1])
        
        if dist_p1 < dist_p2:
            base = p1
            tip = p2
        else:
            base = p2
            tip = p1
            
        # Tie the handle strictly to the wrist to avoid floating disconnected rackets
        offset_x = wx - base[0]
        offset_y = wy - base[1]
        
        return {
            'handle_base': (wx, wy),
            'center': (((base[0] + tip[0]) / 2.0) + offset_x, ((base[1] + tip[1]) / 2.0) + offset_y),
            'tip': (tip[0] + offset_x, tip[1] + offset_y),
            'head_left': (p3[0] + offset_x, p3[1] + offset_y),
            'head_right': (p4[0] + offset_x, p4[1] + offset_y)
        }
    else:
        # Fallback to AABB intersection (can inflate width if racket is diagonal)
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
        nx = -vy / length if length > 0 else 0
        ny = vx / length if length > 0 else 0

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
    dominant_hand: Optional[int] = None,
    racket_mask_points: Optional[dict] = None
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
    racket_skel = get_racket_heuristic_skeleton(person_kpts, racket_bbox, dominant_hand, racket_mask_points)
    if racket_skel is not None:
        wx, wy = int(round(racket_skel['handle_base'][0])), int(round(racket_skel['handle_base'][1]))
        cx, cy = int(round(racket_skel['center'][0])), int(round(racket_skel['center'][1]))
        tx, ty = int(round(racket_skel['tip'][0])), int(round(racket_skel['tip'][1]))
        hx1, hy1 = int(round(racket_skel['head_left'][0])), int(round(racket_skel['head_left'][1]))
        hx2, hy2 = int(round(racket_skel['head_right'][0])), int(round(racket_skel['head_right'][1]))
        
        # Calculate dynamic line thickness based on canvas size
        stick_width = max(1, int(round(min(width, height) / 80)))
        head_width = max(1, int(round(min(width, height) / 100)))
        circle_radius = max(1, int(round(min(width, height) / 90)))
        
        # Draw racket shaft (wrist to tip)
        cv2.line(canvas, (wx, wy), (tx, ty), (255, 255, 255), stick_width, cv2.LINE_AA)
        
        # Draw racket head width
        cv2.line(canvas, (hx1, hy1), (hx2, hy2), (200, 200, 200), head_width, cv2.LINE_AA)
        
        # Draw points
        cv2.circle(canvas, (wx, wy), circle_radius, (0, 0, 255), -1, cv2.LINE_AA) # wrist (red)
        cv2.circle(canvas, (cx, cy), circle_radius, (0, 255, 0), -1, cv2.LINE_AA) # center (green)
        cv2.circle(canvas, (tx, ty), circle_radius, (255, 0, 0), -1, cv2.LINE_AA) # tip (blue)

    return canvas

def interpolate_racket_track(meta_data: list[dict]) -> list[dict]:
    """Interpolate missing racket bounding boxes and mask points over time."""
    valid_indices = [idx for idx, m in enumerate(meta_data) if m.get('racket_bbox_crop') is not None]
    if len(valid_indices) > 0 and len(valid_indices) < len(meta_data):
        for i in range(len(meta_data)):
            if meta_data[i].get('racket_bbox_crop') is None:
                before = [idx for idx in valid_indices if idx < i]
                after = [idx for idx in valid_indices if idx > i]
                
                if not before:
                    meta_data[i]['racket_bbox_crop'] = meta_data[after[0]]['racket_bbox_crop']
                    if meta_data[after[0]].get('racket_mask_points'):
                        meta_data[i]['racket_mask_points'] = meta_data[after[0]]['racket_mask_points']
                elif not after:
                    meta_data[i]['racket_bbox_crop'] = meta_data[before[-1]]['racket_bbox_crop']
                    if meta_data[before[-1]].get('racket_mask_points'):
                        meta_data[i]['racket_mask_points'] = meta_data[before[-1]]['racket_mask_points']
                else:
                    idx_b = before[-1]
                    idx_a = after[0]
                    bbox_b = meta_data[idx_b]['racket_bbox_crop']
                    bbox_a = meta_data[idx_a]['racket_bbox_crop']
                    weight = (i - idx_b) / float(idx_a - idx_b)
                    meta_data[i]['racket_bbox_crop'] = [
                        bbox_b[j] + weight * (bbox_a[j] - bbox_b[j]) for j in range(4)
                    ]
                    
                    pts_b = meta_data[idx_b].get('racket_mask_points')
                    pts_a = meta_data[idx_a].get('racket_mask_points')
                    if pts_b and pts_a:
                        interp_pts = {}
                        for k in ['p1', 'p2', 'p3', 'p4']:
                            interp_pts[k] = (
                                pts_b[k][0] + weight * (pts_a[k][0] - pts_b[k][0]),
                                pts_b[k][1] + weight * (pts_a[k][1] - pts_b[k][1])
                            )
                        meta_data[i]['racket_mask_points'] = interp_pts
    return meta_data
