#!/usr/bin/env python3
"""
Advanced Recursive Stitching Component (Top-Down Strategy)

This component implements a top-down, divide-and-conquer stitching strategy.
It attempts to find "shortcuts" by stitching the first and last frames of a
sequence. If successful, it skips processing intermediate frames, offering a
significant speed advantage. If it fails, it recursively splits the sequence.

Features:
- Top-down recursive logic with shortcut detection for speed.
- Final "gap-filling" step to calculate homographies for skipped frames.
- Configurable feature detector: fast ORB (default) or accurate SIFT.
- High-quality, distance-weighted blending to avoid visible seams.
- Homography quality control using an inlier ratio threshold.
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple

# Import decorators and config
from .decorators import track_performance
from .error_handling import safe_execute
from . import config


class Stitcher:
    """Advanced recursive stitching component for scene panorama creation."""

    def __init__(self):
        """Initialize the stitcher with configuration parameters."""
        self.feature_detector_name = config.get_str('stitching', 'feature_detector', 'orb').lower()
        self.sift_nfeatures = config.get_int('stitching', 'sift_nfeatures', 500)
        self.orb_nfeatures = config.get_int('stitching', 'orb_nfeatures', 2000)
        self.min_match_count = config.get_int('stitching', 'min_match_count', 10)
        self.ransac_reproj_threshold = config.get_float('stitching', 'ransac_reproj_threshold', 4.0)
        self.min_inlier_ratio = config.get_float('stitching', 'min_inlier_ratio', 0.5)
        self.static_homography_threshold = config.get_float('stitching', 'static_homography_threshold', 0.01)

        if self.feature_detector_name == 'sift':
            logging.info("Using SIFT detector (high accuracy, slower)")
            self.detector = cv2.SIFT_create(nfeatures=self.sift_nfeatures)
            self.bf_matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        else:
            logging.info("Using ORB detector (high speed, good accuracy)")
            self.detector = cv2.ORB_create(nfeatures=self.orb_nfeatures)
            self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    @track_performance
    def stitch_scene(self, frames: List[np.ndarray], masks: Optional[List[np.ndarray]] = None) -> Dict[str, Any]:
        """Main stitching function implementing the top-down strategy."""
        if not frames or len(frames) == 0:
            return {'scene_type': 'Complex', 'panorama': None, 'homographies': []}
        if len(frames) == 1:
            return {'scene_type': 'Static', 'panorama': frames[0], 'homographies': [np.eye(3)]}

        # Store masks for use throughout stitching
        self.masks = masks or [None] * len(frames)
        
        logging.info(f"Starting top-down stitching for {len(frames)} frames")

        if self._check_static_scene(frames)['is_static']:
            logging.info("Scene classified as Static")
            return {'scene_type': 'Static', 'panorama': frames[0], 'homographies': [np.eye(3)] * len(frames)}

        try:
            # STAGE 1: Recursively build panorama and get sparse homographies
            panorama, sparse_homographies = self._build_panorama_top_down(frames)

            if panorama is None:
                logging.warning("Top-down stitching failed, classifying scene as Complex.")
                return {'scene_type': 'Complex', 'panorama': None, 'homographies': []}

            # STAGE 2: Fill in the homographies for frames that were skipped
            final_homographies = self._fill_skipped_homographies(frames, panorama, sparse_homographies)
            
            logging.info(f"Scene classified as Simple - Final panorama: {panorama.shape}")
            return {'scene_type': 'Simple', 'panorama': panorama, 'homographies': final_homographies}

        except Exception as e:
            logging.error(f"Stitching process failed with exception: {e}", exc_info=True)
            return {'scene_type': 'Complex', 'panorama': None, 'homographies': []}

    def _build_panorama_top_down(self, frames: List[np.ndarray]) -> Tuple[Optional[np.ndarray], Optional[List[np.ndarray]]]:
        """
        Recursively builds a panorama using a top-down approach.
        Tries to stitch the first and last frames as a shortcut.
        """
        num_frames = len(frames)
        if num_frames == 0:
            return None, None
        if num_frames == 1:
            return frames[0], [np.eye(3, dtype=np.float32)]

        # SHORTCUT ATTEMPT: Try to stitch the first and last frames of the current segment.
        if num_frames > 2:
            homography_result = self._find_homography_between_frames(frames[0], frames[-1])
            if homography_result['success']:
                logging.debug(f"Shortcut found for a segment of {num_frames} frames.")
                H = homography_result['homography']
                stitched_pano = self._expand_panorama_with_frame(frames[0], frames[-1], H, len(frames) - 1)
                # Mark intermediate frames as skipped (None)
                homographies = [np.eye(3, dtype=np.float32)] + [None] * (num_frames - 2) + [H]
                return stitched_pano, homographies

        # RECURSIVE STEP: If shortcut fails or segment is too small, split and recurse.
        mid_point = num_frames // 2
        left_pano, left_h = self._build_panorama_top_down(frames[:mid_point])
        right_pano, right_h = self._build_panorama_top_down(frames[mid_point:])

        if left_pano is None or right_pano is None:
            return None, None

        # MERGE STEP: Stitch the two resulting sub-panoramas.
        merge_homography_result = self._find_homography_between_frames(left_pano, right_pano)
        if not merge_homography_result['success']:
            return None, None
        
        H_merge = merge_homography_result['homography']
        merged_panorama = self._expand_panorama_with_frame(left_pano, right_pano, H_merge, 0)

        # Chain the homographies
        updated_right_h = [H_merge @ h if h is not None else None for h in right_h]
        final_homographies = left_h + updated_right_h

        return merged_panorama, final_homographies

    def _fill_skipped_homographies(self, frames: List[np.ndarray], panorama: np.ndarray, sparse_homographies: List[np.ndarray]) -> List[np.ndarray]:
        """Calculates homographies for frames that were skipped during the recursive process."""
        final_homographies = list(sparse_homographies)
        for i, h in enumerate(final_homographies):
            if h is None:
                # This frame was skipped, so calculate its homography against the final panorama
                result = self._find_homography_between_frames(panorama, frames[i])
                if result['success']:
                    final_homographies[i] = result['homography']
                else:
                    # If it fails, use an identity matrix as a fallback
                    final_homographies[i] = np.eye(3, dtype=np.float32)
        return final_homographies

    def _find_homography_between_frames(self, frame1: np.ndarray, frame2: np.ndarray) -> Dict[str, Any]:
        """Find homography and validate its quality using the configured detector."""
        try:
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY) if len(frame1.shape) == 3 else frame1
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY) if len(frame2.shape) == 3 else frame2

            scale_factor = 0.5
            small_gray1 = cv2.resize(gray1, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
            small_gray2 = cv2.resize(gray2, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)

            kp1, desc1 = self.detector.detectAndCompute(small_gray1, None)
            kp2, desc2 = self.detector.detectAndCompute(small_gray2, None)

            if desc1 is None or desc2 is None or len(kp1) < self.min_match_count:
                return {'success': False, 'reason': 'insufficient_features'}

            for kp in kp1: kp.pt = (kp.pt[0] / scale_factor, kp.pt[1] / scale_factor)
            for kp in kp2: kp.pt = (kp.pt[0] / scale_factor, kp.pt[1] / scale_factor)

            matches = self.bf_matcher.match(desc1, desc2)

            if len(matches) < self.min_match_count:
                return {'success': False, 'reason': 'insufficient_matches'}

            src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, self.ransac_reproj_threshold)

            if H is None:
                return {'success': False, 'reason': 'ransac_failed'}

            inlier_count = np.sum(mask)
            inlier_ratio = inlier_count / len(matches)
            if inlier_ratio < self.min_inlier_ratio:
                return {'success': False, 'reason': f'inlier_ratio_too_low ({inlier_ratio:.2%})'}

            return {'success': True, 'homography': H}

        except Exception as e:
            logging.debug(f"Homography calculation failed: {e}")
            return {'success': False, 'reason': 'calculation_error'}

    def _expand_panorama_with_frame(self, panorama: np.ndarray, frame: np.ndarray, H: np.ndarray, frame_idx: int = 0) -> Optional[np.ndarray]:
        """Expand panorama by adding a new frame with proper size calculation and object-aware blending."""
        try:
            h_pano, w_pano = panorama.shape[:2]
            h_frame, w_frame = frame.shape[:2]

            frame_corners = np.float32([[0, 0], [w_frame, 0], [w_frame, h_frame], [0, h_frame]]).reshape(-1, 1, 2)
            warped_corners = cv2.perspectiveTransform(frame_corners, H)
            pano_corners = np.float32([[0, 0], [w_pano, 0], [w_pano, h_pano], [0, h_pano]]).reshape(-1, 1, 2)
            all_corners = np.concatenate([pano_corners, warped_corners], axis=0)

            x_min, y_min = np.int32(all_corners.min(axis=0).ravel() - 0.5)
            x_max, y_max = np.int32(all_corners.max(axis=0).ravel() + 0.5)
            
            tx, ty = -x_min, -y_min
            T_pano = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float32)
            H_combined = T_pano @ H

            new_width, new_height = x_max - x_min, y_max - y_min

            warped_panorama = cv2.warpPerspective(panorama, T_pano, (new_width, new_height))
            warped_frame = cv2.warpPerspective(frame, H_combined, (new_width, new_height))

            mask_pano = (np.sum(warped_panorama, axis=2) > 0).astype(np.float32)
            mask_frame = (np.sum(warped_frame, axis=2) > 0).astype(np.float32)
            
            # Apply object mask to avoid blending objects
            if hasattr(self, 'masks') and self.masks and frame_idx < len(self.masks) and self.masks[frame_idx] is not None:
                # Warp the object mask
                object_mask = self.masks[frame_idx]
                warped_object_mask = cv2.warpPerspective(object_mask, H_combined, (new_width, new_height))
                
                # Convert to binary: objects are 0 (black), background is 255 (white)
                object_areas = (warped_object_mask == 0).astype(np.float32)
                background_areas = (warped_object_mask == 255).astype(np.float32)
                
                # In object areas: use frame data directly (no blending)
                # In background areas: use distance-weighted blending
                
                # Calculate distance weights only for background areas
                pano_dist = cv2.distanceTransform(mask_pano.astype(np.uint8), cv2.DIST_L2, 5) * background_areas
                frame_dist = cv2.distanceTransform(mask_frame.astype(np.uint8), cv2.DIST_L2, 5) * background_areas
                
                # Normalize distances to prevent overflow
                pano_dist = pano_dist / np.max(pano_dist) if np.max(pano_dist) > 0 else pano_dist
                frame_dist = frame_dist / np.max(frame_dist) if np.max(frame_dist) > 0 else frame_dist
                total_dist = pano_dist + frame_dist
                
                # Avoid division by zero
                total_dist[total_dist == 0] = 1.0
                weight_pano = pano_dist / total_dist
                weight_frame = frame_dist / total_dist
                
                # Blend background areas with distance weighting
                blended_bg = (warped_panorama.astype(np.float32) * cv2.merge([weight_pano]*3) +
                              warped_frame.astype(np.float32) * cv2.merge([weight_frame]*3))
                
                # In object areas, prefer the new frame (clearer objects)
                result = blended_bg.copy()
                object_mask_3d = cv2.merge([object_areas]*3)
                result = result * (1 - object_mask_3d) + warped_frame.astype(np.float32) * object_mask_3d
                
                return result.astype(np.uint8)
            
            else:
                # Original distance-weighted blending when no masks available
                pano_dist = cv2.distanceTransform(mask_pano.astype(np.uint8), cv2.DIST_L2, 5)
                frame_dist = cv2.distanceTransform(mask_frame.astype(np.uint8), cv2.DIST_L2, 5)

                # Normalize distances to prevent overflow
                pano_dist = pano_dist / np.max(pano_dist) if np.max(pano_dist) > 0 else pano_dist
                frame_dist = frame_dist / np.max(frame_dist) if np.max(frame_dist) > 0 else frame_dist
                total_dist = pano_dist + frame_dist
                total_dist[total_dist == 0] = 1.0

                weight_pano = pano_dist / total_dist
                weight_frame = frame_dist / total_dist

                blended = (warped_panorama.astype(np.float32) * cv2.merge([weight_pano]*3) +
                           warped_frame.astype(np.float32) * cv2.merge([weight_frame]*3))

                return blended.astype(np.uint8)

        except Exception as e:
            logging.debug(f"Panorama expansion failed: {e}")
            return None

    def _check_static_scene(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """Check if scene is static by analyzing homography between first and last frames."""
        try:
            homography_result = self._find_homography_between_frames(frames[0], frames[-1])
            if not homography_result['success']:
                return {'is_static': False}
            
            H = homography_result['homography']
            identity = np.eye(3, dtype=np.float32)
            diff = np.max(np.abs(H - identity))
            
            return {'is_static': diff < self.static_homography_threshold}
        except Exception:
            return {'is_static': False}
