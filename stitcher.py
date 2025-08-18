#!/usr/bin/env python3
"""
Advanced Recursive Stitching Component

This component implements a divide-and-conquer stitching strategy using native OpenCV functions.
It classifies scenes as Static, Simple, or Complex based on their stitching characteristics.

Features:
- Static scene detection using homography analysis
- Recursive stitching with keyframe identification
- Complex scene detection for high-motion content
- Full frame alignment calculation for skipped frames
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple
from decorators import log_step, time_step
import config


class Stitcher:
    """Advanced recursive stitching component for scene panorama creation."""
    
    def __init__(self):
        """Initialize the stitcher with configuration parameters."""
        # Feature detection parameters
        self.sift_nfeatures = config.get_int('stitching', 'sift_nfeatures', 0)  # 0 = unlimited
        self.sift_octave_layers = config.get_int('stitching', 'sift_octave_layers', 3)
        self.sift_contrast_threshold = config.get_float('stitching', 'sift_contrast_threshold', 0.04)
        self.sift_edge_threshold = config.get_float('stitching', 'sift_edge_threshold', 10)
        self.sift_sigma = config.get_float('stitching', 'sift_sigma', 1.6)
        
        # Matching parameters
        self.bf_cross_check = config.get_bool('stitching', 'bf_cross_check', True)
        self.lowe_ratio = config.get_float('stitching', 'lowe_ratio', 0.7)
        self.min_match_count = config.get_int('stitching', 'min_match_count', 10)
        
        # Homography parameters
        self.ransac_reproj_threshold = config.get_float('stitching', 'ransac_reproj_threshold', 5.0)
        self.ransac_max_iters = config.get_int('stitching', 'ransac_max_iters', 2000)
        self.ransac_confidence = config.get_float('stitching', 'ransac_confidence', 0.99)
        
        # Static scene detection parameters
        self.static_homography_threshold = config.get_float('stitching', 'static_homography_threshold', 0.01)
        
        # Initialize SIFT detector
        self.sift = cv2.SIFT_create(
            nfeatures=self.sift_nfeatures,
            nOctaveLayers=self.sift_octave_layers,
            contrastThreshold=self.sift_contrast_threshold,
            edgeThreshold=self.sift_edge_threshold,
            sigma=self.sift_sigma
        )
        
        # Initialize BF matcher
        self.bf_matcher = cv2.BFMatcher(crossCheck=self.bf_cross_check)
        
        logging.info("Stitcher initialized with recursive divide-and-conquer strategy")
        logging.info(f"SIFT features: {self.sift_nfeatures} (0=unlimited)")
        logging.info(f"Minimum matches required: {self.min_match_count}")
        logging.info(f"RANSAC reprojection threshold: {self.ransac_reproj_threshold}")
        logging.info(f"Static scene threshold: {self.static_homography_threshold}")
    
    @log_step
    @time_step(track_processing=True)
    def stitch_scene(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """
        Main stitching function implementing progressive panorama accumulation.
        
        Args:
            frames: List of scene frames to stitch
            
        Returns:
            Dictionary containing:
            - scene_type: "Static", "Simple", or "Complex"
            - panorama: The final panorama image (or first frame for static)
            - homographies: List of homography matrices for each frame
        """
        if not frames or len(frames) == 0:
            return {
                'scene_type': 'Complex',
                'panorama': None,
                'homographies': []
            }
        
        if len(frames) == 1:
            # Single frame - always static
            return {
                'scene_type': 'Static',
                'panorama': frames[0],
                'homographies': [np.eye(3, dtype=np.float32)]
            }
        
        logging.info(f"Starting progressive panorama stitching for {len(frames)} frames")
        
        # STEP 1: Static Scene Check
        static_result = self._check_static_scene(frames)
        if static_result['is_static']:
            logging.info("Scene classified as Static")
            return {
                'scene_type': 'Static',
                'panorama': frames[0],  # Use first frame as panorama
                'homographies': [np.eye(3, dtype=np.float32) for _ in frames]
            }
        
        # STEP 2: Progressive Panorama Building
        try:
            stitching_result = self._build_progressive_panorama(frames)
            
            if stitching_result['success']:
                # Successful stitching - scene is Simple
                logging.info(f"Scene classified as Simple - Final panorama: {stitching_result['panorama'].shape}")
                
                return {
                    'scene_type': 'Simple',
                    'panorama': stitching_result['panorama'],
                    'homographies': stitching_result['homographies']
                }
            else:
                # Stitching failed - scene is Complex
                logging.info("Scene classified as Complex")
                return {
                    'scene_type': 'Complex',
                    'panorama': None,
                    'homographies': []
                }
                
        except Exception as e:
            logging.error(f"Stitching process failed: {e}")
            return {
                'scene_type': 'Complex',
                'panorama': None,
                'homographies': []
            }
    
    def _check_static_scene(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """
        Check if scene is static by analyzing homography between first and last frames.
        
        Args:
            frames: List of scene frames
            
        Returns:
            Dictionary with 'is_static' boolean and analysis details
        """
        first_frame = frames[0]
        last_frame = frames[-1]
        
        try:
            # Find homography between first and last frames
            homography_result = self._find_homography_between_frames(first_frame, last_frame)
            
            if not homography_result['success']:
                # Can't find homography - likely not static
                return {'is_static': False, 'reason': 'no_homography_found'}
            
            H = homography_result['homography']
            
            # Check if homography is close to identity matrix
            identity = np.eye(3, dtype=np.float32)
            diff = np.abs(H - identity)
            max_diff = np.max(diff)
            
            is_static = max_diff < self.static_homography_threshold
            
            logging.debug(f"Static check: max difference from identity = {max_diff:.6f}")
            logging.debug(f"Static threshold: {self.static_homography_threshold}")
            
            return {
                'is_static': is_static,
                'max_difference': max_diff,
                'homography': H,
                'threshold': self.static_homography_threshold
            }
            
        except Exception as e:
            logging.debug(f"Static scene check failed: {e}")
            return {'is_static': False, 'reason': 'check_failed'}
    
    def _build_progressive_panorama(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """
        Build panorama progressively by adding frames one by one.
        This creates proper wide panoramas by accumulating all camera movement.
        
        Args:
            frames: List of scene frames to stitch
            
        Returns:
            Dictionary with success flag, panorama, and homographies
        """
        # Start with first frame as initial panorama
        current_panorama = frames[0].copy()
        homographies = [np.eye(3, dtype=np.float32)]  # First frame is identity
        
        # Track cumulative transformations
        successful_stitches = 0
        failed_stitches = 0
        max_failures = max(3, len(frames) // 5)  # Allow some failures
        
        logging.info(f"Building progressive panorama from {len(frames)} frames")
        
        # Progressive stitching: add each frame to the growing panorama
        for i in range(1, len(frames)):
            frame = frames[i]
            
            # Find homography between current frame and current panorama
            homography_result = self._find_homography_between_frames(current_panorama, frame)
            
            if homography_result['success']:
                H = homography_result['homography']
                
                # Expand panorama by warping and blending the new frame
                new_panorama = self._expand_panorama_with_frame(current_panorama, frame, H)
                
                if new_panorama is not None:
                    current_panorama = new_panorama
                    homographies.append(H)
                    successful_stitches += 1
                    
                    if i % 10 == 0:  # Log progress every 10 frames
                        logging.info(f"Progress: {i}/{len(frames)} frames, panorama size: {current_panorama.shape[:2]}")
                else:
                    # Failed to expand panorama
                    homographies.append(np.eye(3, dtype=np.float32))
                    failed_stitches += 1
            else:
                # Failed to find homography
                homographies.append(np.eye(3, dtype=np.float32))
                failed_stitches += 1
            
            # Check if too many failures
            if failed_stitches > max_failures:
                logging.warning(f"Too many stitching failures ({failed_stitches}), classifying as Complex")
                return {'success': False, 'reason': 'too_many_failures'}
        
        success_rate = successful_stitches / (len(frames) - 1) if len(frames) > 1 else 1.0
        
        logging.info(f"Progressive stitching complete: {successful_stitches}/{len(frames)-1} successful")
        logging.info(f"Final panorama size: {current_panorama.shape}")
        logging.info(f"Success rate: {success_rate:.1%}")
        
        # Consider successful if we got at least 60% of frames
        if success_rate >= 0.6:
            return {
                'success': True,
                'panorama': current_panorama,
                'homographies': homographies,
                'success_rate': success_rate
            }
        else:
            return {'success': False, 'reason': 'low_success_rate', 'success_rate': success_rate}
    
    def _expand_panorama_with_frame(self, panorama: np.ndarray, frame: np.ndarray, H: np.ndarray) -> Optional[np.ndarray]:
        """
        Expand panorama by adding a new frame with proper size calculation.
        
        Args:
            panorama: Current panorama
            frame: New frame to add
            H: Homography matrix (frame -> panorama coordinates)
            
        Returns:
            Expanded panorama or None if expansion fails
        """
        try:
            h_pano, w_pano = panorama.shape[:2]
            h_frame, w_frame = frame.shape[:2]
            
            # Find where the new frame will be placed in panorama coordinates
            frame_corners = np.float32([[0, 0], [w_frame, 0], [w_frame, h_frame], [0, h_frame]]).reshape(-1, 1, 2)
            warped_corners = cv2.perspectiveTransform(frame_corners, H)
            
            # Combine with existing panorama bounds
            pano_corners = np.float32([[0, 0], [w_pano, 0], [w_pano, h_pano], [0, h_pano]]).reshape(-1, 1, 2)
            all_corners = np.concatenate([pano_corners, warped_corners], axis=0)
            
            # Calculate new bounding box
            x_min = int(np.floor(np.min(all_corners[:, 0, 0])))
            x_max = int(np.ceil(np.max(all_corners[:, 0, 0])))
            y_min = int(np.floor(np.min(all_corners[:, 0, 1])))
            y_max = int(np.ceil(np.max(all_corners[:, 0, 1])))
            
            # Calculate translation needed to keep all content positive
            tx = -x_min if x_min < 0 else 0
            ty = -y_min if y_min < 0 else 0
            
            # New panorama size
            new_width = x_max - x_min
            new_height = y_max - y_min
            
            # Create translation matrix for panorama
            T_pano = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float32)
            
            # Create combined transformation for new frame
            H_combined = T_pano @ H
            
            # Warp the current panorama to new coordinate system
            warped_panorama = cv2.warpPerspective(panorama, T_pano, (new_width, new_height))
            
            # Warp the new frame to panorama coordinate system
            warped_frame = cv2.warpPerspective(frame, H_combined, (new_width, new_height))
            
            # Create masks for blending
            mask_pano = (warped_panorama.sum(axis=2) > 0).astype(np.float32)
            mask_frame = (warped_frame.sum(axis=2) > 0).astype(np.float32)
            overlap_mask = mask_pano * mask_frame
            
            # Create blended result
            result = np.zeros_like(warped_panorama)
            
            # Areas with only panorama
            pano_only = (mask_pano > 0) & (mask_frame == 0)
            result[pano_only] = warped_panorama[pano_only]
            
            # Areas with only new frame
            frame_only = (mask_pano == 0) & (mask_frame > 0)
            result[frame_only] = warped_frame[frame_only]
            
            # Overlap areas - blend smoothly
            if np.sum(overlap_mask) > 0:
                overlap_indices = overlap_mask > 0
                
                # Distance-based blending weights
                # Create distance transform for smooth blending
                pano_dist = cv2.distanceTransform((mask_pano > 0).astype(np.uint8), cv2.DIST_L2, 5)
                frame_dist = cv2.distanceTransform((mask_frame > 0).astype(np.uint8), cv2.DIST_L2, 5)
                
                total_dist = pano_dist + frame_dist
                safe_total = np.maximum(total_dist, 1e-6)  # Avoid division by zero
                
                pano_weight = pano_dist / safe_total
                frame_weight = frame_dist / safe_total
                
                # Apply blending
                result[overlap_indices] = (
                    warped_panorama[overlap_indices].astype(np.float32) * pano_weight[overlap_indices, np.newaxis] +
                    warped_frame[overlap_indices].astype(np.float32) * frame_weight[overlap_indices, np.newaxis]
                ).astype(np.uint8)
            
            return result
            
        except Exception as e:
            logging.debug(f"Panorama expansion failed: {e}")
            return None
    
    def _find_homography_between_frames(self, frame1: np.ndarray, frame2: np.ndarray) -> Dict[str, Any]:
        """
        Find homography between two frames using SIFT and RANSAC.
        
        Args:
            frame1: First frame (reference)
            frame2: Second frame (to be warped)
            
        Returns:
            Dictionary with success flag and homography matrix
        """
        try:
            # Convert to grayscale if needed
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY) if len(frame1.shape) == 3 else frame1
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY) if len(frame2.shape) == 3 else frame2
            
            # Detect SIFT features
            kp1, desc1 = self.sift.detectAndCompute(gray1, None)
            kp2, desc2 = self.sift.detectAndCompute(gray2, None)
            
            if desc1 is None or desc2 is None or len(desc1) < 10 or len(desc2) < 10:
                return {'success': False, 'reason': 'insufficient_features'}
            
            # Match features
            matches = self.bf_matcher.match(desc1, desc2)
            
            # Apply Lowe's ratio test if using KNN matching
            if not self.bf_cross_check:
                matches = self.bf_matcher.knnMatch(desc1, desc2, k=2)
                good_matches = []
                for match_pair in matches:
                    if len(match_pair) == 2:
                        m, n = match_pair
                        if m.distance < self.lowe_ratio * n.distance:
                            good_matches.append(m)
                matches = good_matches
            
            if len(matches) < self.min_match_count:
                return {'success': False, 'reason': 'insufficient_matches', 'match_count': len(matches)}
            
            # Extract matched points
            src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            
            # Find homography using RANSAC
            H, mask = cv2.findHomography(
                dst_pts, src_pts,
                cv2.RANSAC,
                self.ransac_reproj_threshold,
                maxIters=self.ransac_max_iters,
                confidence=self.ransac_confidence
            )
            
            if H is None:
                return {'success': False, 'reason': 'ransac_failed'}
            
            # Count inliers
            inlier_count = np.sum(mask)
            inlier_ratio = inlier_count / len(matches)
            
            logging.debug(f"Homography found: {len(matches)} matches, {inlier_count} inliers ({inlier_ratio:.2%})")
            
            return {
                'success': True,
                'homography': H,
                'matches': len(matches),
                'inliers': inlier_count,
                'inlier_ratio': inlier_ratio
            }
            
        except Exception as e:
            logging.debug(f"Homography calculation failed: {e}")
            return {'success': False, 'reason': 'calculation_error'}
    
    def _warp_and_blend_frames(self, frame1: np.ndarray, frame2: np.ndarray, H: np.ndarray) -> np.ndarray:
        """
        Warp frame2 to frame1's coordinate system and blend them.
        
        Args:
            frame1: Reference frame
            frame2: Frame to warp
            H: Homography matrix (frame2 -> frame1)
            
        Returns:
            Blended panorama image
        """
        h1, w1 = frame1.shape[:2]
        h2, w2 = frame2.shape[:2]
        
        # Find the bounding box of the warped image
        corners2 = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
        warped_corners = cv2.perspectiveTransform(corners2, H)
        
        # Combine with frame1 corners
        corners1 = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)
        all_corners = np.concatenate([corners1, warped_corners], axis=0)
        
        # Find bounding rectangle
        x_min = int(np.min(all_corners[:, 0, 0]))
        x_max = int(np.max(all_corners[:, 0, 0]))
        y_min = int(np.min(all_corners[:, 0, 1]))
        y_max = int(np.max(all_corners[:, 0, 1]))
        
        # Calculate translation to keep all content
        translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float32)
        H_translated = translation @ H
        
        # Calculate output size
        output_width = x_max - x_min
        output_height = y_max - y_min
        
        # Warp frame2
        warped_frame2 = cv2.warpPerspective(frame2, H_translated, (output_width, output_height))
        
        # Warp frame1 (just translate)
        frame1_translated = cv2.warpPerspective(frame1, translation, (output_width, output_height))
        
        # Simple blending: use average where both images have content
        mask1 = (frame1_translated.sum(axis=2) > 0).astype(np.float32)
        mask2 = (warped_frame2.sum(axis=2) > 0).astype(np.float32)
        overlap_mask = mask1 * mask2
        
        # Blend in overlap regions, use individual images elsewhere
        panorama = np.zeros_like(frame1_translated)
        
        # Non-overlap regions
        panorama[mask1 > 0] = frame1_translated[mask1 > 0]
        panorama[mask2 > 0] = warped_frame2[mask2 > 0]
        
        # Overlap regions (average)
        if np.sum(overlap_mask) > 0:
            overlap_indices = overlap_mask > 0
            panorama[overlap_indices] = (
                frame1_translated[overlap_indices].astype(np.float32) * 0.5 +
                warped_frame2[overlap_indices].astype(np.float32) * 0.5
            ).astype(np.uint8)
        
        return panorama
    
    def _align_skipped_frames(self, frames: List[np.ndarray], keyframe_indices: List[int], 
                             keyframe_homographies: Dict[int, np.ndarray], 
                             panorama: np.ndarray) -> List[np.ndarray]:
        """
        Calculate homographies for all frames not used as keyframes.
        
        Args:
            frames: All scene frames
            keyframe_indices: Indices of frames used as keyframes
            keyframe_homographies: Homographies for keyframes
            panorama: Final panorama image
            
        Returns:
            List of homography matrices for all frames
        """
        all_homographies = [None] * len(frames)
        
        # Set homographies for keyframes
        for idx in keyframe_indices:
            if idx in keyframe_homographies:
                all_homographies[idx] = keyframe_homographies[idx]
        
        # Calculate homographies for skipped frames
        for i, frame in enumerate(frames):
            if i not in keyframe_indices:
                # Find homography between this frame and the panorama
                homography_result = self._find_homography_to_panorama(frame, panorama)
                
                if homography_result['success']:
                    all_homographies[i] = homography_result['homography']
                else:
                    # Fallback: interpolate from nearest keyframes
                    all_homographies[i] = self._interpolate_homography(i, keyframe_indices, keyframe_homographies)
        
        # Fill any remaining None values with identity matrices
        for i in range(len(all_homographies)):
            if all_homographies[i] is None:
                all_homographies[i] = np.eye(3, dtype=np.float32)
        
        return all_homographies
    
    def _find_homography_to_panorama(self, frame: np.ndarray, panorama: np.ndarray) -> Dict[str, Any]:
        """
        Find homography between a frame and the final panorama.
        
        Args:
            frame: Frame to align
            panorama: Reference panorama
            
        Returns:
            Dictionary with success flag and homography
        """
        try:
            # Use the same method as frame-to-frame homography
            return self._find_homography_between_frames(panorama, frame)
        except Exception as e:
            logging.debug(f"Frame-to-panorama homography failed: {e}")
            return {'success': False, 'reason': 'calculation_error'}
    
    def _interpolate_homography(self, frame_idx: int, keyframe_indices: List[int], 
                               keyframe_homographies: Dict[int, np.ndarray]) -> np.ndarray:
        """
        Interpolate homography for a skipped frame based on nearest keyframes.
        
        Args:
            frame_idx: Index of frame to interpolate
            keyframe_indices: Indices of keyframes
            keyframe_homographies: Homographies for keyframes
            
        Returns:
            Interpolated homography matrix
        """
        if not keyframe_indices:
            return np.eye(3, dtype=np.float32)
        
        # Find nearest keyframes
        sorted_keyframes = sorted(keyframe_indices)
        
        # Find the keyframes before and after this frame
        before_keyframe = None
        after_keyframe = None
        
        for kf_idx in sorted_keyframes:
            if kf_idx < frame_idx:
                before_keyframe = kf_idx
            elif kf_idx > frame_idx and after_keyframe is None:
                after_keyframe = kf_idx
                break
        
        # Simple interpolation strategy
        if before_keyframe is not None and after_keyframe is not None:
            # Interpolate between two keyframes
            weight = (frame_idx - before_keyframe) / (after_keyframe - before_keyframe)
            H_before = keyframe_homographies.get(before_keyframe, np.eye(3, dtype=np.float32))
            H_after = keyframe_homographies.get(after_keyframe, np.eye(3, dtype=np.float32))
            
            # Simple linear interpolation of homography elements
            H_interpolated = (1 - weight) * H_before + weight * H_after
            return H_interpolated
        
        elif before_keyframe is not None:
            # Use the previous keyframe
            return keyframe_homographies.get(before_keyframe, np.eye(3, dtype=np.float32))
        
        elif after_keyframe is not None:
            # Use the next keyframe
            return keyframe_homographies.get(after_keyframe, np.eye(3, dtype=np.float32))
        
        else:
            # No keyframes found - return identity
            return np.eye(3, dtype=np.float32)