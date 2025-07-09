# pointstream/pipelines/server.py
#
# Optimized version using a CPU-only OpenCV and a GPU-native RAFT model
# from torchvision for high-performance optical flow calculation.

import cv2
import numpy as np
import torch
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from torchvision.transforms.functional import to_tensor
from typing import List, Dict, Any, Optional

from pointstream.core.scene import Scene, Frame, DetectedObject
from pointstream.core.tracker import SimpleTracker


class StreamProcessor:
    """
    Handles real-time analysis using a chunk-based binary search on downsampled frames.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.scene_threshold = config.get('scene_threshold', 0.8)
        self.min_scene_duration = config.get('min_scene_duration', 10)
        self.buffer_size = config.get('buffer_size', 90)
        self.frame_skip = config.get('frame_skip', 1)
        self.proc_resolution = config.get('processing_resolution', (640, 360))

        self.frame_buffer: List[Frame] = []
        self.scene_id_counter = 0
        self.frame_counter = 0

    def _calculate_hist(self, frame_bgr: np.ndarray) -> np.ndarray:
        hist_size = [16, 16, 16]
        hist_ranges = [0, 256, 0, 256, 0, 256]
        hist = cv2.calcHist([frame_bgr], [0, 1, 2], None, hist_size, hist_ranges, accumulate=False)
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        return hist

    def _are_frames_similar(self, hist1: np.ndarray, hist2: np.ndarray) -> bool:
        correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        return correlation >= self.scene_threshold

    def _find_cut_in_buffer(self, buffer: List[Frame]) -> Optional[int]:
        if len(buffer) < 2: return None
        low, high, cut_point = 0, len(buffer) - 1, None
        start_hist = self._calculate_hist(buffer[low].image)
        end_hist = self._calculate_hist(buffer[high].image)
        if self._are_frames_similar(start_hist, end_hist): return None

        while low <= high:
            mid = (low + high) // 2
            if mid == 0: break
            mid_hist = self._calculate_hist(buffer[mid].image)
            prev_mid_hist = self._calculate_hist(buffer[mid-1].image)
            if not self._are_frames_similar(mid_hist, prev_mid_hist):
                cut_point, high = mid, mid - 1
            else:
                low = mid + 1
        return cut_point

    def process_frame(self, frame_num: int, frame_bgr: np.ndarray) -> List[Scene]:
        self.frame_counter += 1
        if self.frame_counter % self.frame_skip != 0:
            return []

        resized_frame = cv2.resize(frame_bgr, self.proc_resolution, interpolation=cv2.INTER_AREA)
        self.frame_buffer.append(Frame(frame_number=frame_num, image=resized_frame))

        if len(self.frame_buffer) < self.buffer_size:
            return []
        return self._analyze_buffer()

    def _analyze_buffer(self, is_flush: bool = False) -> List[Scene]:
        if not self.frame_buffer: return []
        scenes_found, cut_index = [], self._find_cut_in_buffer(self.frame_buffer)

        if cut_index is None:
            if is_flush and len(self.frame_buffer) >= self.min_scene_duration:
                scenes_found.append(self._create_scene_from_frames(self.frame_buffer))
                self.frame_buffer.clear()
        else:
            scene_frames = self.frame_buffer[:cut_index]
            if len(scene_frames) >= self.min_scene_duration:
                scenes_found.append(self._create_scene_from_frames(scene_frames))
            self.frame_buffer = self.frame_buffer[cut_index:]
        return scenes_found

    def _create_scene_from_frames(self, frames: List[Frame]) -> Scene:
        scene = Scene(scene_id=self.scene_id_counter, start_frame=frames[0].frame_number, end_frame=frames[-1].frame_number)
        scene.frames = list(frames)
        self.scene_id_counter += 1
        return scene

    def flush(self) -> List[Scene]:
        all_scenes = []
        while self.frame_buffer:
            scenes = self._analyze_buffer(is_flush=True)
            if not scenes:
                if len(self.frame_buffer) >= self.min_scene_duration:
                    all_scenes.append(self._create_scene_from_frames(self.frame_buffer))
                self.frame_buffer.clear()
                break
            all_scenes.extend(scenes)
        return all_scenes


class ServerPipeline:
    """Orchestrates the processing of completed scenes using RAFT for optical flow."""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"RAFT Optical Flow will run on: {self.device.upper()}")

        # Load the RAFT model and move it to the GPU
        weights = Raft_Large_Weights.DEFAULT
        self.raft_transforms = weights.transforms()
        self.raft_model = raft_large(weights=weights, progress=False).to(self.device)
        self.raft_model.eval()

    def process_scene(self, scene: Scene):
        """Takes a single, complete scene and runs all necessary processing on it."""
        print(f"\n--- Processing Scene {scene.scene_id} (Frames {scene.start_frame}-{scene.end_frame}) ---")
        self._classify_scene_background(scene)
        if scene.is_static_background:
            print(f"  -> Scene classified as Static BG. Running PointStream pipeline.")
            # We would generate the full-res background here if needed
        else:
            print(f"  -> Scene classified as Dynamic BG. Handing off to fallback codec.")
        scene.frames.clear()

    def _classify_scene_background(self, scene: Scene):
        """Classifies a scene's background by analyzing optical flow with RAFT."""
        if scene.duration < 2:
            scene.is_static_background = False
            return

        flow_magnitudes = []
        num_flow_checks = self.config.get('num_flow_checks', 10)
        check_indices = np.linspace(0, scene.duration - 2, num=num_flow_checks, dtype=int)

        with torch.no_grad():
            for i in check_indices:
                # Get frame pair and convert to RGB
                frame1_rgb = cv2.cvtColor(scene.frames[i].image, cv2.COLOR_BGR2RGB)
                frame2_rgb = cv2.cvtColor(scene.frames[i+1].image, cv2.COLOR_BGR2RGB)

                # Convert to tensors
                tensor1 = to_tensor(frame1_rgb).unsqueeze(0).to(self.device)
                tensor2 = to_tensor(frame2_rgb).unsqueeze(0).to(self.device)

                # Preprocess using RAFT's specific transforms
                tensor1, tensor2 = self.raft_transforms(tensor1, tensor2)

                # Get flow prediction
                list_of_flows = self.raft_model(tensor1, tensor2)
                predicted_flow = list_of_flows[-1][0].cpu().numpy()

                # Calculate magnitude
                magnitude = np.sqrt(predicted_flow[0]**2 + predicted_flow[1]**2)
                flow_magnitudes.append(np.mean(magnitude))

        if not flow_magnitudes:
            scene.is_static_background = False
            return

        avg_scene_flow = np.mean(flow_magnitudes)
        threshold = self.config.get('optical_flow_threshold', 2.0) # RAFT's scale is different, may need tuning
        
        print(f"  -> Avg. RAFT Optical Flow: {avg_scene_flow:.4f}")
        scene.is_static_background = avg_scene_flow < threshold
