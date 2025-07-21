# --------------------------------------------------------------------------
# GENERAL DEPENDENCIES
# --------------------------------------------------------------------------
import subprocess
import os
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from skimage.metrics import structural_similarity as ssim
import cv2

# --------------------------------------------------------------------------
# MODEL DEPENDENCIES
# --------------------------------------------------------------------------
from PIL import Image
import torch
from nltk.corpus import wordnet
from ultralytics import YOLOE # New main model import

# --------------------------------------------------------------------------
# GENERAL UTILITY FUNCTIONS
# --------------------------------------------------------------------------

def classify_scene_motion(frames: List[np.ndarray]) -> str:
    """Classifies the motion in a sequence of frames as 'static' or 'dynamic'."""
    if len(frames) < 2: return "static"
    MOTION_THRESHOLD = 0.5
    MAX_FRAMES_TO_SAMPLE = 45
    frame_interval = max(1, len(frames) // MAX_FRAMES_TO_SAMPLE)
    processed_frames = [cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), None, fx=0.25, fy=0.25) for f in frames]
    total_flow, flow_calculations = 0, 0
    for i in range(1, len(processed_frames)):
        if i % frame_interval == 0:
            flow = cv2.calcOpticalFlowFarneback(processed_frames[i-1], processed_frames[i], None, 0.5, 3, 15, 3, 5, 1.2, 0)
            magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            total_flow += np.mean(magnitude)
            flow_calculations += 1
    if flow_calculations == 0: return "static"
    avg_motion = total_flow / flow_calculations
    scene_type = "dynamic" if avg_motion > MOTION_THRESHOLD else "static"
    print(f"  -> Analyzed motion for segment. Avg: {avg_motion:.2f}. Type: {scene_type.upper()}")
    return scene_type

def save_video_segment(video_path: str, start_frame: int, end_frame: int, fps: float, output_path: str):
    """Saves a segment of the video from start_frame to end_frame."""
    start_time, end_time = start_frame / fps, end_frame / fps
    print(f"  -> Saving segment to '{output_path}'...")
    command = ['ffmpeg', '-y', '-i', video_path, '-ss', str(start_time), '-to', str(end_time), '-avoid_negative_ts', '1', output_path]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def get_video_properties(video_path: str) -> Tuple[int, float]:
    """Returns the total number of frames and FPS of the video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): raise ValueError(f"Cannot open video file: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return total_frames, fps

def extract_frames(video_path: str, frame_range: Tuple[int, int]) -> List[np.ndarray]:
    """Extracts a range of frames from a video file."""
    start_frame, end_frame = frame_range
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    
    frames = [] # <-- This line was the issue
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    current_frame_pos = start_frame
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    effective_end_frame = min(end_frame, total_frames - 1)

    while current_frame_pos <= effective_end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        current_frame_pos += 1
    
    cap.release()
    return frames

def detect_scene_changes(frames: List[np.ndarray], threshold: float, analysis_window: int) -> Dict[int, float]:
    """Detects significant scene changes within a list of frames."""
    if len(frames) < 2:
        return {}

    processed_frames = [
        cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (0, 0), fx=0.25, fy=0.25)
        for frame in frames
    ]
    scene_changes = {}

    def _find_best_cut_in_range(start: int, end: int) -> Optional[Tuple[int, float]]:
        """Finds the single most significant cut in a small window."""
        min_score, cut_location = 1.0, None
        if start >= end: return None
        for i in range(start + 1, min(end + 1, len(processed_frames))):
            score, _ = ssim(processed_frames[i-1], processed_frames[i], full=True)
            if score < min_score:
                min_score, cut_location = score, i
        if cut_location is not None and min_score < threshold:
            return cut_location, min_score
        return None

    def _find_changes_recursive(start: int, end: int):
        """Recursively searches for scene changes."""
        if (end - start) <= analysis_window:
            best_cut = _find_best_cut_in_range(start, end)
            if best_cut:
                cut_idx, cut_score = best_cut
                if not any(abs(cut_idx - exist_cut) < analysis_window // 2 for exist_cut in scene_changes):
                    scene_changes[cut_idx] = cut_score
            return
        
        if end >= len(processed_frames): end = len(processed_frames) -1
        boundary_score, _ = ssim(processed_frames[start], processed_frames[end], full=True)
        if boundary_score < threshold:
            mid = (start + end) // 2
            _find_changes_recursive(start, mid)
            _find_changes_recursive(mid + 1, end)

    _find_changes_recursive(0, len(frames) - 1)
    return scene_changes

# --------------------------------------------------------------------------
# MODELS INITIALIZATION
# --------------------------------------------------------------------------

_vlm_processor = None
_vlm_model = None
_nlp = None
_segmentation_model = None # For YOLOE

def _initialize_models():
    """Initializes the VLM and NLP models (which are safe to load once)."""
    global _vlm_processor, _vlm_model, _nlp
    
    if _vlm_processor is None:
        print(" -> Initializing Vision-Language Model...")
        from transformers import BlipProcessor, BlipForConditionalGeneration
        _vlm_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        _vlm_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base", use_safetensors=True
        )

    if _nlp is None:
        print(" -> Initializing NLP Model...")
        import spacy
        _nlp = spacy.load("en_core_web_sm")

# --------------------------------------------------------------------------
# MODELS FUNCTIONS
# --------------------------------------------------------------------------

def generate_caption(keyframe: np.ndarray) -> str:
    """Uses the VLM to generate a text caption for a single image frame."""
    _initialize_models() # Ensure models are ready
    pil_image = Image.fromarray(cv2.cvtColor(keyframe, cv2.COLOR_BGR2RGB))
    
    inputs = _vlm_processor(images=pil_image, return_tensors="pt")
    output_ids = _vlm_model.generate(**inputs, max_length=50)
    caption = _vlm_processor.decode(output_ids[0], skip_special_tokens=True)
    
    print(f"  -> Generated Caption: '{caption}'")
    return caption

def extract_prompts_from_caption(caption: str) -> List[str]:
    """Uses NLP to parse a caption and extract clean noun phrases for prompts."""
    _initialize_models() # Ensure models are ready
    doc = _nlp(caption)
    prompts = set()
    for chunk in doc.noun_chunks:
        phrase_tokens = [
            token.lemma_ for token in chunk if token.pos_ not in ['DET', 'PRON']
        ]
        if phrase_tokens:
            prompts.add(" ".join(phrase_tokens))
    
    sorted_prompts = sorted(list(prompts))
    print(f"  -> Using Text Prompts: {sorted_prompts}")
    return sorted_prompts

def run_segmentation(keyframe: np.ndarray, prompts: List[str]) -> List[Dict[str, Any]]:
    """Runs the YOLOE model on a frame with a given list of text prompts."""
    if not prompts:
        print("  -> No usable text prompts provided. Skipping segmentation.")
        return []

    # --- FIX: Instantiate a fresh model for each run to avoid state issues ---
    print(" -> Initializing new YOLOE model instance for this segment...")
    segmentation_model = YOLOE("/home/itec/emanuele/models/yoloe-11l-seg.pt")

    # 1. Set the custom vocabulary on the new model instance
    segmentation_model.set_classes(prompts, segmentation_model.get_text_pe(prompts))

    # 2. Run inference to get segmentation results
    results = segmentation_model.predict(keyframe, save=False, verbose=False)
    
    # 3. Process and return results in a clean format
    segmented_objects = []
    if results and results[0].masks is not None:
        for i, mask_data in enumerate(results[0].masks.data):
            class_id = int(results[0].boxes.cls[i])
            segmented_objects.append({
                "prompt": prompts[class_id],
                "mask": mask_data.cpu().numpy(),
                "confidence": float(results[0].boxes.conf[i])
            })
    return segmented_objects