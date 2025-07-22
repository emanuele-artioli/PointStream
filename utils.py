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

def save_frames_as_video(output_path: str, frames: List[np.ndarray], fps: float):
    """Saves a list of NumPy frames to a video file using an FFmpeg pipe."""
    print(f"  -> Saving generated frames to '{output_path}' using FFmpeg...")
    height, width, _ = frames[0].shape
    
    # Command to pipe frames to FFmpeg
    # -i - : read from stdin
    # -vcodec libx264: Use the highly compatible H.264 codec
    # -pix_fmt yuv420p: Standard pixel format for web/mobile compatibility
    command = [
        'ffmpeg',
        '-y',  # Overwrite output file if it exists
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{width}x{height}',  # Frame size
        '-pix_fmt', 'bgr24',       # Input pixel format from OpenCV
        '-r', str(fps),            # Frames per second
        '-i', '-',                 # The input comes from stdin
        '-an',                     # No audio
        '-vcodec', 'libx264',
        '-pix_fmt', 'yuv420p',
        output_path
    ]
    
    # Open a subprocess pipe
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Write each frame to the pipe
    for frame in frames:
        process.stdin.write(frame.tobytes())
        
    process.stdin.close()
    process.wait()

def extract_video_segment(video_path: str, start_frame: int, end_frame: int, fps: float, output_path: str):
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
        cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (0, 0), fx=0.5, fy=0.5)
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

def track_objects_in_segment(frames: List[np.ndarray], prompts: List[str]) -> List[List[Dict[str, Any]]]:
    """
    Initializes a YOLOE model and tracks objects frame-by-frame through a segment.
    """
    print(f" -> Initializing YOLOE and tracking objects across {len(frames)} frames...")

    # 1. Initialize the model and set classes once for the entire segment
    model = YOLOE("/home/itec/emanuele/models/yoloe-11l-seg.pt")
    model.set_classes(prompts, model.get_text_pe(prompts))
    
    all_frame_results = []
    # 2. Loop through frames and call track with persist=True
    for frame in frames:
        results = model.track(source=frame, persist=True, verbose=False)
        frame_results = results[0] # track returns a list with one result for the single frame
        
        current_frame_objects = []
        if frame_results.masks is not None and frame_results.boxes.id is not None:
            for i, mask_data in enumerate(frame_results.masks.data):
                class_id = int(frame_results.boxes.cls[i])
                track_id = int(frame_results.boxes.id[i])
                current_frame_objects.append({
                    "prompt": prompts[class_id],
                    "mask": mask_data.cpu().numpy(),
                    "confidence": float(frame_results.boxes.conf[i]),
                    "track_id": track_id
                })
        all_frame_results.append(current_frame_objects)
        
    return all_frame_results