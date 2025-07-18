import subprocess
import os
import numpy as np
from typing import List, Tuple, Dict, Optional
from skimage.metrics import structural_similarity as ssim
import cv2

# --- New imports for contextual understanding ---
from PIL import Image
import torch

# --- Lazy-loaded model placeholders ---
_vlm_processor = None
_vlm_model = None
_nlp = None

import subprocess
import os
import numpy as np
from typing import List, Tuple, Dict, Optional
from skimage.metrics import structural_similarity as ssim
import cv2

# --- Imports for contextual understanding ---
from PIL import Image
import torch
from nltk.corpus import wordnet # <-- New import for generalization

# --- Lazy-loaded model placeholders ---
_vlm_processor = None
_vlm_model = None
_nlp = None

def _initialize_models():
    """Initializes the VLM and NLP models on first use."""
    global _vlm_processor, _vlm_model, _nlp
    
    if _vlm_processor is None:
        print(" -> Initializing Vision-Language Model for the first time...")
        from transformers import BlipProcessor, BlipForConditionalGeneration
        _vlm_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        _vlm_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base", use_safetensors=True
        )
        print(" -> VLM ready.")

    if _nlp is None:
        print(" -> Initializing NLP Model for the first time...")
        import spacy
        _nlp = spacy.load("en_core_web_sm")
        print(" -> NLP ready.")

def _get_general_category(noun_chunk: str) -> str:
    """
    Finds a general category (hypernym) for a noun using WordNet.
    Example: 'mini cooper' -> 'car'
    """
    # Look up the noun in WordNet
    synsets = wordnet.synsets(noun_chunk.replace(" ", "_"))
    if not synsets:
        return noun_chunk # Return original if not found

    # Get the first and most common meaning
    first_synset = synsets[0]
    
    # Get its hypernyms (more general categories)
    hypernyms = first_synset.hypernyms()
    if not hypernyms:
        return noun_chunk # Return original if it has no broader category

    # Return the name of the most direct hypernym
    return hypernyms[0].lemmas()[0].name().replace("_", " ")

def get_scene_context(frames: List[np.ndarray]) -> Tuple[str, List[str]]:
    """
    Analyzes a keyframe to generate a caption and extract generalized key nouns.
    """
    if not frames:
        return "No frames provided", []

    _initialize_models()

    keyframe = frames[len(frames) // 2]
    
    # Generate caption with VLM
    rgb_image = cv2.cvtColor(keyframe, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)
    inputs = _vlm_processor(images=pil_image, return_tensors="pt")
    output_ids = _vlm_model.generate(**inputs, max_length=50)
    caption = _vlm_processor.decode(output_ids[0], skip_special_tokens=True).strip()

    # --- UPDATED LOGIC FOR NOUN GENERALIZATION ---
    doc = _nlp(caption)
    
    # 1. Extract noun chunks (e.g., "a mini cooper", "the two horses")
    # 2. Get the root noun text of the chunk (e.g., "mini cooper", "horses")
    # 3. Generalize each noun using our WordNet helper
    # 4. Use a set to store unique categories, then sort
    
    generalized_nouns = set()
    for chunk in doc.noun_chunks:
        # Use the root of the chunk for better lookup
        noun_to_generalize = chunk.root.text
        if chunk.root.text in chunk.text and len(chunk.text) > len(chunk.root.text):
             # Handle cases like 'mini cooper' where root is 'cooper' but we want the whole chunk
             noun_to_generalize = chunk.text
        
        # Exclude pronouns like 'it', 'he', 'they' which spaCy can tag as nouns
        if chunk.root.pos_ == 'PRON':
            continue

        category = _get_general_category(noun_to_generalize)
        generalized_nouns.add(category)
        
    # If no noun chunks found, fall back to single nouns
    if not generalized_nouns:
        for token in doc:
            if token.pos_ == 'NOUN':
                category = _get_gengeral_category(token.lemma_)
                generalized_nouns.add(category)

    return caption, sorted(list(generalized_nouns))

def classify_scene_motion(frames: List[np.ndarray]) -> str:
    """
    Classifies a list of frames as 'static' or 'dynamic' based on optical flow.
    This version samples which CONSECUTIVE pairs to analyze for better accuracy.
    """
    if len(frames) < 2:
        return "static"

    MOTION_THRESHOLD = 0.5 
    MAX_FRAMES_TO_SAMPLE = 45
    frame_interval = max(1, len(frames) // MAX_FRAMES_TO_SAMPLE)
    
    # 1. Pre-process all frames once for efficiency (grayscale + downscale)
    processed_frames = [
        cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), None, fx=0.25, fy=0.25) 
        for f in frames
    ]

    total_flow = 0
    flow_calculations = 0
    
    # 2. Loop through consecutive frames, but only calculate flow periodically
    for i in range(1, len(processed_frames)):
        # This check determines if we run the expensive calculation for this pair
        if i % frame_interval == 0:
            prev_frame = processed_frames[i-1]
            current_frame = processed_frames[i]
        
            flow = cv2.calcOpticalFlowFarneback(prev_frame, current_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
            total_flow += np.mean(magnitude)
            flow_calculations += 1

    if flow_calculations == 0:
        return "static"

    avg_motion = total_flow / flow_calculations
    scene_type = "dynamic" if avg_motion > MOTION_THRESHOLD else "static"
    print(f"  -> Analyzed motion for segment. Avg: {avg_motion:.2f}. Type: {scene_type.upper()}")
    
    return scene_type

def save_video_segment(video_path: str, start_frame: int, end_frame: int, fps: float, output_path: str):
    """Saves a video segment using FFmpeg by re-encoding for accuracy."""
    start_time = start_frame / fps
    end_time = end_frame / fps
    
    print(f"  -> Saving and re-encoding segment to '{output_path}'...")
    command = [
        'ffmpeg', '-y', 
        '-i', video_path,
        '-ss', str(start_time), 
        '-to', str(end_time),
        '-avoid_negative_ts', '1', 
        output_path
    ]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def get_video_properties(video_path: str) -> Tuple[int, float]:
    """Gets total frame count and FPS of a video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
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
    
    frames = []
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