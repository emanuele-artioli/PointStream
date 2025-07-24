# --------------------------------------------------------------------------
# DEPENDENCIES
# --------------------------------------------------------------------------
import subprocess
import os
import json
import io
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any
import cv2
from PIL import Image
from ultralytics import YOLOE, SAM
from google import genai
from google.genai import types

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

def create_segmentation_video(frames: List[np.ndarray], all_frame_results: List[List[Dict[str, Any]]], output_path: str, fps: float):
    """Creates a video with tracked segmentation masks drawn on every frame."""
    if not any(all_frame_results):
        return

    # Generate all visualized frames first
    visualized_frames = []
    track_colors = {}
    color_palette = [plt.cm.viridis(i) for i in np.linspace(0, 1, 20)]

    for i, frame in enumerate(frames):
        # Create a mutable copy for drawing on
        draw_frame = frame.copy()
        overlay = draw_frame.copy()
        
        segmented_objects = all_frame_results[i]
        
        for obj in segmented_objects:
            track_id = obj['track_id']
            if track_id not in track_colors:
                track_colors[track_id] = color_palette[len(track_colors) % len(color_palette)]
            
            color_bgr = [c * 255 for c in track_colors[track_id][:3]][::-1]
            h, w, _ = draw_frame.shape
            mask = cv2.resize(obj['mask'].astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            overlay[mask > 0.5] = color_bgr
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(draw_frame, contours, -1, color_bgr, 2)
            
            label = f"ID {track_id}: {obj['prompt']}"
            if contours:
                M = cv2.moments(contours[0])
                if M['m00'] > 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    cv2.putText(draw_frame, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        final_image = cv2.addWeighted(draw_frame, 0.7, overlay, 0.3, 0)
        visualized_frames.append(final_image)
    
    # Use the new, robust FFmpeg saver
    save_frames_as_video(output_path, visualized_frames, fps)
    print(f"  -> Saved tracking visualization to '{output_path}'")

# --------------------------------------------------------------------------
# AI MODELS UTILITY FUNCTIONS
# --------------------------------------------------------------------------

_gemini_client = None
_yoloe_model = None
_sam_model = None

GEMINI_SYSTEM_INSTRUCTION_TEXT = """
You are a highly efficient scene analysis engine. Your task is to analyze an image from a video and identify only the primary, dynamic subjects.

Your response MUST follow these rules:
1.  From the image, identify all animate or mobile objects that are central to the action.
2.  IGNORE all static background elements like scenery, buildings, rooms, or passive crowds.
3.  For each identified primary subject, map it to one of the following base categories ONLY: ["person", "animal", "vehicle", "object"].
4.  Return a single JSON array containing the unique, simplified base categories. Do not include adjectives, numbers, or descriptions.

Example Input: An image of two men playing tennis.
Response:
["person", "object"]
"""

GEMINI_SYSTEM_INSTRUCTION_POINT = """
You are a motion detection expert analyzing two sequential frames from a video to identify and locate moving objects.

Your response MUST follow these rules:
1.  Compare the two images to identify primary subjects that have significantly changed position. IGNORE static background elements.
2.  For each moving subject, determine its base category from this list ONLY: ["person", "animal", "vehicle", "object"].
3.  For each moving subject, provide its NORMALIZED bounding box [x_min, y_min, x_max, y_max] in the *second* image. All coordinates must be floats between 0.0 and 1.0.
4.  Return a single JSON array of objects. Each object must have a "category" and a "bbox_normalized" key.

Example Input: Two images of a person walking a dog.
Response:
[
  {"category": "person", "bbox_normalized": [0.6, 0.4, 0.75, 0.9]},
  {"category": "animal", "bbox_normalized": [0.5, 0.7, 0.6, 0.85]}
]
"""

def _initialize_models():
    """Initializes all necessary AI models on first use."""
    global _gemini_client, _sam_model, _yoloe_model
    
    # Initialize YOLOE (no change)
    if _yoloe_model is None:
        print(" -> Initializing YOLOE Model...")
        if not os.path.exists("/home/itec/emanuele/models/yoloe-11l-seg.pt"):
            raise FileNotFoundError("YOLOE model file not found at /home/itec/emanuele/models/yoloe-11l-seg.pt")
        _yoloe_model = YOLOE("/home/itec/emanuele/models/yoloe-11l-seg.pt")

    # Initialize Gemini (no change)
    if _gemini_client is None:
        print(" -> Initializing Gemini Client...")
        if not os.getenv("GEMINI_API_KEY"):
            raise ValueError("GEMINI_API_KEY environment variable not set.")
        _gemini_client = genai.Client()

    # Initialize SAM (new)
    if _sam_model is None:
        print(" -> Initializing Segment Anything Model (SAM)...")
        # Model will auto-download on first use
        _sam_model = SAM('/home/itec/emanuele/models/sam2.1_l.pt')

def _query_gemini_vision(contents: List, mode: str) -> str:
    """Sends a multimodal prompt to the Gemini API."""
    print(f"  -> Querying Gemini API in '{mode}' mode...")
    system_instruction = GEMINI_SYSTEM_INSTRUCTION_POINT if mode == 'point' else GEMINI_SYSTEM_INSTRUCTION_TEXT
    try:
        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            response_mime_type="application/json",
            system_instruction=system_instruction,
        )
        
        response = _gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=config,
        )
        return response.text
    except Exception as e:
        print(f"  -> Error: Gemini API call failed. Reason: {e}")
        return "[]"

def get_filtered_prompts(keyframe: np.ndarray) -> List[str]:
    """Uses Gemini to analyze a keyframe and return simplified prompts."""
    _initialize_models() # Corrected from _initialize_gemini_client
    rgb_image = cv2.cvtColor(keyframe, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)
    image_byte_buffer = io.BytesIO()
    pil_image.save(image_byte_buffer, format="JPEG")
    contents = [types.Part.from_bytes(data=image_byte_buffer.getvalue(), mime_type='image/jpeg'), 
                "Analyze this image based on the system instruction."]
    llm_response_str = _query_gemini_vision(contents, mode='text')
    try:
        prompts = json.loads(llm_response_str)
        if isinstance(prompts, list):
            clean_prompts = [str(p) for p in prompts]
            print(f"  -> Using Filtered Prompts from Gemini: {clean_prompts}")
            return clean_prompts
        else:
            print(f"  -> Error: Gemini response was not a JSON list. Response: {llm_response_str}")
            return []
    except (json.JSONDecodeError, TypeError):
        print(f"  -> Error: Could not parse JSON response. Response:\n{llm_response_str}")
        return []

def track_objects_in_segment(frames: List[np.ndarray], prompts: List[str]) -> List[List[Dict[str, Any]]]:
    """
    Initializes a YOLOE model and tracks objects frame-by-frame through a segment.
    """
    print(f" -> Initializing YOLOE and tracking objects across {len(frames)} frames...")
    _initialize_models()

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

def get_box_prompts_from_frames(frame_start: np.ndarray, frame_end: np.ndarray) -> List[Dict[str, Any]]:
    """Uses Gemini to identify moving objects between two frames and return their center points."""
    _initialize_models()

    def to_part(frame):
        """Converts a frame to a Gemini Part object."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        buffer = io.BytesIO()
        pil.save(buffer, format="JPEG")
        return types.Part.from_bytes(data=buffer.getvalue(), mime_type='image/jpeg')

    # The prompt to the model is now simpler, without dimensions
    contents = [
        to_part(frame_start),
        to_part(frame_end),
        "Analyze these two frames to identify moving objects based on the system instruction."
    ]

    llm_response_str = _query_gemini_vision(contents, mode='point')

    try:
        boxes = json.loads(llm_response_str)
        if isinstance(boxes, list):
            print(f"  -> Got Box Prompts from Gemini: {boxes}")
            return boxes
        return []
    except (json.JSONDecodeError, TypeError):
        print(f"  -> Error: Could not parse JSON response for points. Response:\n{llm_response_str}")
        return []

def segment_with_boxes(frames: List[np.ndarray], box_prompts: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """Segments objects in frames using a list of bounding box prompts with SAM."""
    print(f" -> Segmenting {len(frames)} frames with SAM using {len(box_prompts)} box prompts...")
    _initialize_models() # Ensures SAM is loaded
    all_frame_results = []
    
    h, w, _ = frames[0].shape

    for frame in frames:
        current_frame_objects = []
        for i, prompt in enumerate(box_prompts):
            normalized_bbox = prompt.get("bbox_normalized")
            category = prompt.get("category")
            if not normalized_bbox or not category:
                continue

            # Convert normalized bbox to absolute pixel values [x1, y1, x2, y2]
            x1 = int(normalized_bbox[0] * w)
            y1 = int(normalized_bbox[1] * h)
            x2 = int(normalized_bbox[2] * w)
            y2 = int(normalized_bbox[3] * h)
            
            # --- UPDATED: Run inference with SAM using a bounding box ---
            results = _sam_model(source=frame, bboxes=[x1, y1, x2, y2], verbose=False)
            
            if results and results[0].masks:
                mask_data = results[0].masks.data[0]
                current_frame_objects.append({
                    "prompt": category,
                    "mask": mask_data.cpu().numpy(),
                    "confidence": None, 
                    "track_id": i + 1 
                })
        all_frame_results.append(current_frame_objects)
    return all_frame_results
