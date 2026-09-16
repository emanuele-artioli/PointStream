# PointStream Interactive Demo Architecture & Roadmap (Modes 2 & 3)

This document specifies the technical architecture and implementation roadmap for turning PointStream into a dynamic, end-to-end interactive demonstration platform.

---

## 1. Executive Summary & Decoupling Philosophy

The PointStream demonstration architecture is structured around two key constraints:
1. **Zero Downtime During Performance Optimizations**: The performance team is actively optimizing model inference (TensorRT, CUDA kernel fusion, batching, SVT-AV1 parameter tuning). The interactive front-end and transport layers **must not depend on internal pipeline classes or file structures**.
2. **Path to Production (`emanueleartioli.com`)**: While development runs on the remote research server (`data3`/`gpu5`), the goal is hosting a showcase on `emanueleartioli.com` that visitors can explore or that can be run locally during video calls with partners like Figure.ai.

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend (Web / UI)                      │
│      • Mode 1: Synchronized Canvas A/B Inspector            │
│      • Mode 2: Video Upload / Library Player                │
│      • Mode 3: Live Webcam Teleoperation HUD                │
└──────────────────────────────┬──────────────────────────────┘
                               │ WebSocket / WebRTC Binary Protocol
┌──────────────────────────────▼──────────────────────────────┐
│                  Decoupled Gateway / API                    │
│           (FastAPI / aiohttp async worker pool)             │
└──────────────────────────────┬──────────────────────────────┘
                               │ Stable Python Interface (StreamingCodecAdapter)
┌──────────────────────────────▼──────────────────────────────┐
│                    Codec Engine Adapter                     │
│   class StreamingCodec(Protocol):                           │
│       def initialize(config: CodecConfig) -> SessionInfo    │
│       def capture_anchor(frame_crop: np.ndarray) -> bytes   │
│       def process_frame(frame: np.ndarray) -> FrameResult   │
└──────────────────────────────┬──────────────────────────────┘
                               │
       ┌───────────────────────┴───────────────────────┐
       ▼                                               ▼
[Current Demo Engine]                      [Next-Gen Optimized Engine]
• MediaPipe Python (14.9 ms)               • Fused TensorRT UNet (1.2 ms)
• PyTorch UNet (2.9 ms)                    • Hardware AV1 Encoder (NVENC / QuickSync)
• SVT-AV1 CPU (13.6 ms)                    • Zero-copy GPU memory pipeline
```

---

## 2. Mode 2: Dynamic Video Selection & Server Processing

### 2.1 User Experience & Interaction
1. **Video Selection**: The user chooses an egocentric sequence from an extended benchmark library (e.g. factory assembly, surgical teleop, kitchen manipulation) or uploads a custom `.mp4`/`.mov` clip via drag-and-drop.
2. **Configurable Operating Points**:
   - Target Bitrate ladder: 150 kbps, 250 kbps (PointStream matched), 500 kbps, 1000 kbps.
   - Codec Baseline: SVT-AV1 (presets 6, 7, 10), H.265 / HEVC.
3. **Live Dual-Stream HUD**:
   - Video frames are streamed dynamically as they are processed.
   - A real-time telemetry HUD charts:
     - Instantaneous bitrate ($R(t)$) in kbps.
     - Per-stage latency breakdown (pose extraction, background coding, neural synthesis).
     - Downstream robot tracking accuracy (MPJPE joint error in pixels).

### 2.2 System Architecture & Protocols
To isolate the web application from codec refactors, the backend exposes a standard async interface:

```python
from typing import Protocol, Iterator
from dataclasses import dataclass
import numpy as np

@dataclass
class CodecTelemetry:
    frame_idx: int
    timestamp_ms: float
    pointstream_bytes: int
    av1_bytes: int
    ps_latency_breakdown_ms: dict[str, float]
    av1_latency_ms: float
    joint_tracking_error_px: float
    detection_rate: float

@dataclass
class FramePacket:
    frame_idx: int
    pointstream_rgb: np.ndarray
    av1_rgb: np.ndarray
    telemetry: CodecTelemetry

class StreamingCodecAdapter(Protocol):
    def configure(self, width: int, height: int, target_kbps: int) -> None: ...
    def process_stream(self, video_path_or_frames: Iterator[np.ndarray]) -> Iterator[FramePacket]: ...
```

- **Transport**: Frames are encoded into lightweight JPEG or WebP binary buffers; telemetry is multiplexed into a companion JSON WebSocket payload.
- **Worker Isolation**: Video processing runs in isolated subprocesses or Celery/Redis tasks, ensuring long encodes never block the HTTP event loop.

---

## 3. Mode 3: Live Camera Teleoperation (Hand Motion via Webcam)

### 3.1 The Robotic Teleoperation Problem
In real-world robotic teleoperation (e.g., controlling a humanoid robot via an operator wearing an Apple Vision Pro or looking at an egocentric monitor):
- **Bandwidth is constrained**: Cellular / Wi-Fi links often have <500 kbps reliable uplink.
- **Latency is fatal**: Delays >50 ms cause operator nausea and teleop instability.
- **Finger precision is critical**: Conventional codecs (AV1/H.265) smear fast finger movements into macroblock mush, dropping downstream robot joint tracking.

Mode 3 brings this exact experience to any user with a webcam.

### 3.2 User Interaction Flow
1. **Camera Permission**: The browser requests webcam access (`navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720, frameRate: 30 } })`).
2. **Anchor Calibration Step (1 Second UX)**:
   - PointStream requires an initial appearance anchor to condition its neural generator.
   - The UI shows a lightweight bounding box with the prompt: *"Hold your hand still inside the target box for 1 second"*.
   - Once detected with confidence >0.9, the anchor crop is transmitted to the server.
3. **Live Teleoperation Stream**:
   - The user moves their hands, gestures, or performs fine manipulation motions.
   - The screen splits into:
     - **Left: PointStream (Sub-300 kbps)**: Neural generator synthesizes crisp finger articulation from the 47-byte/hand skeleton stream.
     - **Right: AV1 Baseline (Sub-300 kbps)**: Real-time SVT-AV1 encode, showing visible motion blur, finger dropping, and macroblocking.
   - **Interactive Latency & Error Counters**: Live running counter showing sub-50ms teleoperation budget compliance.

### 3.3 Streaming Protocol Options (WebSockets vs WebRTC)

| Protocol | Latency | Complexity | Implementation Path |
|---|---|---|---|
| **Binary WebSocket + JPEG/WebP** | ~25–40 ms | Low | Recommended for Phase 1. Canvas captures webcam frames, sends binary blobs over WebSocket, receives dual-stream frames back. |
| **WebRTC MediaStream** | ~15–25 ms | Medium-High | Pure peer-to-peer real-time streaming using `aiortc` on the backend. Ideal for production low-latency teleop. |

### 3.4 Handling Generator Generalization
The current demo generator ([overfit_generator.pt](file:///home/itec/emanuele/pointstream/demo/outputs/models/overfit_generator.pt)) was trained on worker 001 from factory 001. For arbitrary users on webcams with varied lighting and skin tones:
1. **Appearance Anchor Conditioning**: The conditional UNet takes the user's calibrated hand crop concatenated with the skeleton canvas.
2. **Fail-Safe Telemetry Visualization**: Even if the neural synthesis has minor appearance artifacts on unseen skin tones, the **47-byte wireframe stream** is 100% robust. The UI prominently highlights the wireframe telemetry overlay, showing that the robot receives flawless 3D joint telemetry regardless of video compression.

---

## 4. Production Deployment to `emanueleartioli.com`

The transition from research server to public portfolio follows three stages:

### Stage 1: Static Showcase (Mode 1 - Current)
- **Zero Server Cost**: Host the Mode 1 interactive inspector as a static web application on GitHub Pages, Cloudflare Pages, or Netlify under `pointstream.emanueleartioli.com`.
- **High Impact**: Zero cold-start delay, 60fps canvas performance, instant split-curtain inspection and telemetry HUD.

### Stage 2: Hosted Demo Backend (Modes 2 & 3)
- Deploy the FastAPI `StreamingCodecAdapter` gateway inside a Docker container on an on-demand GPU instance (e.g. RunPod Serverless, Modal, or a dedicated RTX 4000/4090 host).
- When a visitor visits `emanueleartioli.com/pointstream/live`, the frontend opens a secure WebSocket (`wss://api.pointstream.emanueleartioli.com/ws/teleop`).

### Stage 3: Edge WebGPU Client Synthesis (Ultimate Vision)
- Export the conditional UNet generator to **ONNX Runtime Web / WebGPU**.
- Run MediaPipe Hands directly in the browser via WebAssembly.
- The browser encodes keypoints locally and runs neural synthesis directly on the visitor's local GPU, eliminating all server GPU hosting costs.

