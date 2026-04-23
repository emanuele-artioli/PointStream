## **description: 'Master Agent System Instructions for Pointstream: Object-Centric Semantic Neural Codec' applyTo: '\*\*/\*.py, \*\*/environment.yaml, \*\*/pyproject.toml, \*\*/README.md'**

# **Pointstream System Development**

## **Instructions (Highest Priority)**

* **Permissions:** You are NOT an admin. DO NOT run commands with sudo or apt.  
* **Headless Mode:** Assume a headless multi-GPU Linux environment (CUDA). DO NOT run graphical applications (e.g., cv2.imshow(), plt.show()). **Instead of displaying graphical interfaces, always save media files and visualizations to the disk.**  
* **Dependency Management:** pyproject.toml is the ONE AND ONLY source of truth for standard Python packages. If you need a new pip package, add it there. Use environment.yaml STRICTLY as a GPU environment bootstrapper for heavy CUDA drivers/PyTorch binaries. Never use raw requirements.txt.  
* **Execution Environment:** Check for the existence of the pointstream conda environment and run all scripts within it.  
* **Default Test File:** Unless specified otherwise, always use the following file for testing and runs: /home/itec/emanuele/pointstream/assets/real\_tennis.mp4  
* **Real Experiment Input Requirement:** Any real experiment (non-synthetic evaluation intended to validate end-to-end behavior) MUST pass `--input /home/itec/emanuele/pointstream/assets/real_tennis.mp4` explicitly. Do not rely on the mock-source fallback.  
* **Testing & Review Workflow:** Use test-driven generation. Immediately test generated code and continue bugfixing until successful. When a task is complete, review it:  
  * **Analysis Focus:** Analyze code quality, structure, performance bottlenecks, potential bugs, security issues, and user experience/accessibility.  
  * **Important Guidelines:** Structure feedback with clear headings. Ask clarifying questions about design decisions. Focus on *what* should be changed and *why*.  
  * **CRITICAL:** DO NOT write or suggest specific code changes directly during the review phase.  
* **Overrides:** If a modification you made is missing later, assume the user explicitly changed it. DO NOT revert it.  
* **Documentation:** Always update pyproject.toml and README.md with relevant details when modifying project structure or dependencies.  
* **Bootstrapping (Mock-First):** When adding *new* AI models (e.g. a new extractor), continue to use a mock-first approach. Return deterministic dummy PyTorch tensors of the exact correct shape until the pipeline works, before loading the heavy real model. Use MockActorExtractor for all unit plumbing tests.  
* **Model Weights (Local Symlinking):** For local execution, you should search for model weights in /home/itec/emanuele/Models and symlink them into assets/weights/. However, DO NOT mention the /home/itec/... path in the user-facing README.md—just tell users to place weights in assets/weights/.  
* **Execution Paradigm:** Operate on VideoChunks (e.g., 2-second clips), not infinite streams.

## **Best Practices**

* Prefer functional programming for data transformations, but use OOP for stateful engines and Base Interfaces.  
* Use pydantic models or dataclasses for all cross-module communication. Never use raw dictionaries.  
* **Shape Hints are mandatory:** Every PyTorch tensor must have a shape hint in the comments (e.g., \# Shape: \[Batch, Frames, Keypoints, Coords\]).  
* Implement **Event-Driven Sparsity (Non-Uniform Keyframing)**.  
* Include frame\_id and object\_id in every transmitted semantic event for strict tracking.  
* **Shared Synthesis Architecture:** The server MUST use the exact same standalone SynthesisEngine as the client to predict hallucinated frames and calculate true generative residuals.  
* **Scene Classification Routing:** The architecture must modularly support routing scenes to a traditional fallback codec (for static "Interludes" like crowd shots) versus the Pointstream semantic pipeline (for active "Exchanges").  
* **Ablation & Baseline Strategy:** Evaluate new components by comparing them against the "Whole-Frame Residual Baseline" (panorama \+ generated actors \+ whole-frame residual catching missing rackets/balls). Iteratively enable specialized extractors to measure their exact impact on reducing the residual payload size.  
* Orchestrate tasks using a Directed Acyclic Graph (DAG) and the InlineExecutionPool or TaggedMultiprocessPool.  
* Tag node/function routing logically using decorators (@cpu\_bound for I/O/FFmpeg, @gpu\_bound for PyTorch/Tensor math).  
* Use torch.multiprocessing (shared memory) to pass tensors between processes to avoid PCIe/Pickling bottlenecks (make\_shared\_cpu\_tensor).  
* Abstract network transmission behind a BaseTransport interface with .send(payload) and .receive().  
* Maintain the exact project scaffold. Do not dump code into a monolithic file:  
  * src/main.py (Main entrypoint)  
  * src/shared/ (Schemas, SynthesisEngine, Interfaces)  
  * src/encoder/ (Extractors, DAG orchestrator, residual calculator)  
  * src/decoder/ (Renderer, residual compositing)  
  * src/transport/ (DiskTransport)  
  * scripts/ (Weight downloads, conda setup, run\_mock\_pipeline)  
  * assets/ (Weights, test video chunks)  
  * tests/ (Test scripts)  
  * outputs/ (Timestamped subfolders containing media files for different runs)

## **Common Patterns**

### **CLI / Main Execution Pattern**

import argparse  
import os  
from datetime import datetime

def main():  
    parser \= argparse.ArgumentParser(description="Pointstream Pipeline")  
    parser.add\_argument("--input", type=str, default="/home/itec/emanuele/pointstream/assets/real\_tennis.mp4")  
      
    \# Generate timestamped outputs directory (plural 'outputs/')  
    timestamp \= datetime.now().strftime("%Y%m%d\_%H%M%S")  
    default\_out \= os.path.join(os.getcwd(), "outputs", timestamp)  
    parser.add\_argument("--output\_dir", type=str, default=default\_out)  
      
    args \= parser.parse\_args()  
    os.makedirs(args.output\_dir, exist\_ok=True)  
    \# Pipeline execution...

### **Test-Time Setup and Mocking**

import os  
import unittest  
from src.encoder.mock\_extractors import MockActorExtractor

class TestEncoderPlumbing(unittest.TestCase):  
    def setUp(self):  
        \# Always check for run-scoped debug artifact directory  
        self.debug\_dir \= os.environ.get("POINTSTREAM\_DEBUG\_ARTIFACT\_DIR", "./outputs/tests/debug")  
        os.makedirs(self.debug\_dir, exist\_ok=True)  
          
        \# Bypass real models for plumbing tests  
        self.extractor \= MockActorExtractor()

### **AI Model Integration (Wrapper Pattern)**

import torch

class YOLOWrapper:  
    def \_\_init\_\_(self, weights\_path: str):  
        \# Load weights from assets/weights/ here  
        pass

    def process(self, frame\_batch: torch.Tensor) \-\> torch.Tensor:  
        """  
        Wraps YOLO execution.   
        Input Shape: \[Batch, Channels, Height, Width\]  
        Output Shape: \[Batch, Num\_Boxes, 6\] (x1, y1, x2, y2, conf, cls)  
        """  
        \# Actual inference goes here. For mock pass, return zeros.  
        batch\_size \= frame\_batch.shape\[0\]  
        return torch.zeros(batch\_size, 10, 6\)

### **Data Contracts with Pydantic**

from pydantic import BaseModel  
from typing import Literal, Any

class SemanticEvent(BaseModel):  
    """Event-driven sparsity schema for non-uniform keyframing"""  
    frame\_id: int  
    object\_id: str  
    command: Literal\["Keyframe", "Interpolate", "Static"\]  
    data: Any | None \= None 

### **Resource Tagging (The DAG)**

from typing import Callable

def cpu\_bound(func: Callable):  
    """Tag for file I/O, FFmpeg, and graph logic"""  
    func.is\_cpu\_bound \= True  
    return func

def gpu\_bound(func: Callable):  
    """Tag for PyTorch inference and tensor math"""  
    func.is\_gpu\_bound \= True  
    return func

@gpu\_bound  
def calculate\_residuals(original: torch.Tensor, synthetic: torch.Tensor) \-\> torch.Tensor:  
    \# Shape: \[Batch, Frames, Channels, Height, Width\]  
    return original \- synthetic  
