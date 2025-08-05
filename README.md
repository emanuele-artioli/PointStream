# **Pointstream: A Content-Aware Neural Video Codec**

Pointstream is a research project and prototype implementation of a **semantic video codec**. Unlike traditional codecs that compress raw pixels, Pointstream analyzes video content to separate the static background from dynamic foreground objects. It then transmits a highly compressed, structured representation of the scene—consisting of appearance and motion data—which is reconstructed on the client-side using generative AI models.

This approach aims to achieve significantly lower bitrates for videos with simple camera motion compared to traditional codecs like AV1, forming the core thesis of the associated research paper.

## **Core Concepts**

* **Content-Awareness:** The codec differentiates between static backgrounds, moving objects, and camera motion, applying the optimal strategy for each.
* **Disentanglement:** The system separates an object's visual **appearance** (what it looks like, sent once) from its **motion** (how it moves, sent as a continuous, low-data stream of keypoints).
* **Hybrid Encoding:** The pipeline intelligently analyzes camera motion. **Simple scenes** (static, pans, zooms) are processed by the neural codec, while **complex scenes** are gracefully handed off to a traditional codec to ensure quality.

## **Project Structure**

The project is organized with a clean separation between the core Python package, executable scripts, and generated artifacts.

```
PointStream/
├── .gitignore
├── README.md
├── environment.yml             # Conda env for the main pipeline
├── environment-training.yml    # Conda env for the training script
|
├── artifacts/                  # All generated files (ignored by git)
│   ├── training/               # Labeled datasets and trained models
│   └── pipeline_output/        # JSON results and images from the server
|
├── data/                       # Your raw, unlabeled images for training
|
├── pointstream/                # The core, importable Python package
│   ├── client/                 # Client-side reconstruction logic
│   ├── config.py               # Centralized configuration
│   ├── models/                 # Wrappers for AI models
│   ├── pipeline/               # The four stages of the server pipeline
│   └── utils/                  # Reusable helper functions
|
├── tests/                      # Unit and integration tests
|
├── run_server.py               # Entry point to run the main pipeline
├── run_client.py               # Entry point to run video reconstruction
└── train.py                    # Entry point to train a new student model
```

## **Setup**

The project requires two separate Conda environments to avoid dependency conflicts between the main application and the training script.

### **1. Main Environment Setup (`pointstream`)**

This environment is for running the main server pipeline and all tests.

1. **Create the Conda environment:**
   ```bash
   conda env create -f environment.yml
   ```

2. **Activate the environment:**
   ```bash
   conda activate pointstream
   ```

3. **Install MMLab Dependencies:**
   ```bash
   mim install mmcv==2.1.0 mmdet==3.2.0 mmpose==1.3.2 mmpretrain
   ```

4. **Uninstall the old version to ensure a clean build:**
   ```bash
   pip uninstall mmcv -y
   ```

5. **Set the environment variables and run the install:**
   ```bash
   FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="8.9" pip install mmcv==2.1.0 --no-cache-dir
   ```

### **2. Training Environment Setup (`pointstream-training`)**

This isolated environment is used **only** for running the `train.py` script.

1. **Create the Conda environment:**
   ```bash
   conda env create -f environment-training.yml
   ```

2. **Activate the environment:**
   ```bash
   conda activate pointstream-training
   ``` 

## **Usage**

### **1. Train a Custom Student Model**

(Requires the `pointstream-training` environment)

To train a new, specialized model (e.g., for sports), modify the `ontology` in `train.py` to include your desired classes and prompts.

```bash
# Activate the training environment
conda activate pointstream-training

# Run the training script on your image data
python train.py --data_path ./data --epochs 200 --max_images 1000

# Deactivate when done
conda deactivate
```

After training, copy the resulting `best.pt` file from `artifacts/training/train/weights/` to `pointstream/models/weights/yolo_sports_student.pt`. Then, add this new model to the `MODEL_REGISTRY` in `pointstream/config.py`.

### **2. Run the Server Pipeline**

(Requires the `pointstream` environment)

```bash
# Activate the main environment
conda activate pointstream

# Run the pipeline using the general-purpose pre-trained model
python run_server.py --input-video /path/to/your/video.mp4

# Or, run the pipeline using your custom-trained sports model
python run_server.py --input-video /path/to/your/video.mp4 --content-type sports
```


This will generate a `_final_results.json` file and all associated image artifacts in the `artifacts/pipeline_output/` directory.

### **3. Run the Client Reconstruction**

(Requires the `pointstream` environment)

```bash
# Activate the main environment
conda activate pointstream

# Reconstruct the video from the processed data
python run_client.py --input-json artifacts/pipeline_output/YOUR_VIDEO_final_results.json
```

This will generate a `_reconstructed.mp4` video (or a folder of scene videos) in your project root.

## **Current Status & Placeholders**

The end-to-end pipeline is functional, but certain advanced components are currently implemented as simplified placeholders to allow for stable development and testing. These are primary areas for future research and implementation.

* **Rigid Object Keypoint Extraction:** In `stage_04_foreground.py`, the `_extract_feature_keypoints` function is a placeholder that currently returns only the four corners of an object's bounding box.
* **Background Mosaicking:** In `stage_03_background.py`, for scenes with `SIMPLE` camera motion, the background is currently represented by the first frame of the scene.
* **Client-Side Generative Model:** The `Reconstructor` in `pointstream/client/` uses a simple affine warp to demonstrate the reconstruction principle. This is a placeholder for the final, trained generative model.

## **Ablation Studies & Research Directions**

This project is designed to facilitate several key research experiments:

* **Student Model Architecture:** Compare the performance of different YOLO architectures when trained on the same auto-labeled dataset.
* **Specialized vs. Generalist Models:** Evaluate the trade-offs between using a single, general-purpose model versus a suite of specialized models.
* **Motion Representation for Rigid Objects:** Compare classical feature tracking against a custom, jointly trained keypoint extractor and object reconstructor.
* **Appearance Representation:** Investigate whether a learned appearance vector can achieve a better quality-to-bitrate ratio than transmitting a raw image patch.
* **End-to-End Codec Comparison:** Generate Rate-Distortion (RD) curves for the complete Pointstream system and compare them against traditional codecs like AV1.

