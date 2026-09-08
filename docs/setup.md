# PointStream Setup and Environment Guide

This guide describes how to configure external data storage, local caching, and dependencies for PointStream.

---

## 1. External Data Storage (`PS_DATA_ROOT`)

PointStream datasets (`assets/`) and run outputs (`outputs/`) hold hundreds of thousands of files. To prevent Git index bloating and editor file-watcher lockups on network filesystems (NFS), **data lives outside the tracked code repository**.

### Precedence Resolution
The runtime path resolver (`src/contracts/paths.py`) resolves `assets/` and `outputs/` using this strict precedence:

1. **Environment Variable**: `PS_DATA_ROOT` (if set and non-empty).
2. **Marker File**: `.ps-data-root` at the repository root. A plain-text, one-line file containing the absolute path to the data root (unquoted). This marker is gitignored and stays local to each checkout or worktree.
3. **Fallback**: The repository root itself (historical default).

### Inspecting Paths
Verify your active data paths with:
```bash
python -c "from src.contracts.paths import describe; print(describe())"
```

### Setting up a Worktree Marker
When creating a new Git worktree, link it to the shared host data directory by creating a `.ps-data-root` file:
```bash
echo "/path/to/shared/pointstream-data" > .ps-data-root
```
> [!WARNING]
> Never create symlinks named `assets` or `outputs` inside the repository tree. File indexers and editors will follow them, defeating the isolation.

---

## 2. Dependencies and Tooling

### Conda Environment
PointStream runs under a pinned Conda environment (typically Python 3.10):
```bash
conda env create -f environment.yaml
conda activate pointstream
```
*Note for shared server installations*: Do not run arbitrary `pip install` commands that can mutate pinned dependencies.

### Native Codec Binaries
Full evaluation requires native video encoders and filters:
- **FFmpeg**: Compiled with `libvmaf`, `libsvtav1`, and `libaom`. Check with:
  ```bash
  ffmpeg -hide_banner -filters | grep libvmaf
  ```
- **VVC Intra**: `vvencapp` / `libvvenc` for VVC background plate encoding:
  ```bash
  which vvencapp
  ```
- **OpenCV**: Needs native WebP image encoding support (`cv2.IMWRITE_WEBP_QUALITY`).

---

## 3. Host-Local Caches

On distributed or NFS filesystems, serial metadata reads for caches impose high latency taxes. Keep all regenerable caches on local host disks (such as `/tmp` or `/var/tmp`), namespaced by checkout:

```bash
export MYPY_CACHE_DIR="/tmp/mypy-$(basename "$PWD")"
export PYTEST_CACHE_DIR="/tmp/pytest-$(basename "$PWD")"
export RUFF_CACHE_DIR="/tmp/ruff-$(basename "$PWD")"
```

Always import `sqlite3` before `torch` in scripts and modules to avoid dynamic linker CXXABI collisions.
