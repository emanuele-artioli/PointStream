"""Download sample dataset shard from Hugging Face builddotai/Egocentric-10K."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from huggingface_hub import hf_hub_download

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

REPO_ID = "builddotai/Egocentric-10K"
DEFAULT_TAR = "factory_001/workers/worker_001/factory001_worker001_part00.tar"
DEFAULT_OUT_DIR = Path("/home/itec/emanuele/Datasets/Egocentric-10K/raw")


def download_tar_shard(
    repo_id: str = REPO_ID,
    filename: str = DEFAULT_TAR,
    dest_dir: Path = DEFAULT_OUT_DIR,
) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    expected_path = dest_dir / Path(filename).name

    if expected_path.exists() and expected_path.stat().st_size > 0:
        logger.info(f"Archive already exists at {expected_path} ({expected_path.stat().st_size} bytes)")
        return expected_path

    logger.info(f"Downloading {filename} from {repo_id} to {dest_dir}...")
    downloaded_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        local_dir=dest_dir,
    )
    final_path = Path(downloaded_path)
    logger.info(f"Successfully downloaded to {final_path} ({final_path.stat().st_size} bytes)")
    return final_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download sample shard from Egocentric-10K")
    parser.add_argument("--repo-id", default=REPO_ID, help="Hugging Face repo ID")
    parser.add_argument("--filename", default=DEFAULT_TAR, help="Path inside dataset repo")
    parser.add_argument("--dest-dir", type=Path, default=DEFAULT_OUT_DIR, help="Destination directory")
    args = parser.parse_args()

    download_tar_shard(repo_id=args.repo_id, filename=args.filename, dest_dir=args.dest_dir)


if __name__ == "__main__":
    main()

