#!/usr/bin/env python3
"""
Cleanup script for PointStream runtime and experiment files.
Removes temporary files generated during pipeline runs while preserving important data.
"""

import os
import shutil
import glob
from pathlib import Path

def cleanup_runtime_files(dry_run=False):
    """
    Clean up runtime and experiment files.
    
    Args:
        dry_run (bool): If True, only print what would be deleted without actually deleting
    """
    base_dir = Path(__file__).parent
    
    # Files and directories to clean up
    cleanup_patterns = [
        # Evaluation results directories (keeping only final ones in artifacts)
        "evaluation_results_*",
        "final_results_with_working_metrics",
        "final_evaluation_summary",
        
        # Reconstructed scene directories
        "*_reconstructed_scenes",
        
        # Test videos (except ones in tests/data)
        "simple_test_video.mp4",
        "test_video.mp4",
        
        # Temporary YOLO weights (keeping original in weights or downloading fresh)
        "yolo11n.pt",
        
        # Any temporary log files
        "*.log",
        
        # Python cache
        "__pycache__",
        "*.pyc",
        "*.pyo",
    ]
    
    preserved_paths = [
        "tests/data",
        "artifacts",
        "pointstream",
        ".git"
    ]
    
    deleted_items = []
    
    for pattern in cleanup_patterns:
        matches = glob.glob(str(base_dir / pattern), recursive=True)
        for match in matches:
            match_path = Path(match)
            
            # Check if this path should be preserved
            should_preserve = False
            for preserved in preserved_paths:
                if preserved in str(match_path.relative_to(base_dir)):
                    should_preserve = True
                    break
            
            if should_preserve:
                continue
                
            if dry_run:
                print(f"Would delete: {match_path}")
            else:
                try:
                    if match_path.is_dir():
                        shutil.rmtree(match_path)
                        print(f"Deleted directory: {match_path}")
                    else:
                        match_path.unlink()
                        print(f"Deleted file: {match_path}")
                    deleted_items.append(str(match_path))
                except Exception as e:
                    print(f"Error deleting {match_path}: {e}")
    
    # Clean up Python cache recursively
    for pycache_dir in base_dir.glob("**/__pycache__"):
        if not any(preserved in str(pycache_dir) for preserved in preserved_paths):
            if dry_run:
                print(f"Would delete: {pycache_dir}")
            else:
                try:
                    shutil.rmtree(pycache_dir)
                    print(f"Deleted cache directory: {pycache_dir}")
                    deleted_items.append(str(pycache_dir))
                except Exception as e:
                    print(f"Error deleting {pycache_dir}: {e}")
    
    return deleted_items

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Clean up PointStream runtime files")
    parser.add_argument("--dry-run", action="store_true", 
                       help="Show what would be deleted without actually deleting")
    parser.add_argument("--force", action="store_true",
                       help="Delete files without confirmation")
    
    args = parser.parse_args()
    
    if args.dry_run:
        print("DRY RUN - showing what would be deleted:")
        cleanup_runtime_files(dry_run=True)
    else:
        if not args.force:
            response = input("This will delete runtime and experiment files. Continue? (y/N): ")
            if response.lower() != 'y':
                print("Cleanup cancelled.")
                exit(0)
        
        print("Cleaning up runtime files...")
        deleted = cleanup_runtime_files(dry_run=False)
        print(f"\nCleanup complete. Deleted {len(deleted)} items.")
