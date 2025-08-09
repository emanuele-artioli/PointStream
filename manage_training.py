#!/usr/bin/env python3
"""
Training management script for PointStream.

This script helps manage training processes, check cache status, and restart 
training with the enhanced caching system.
"""
import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional


def find_training_processes() -> List[dict]:
    """Find running training processes."""
    try:
        # Use ps to find Python processes that might be training
        result = subprocess.run([
            'ps', 'aux'
        ], capture_output=True, text=True, check=True)
        
        processes = []
        for line in result.stdout.split('\n'):
            if 'python' in line.lower() and any(keyword in line.lower() for keyword in ['train', 'yolo']):
                parts = line.split()
                if len(parts) >= 11:
                    processes.append({
                        'pid': int(parts[1]),
                        'cpu': parts[2],
                        'mem': parts[3],
                        'command': ' '.join(parts[10:])
                    })
        
        return processes
    except subprocess.CalledProcessError:
        return []


def stop_training_process(pid: int, force: bool = False) -> bool:
    """Stop a training process gracefully or forcefully."""
    try:
        if force:
            os.kill(pid, signal.SIGKILL)
            print(f"Forcefully terminated process {pid}")
        else:
            os.kill(pid, signal.SIGTERM)
            print(f"Sent termination signal to process {pid}")
            
            # Wait a bit for graceful shutdown
            time.sleep(5)
            
            # Check if process is still running
            try:
                os.kill(pid, 0)  # This doesn't actually kill, just checks if process exists
                print(f"Process {pid} is still running. Use --force to terminate forcefully.")
                return False
            except OSError:
                print(f"Process {pid} terminated gracefully")
                
        return True
    except OSError as e:
        print(f"Error terminating process {pid}: {e}")
        return False


def get_training_status() -> dict:
    """Get the current training status."""
    training_dir = Path("artifacts/training")
    
    status = {
        "processes_running": len(find_training_processes()),
        "training_dir_exists": training_dir.exists(),
        "datasets": [],
        "models": []
    }
    
    if training_dir.exists():
        # Find datasets
        for item in training_dir.iterdir():
            if item.is_dir() and item.name.startswith("dataset"):
                status["datasets"].append(str(item))
        
        # Find model directories
        for item in training_dir.iterdir():
            if item.is_dir() and (item / "weights").exists():
                weights_dir = item / "weights"
                if (weights_dir / "best.pt").exists():
                    status["models"].append({
                        "name": item.name,
                        "path": str(weights_dir / "best.pt"),
                        "size_mb": (weights_dir / "best.pt").stat().st_size / (1024 * 1024)
                    })
    
    return status


def show_cache_status():
    """Show annotation cache status."""
    try:
        sys.path.append(str(Path(__file__).parent.parent))
        from pointstream.utils.training_utils import get_cache_size, check_cache_health
        from pointstream import config
        
        cache_dir = config.ANNOTATIONS_CACHE_DIR
        
        print("\n" + "="*50)
        print("ANNOTATION CACHE STATUS")
        print("="*50)
        
        if not cache_dir.exists():
            print("Cache directory does not exist yet.")
            print(f"Will be created at: {cache_dir}")
            return
        
        cache_health = check_cache_health()
        print(f"Cache Directory: {cache_dir}")
        print(f"Cache Size: {cache_health['size_gb']:.2f} GB")
        print(f"Max Size: {cache_health['max_size_gb']} GB")
        print(f"Usage: {cache_health['size_percentage']:.1f}%")
        print(f"Status: {cache_health['status'].upper()}")
        print(f"Recommendation: {cache_health['recommendation']}")
        
        # Count files by content type
        print(f"\nCache Contents:")
        content_dirs = list((cache_dir / "content_types").glob("*")) if (cache_dir / "content_types").exists() else []
        
        if content_dirs:
            for content_dir in content_dirs:
                if content_dir.is_dir():
                    file_count = len(list(content_dir.glob("*.json")))
                    print(f"  {content_dir.name}: {file_count} annotations")
        else:
            print("  No cached annotations found")
        
        print("="*50)
        
    except ImportError as e:
        print(f"Error importing training utilities: {e}")
        print("Make sure you're running this from the PointStream directory")


def main():
    """Main function for training management."""
    parser = argparse.ArgumentParser(description="Manage PointStream training processes")
    parser.add_argument("--status", action="store_true", help="Show current training status")
    parser.add_argument("--stop", type=int, metavar="PID", help="Stop training process by PID")
    parser.add_argument("--stop-all", action="store_true", help="Stop all training processes")
    parser.add_argument("--force", action="store_true", help="Force stop processes (use with --stop or --stop-all)")
    parser.add_argument("--cache-status", action="store_true", help="Show annotation cache status")
    parser.add_argument("--restart-enhanced", type=str, metavar="CONTENT_TYPE", 
                       help="Stop current training and restart with enhanced caching for specified content type")
    parser.add_argument("--data-path", type=str, help="Data path for restart (required with --restart-enhanced)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs for restart")
    
    args = parser.parse_args()
    
    if args.status or (not any([args.stop, args.stop_all, args.cache_status, args.restart_enhanced])):
        # Show status by default
        print("POINTSTREAM TRAINING STATUS")
        print("="*40)
        
        processes = find_training_processes()
        if processes:
            print(f"Found {len(processes)} training process(es):")
            for proc in processes:
                print(f"  PID: {proc['pid']}, CPU: {proc['cpu']}%, Memory: {proc['mem']}%")
                print(f"  Command: {proc['command'][:80]}...")
        else:
            print("No training processes found.")
        
        status = get_training_status()
        print(f"\nTraining Directory: {'exists' if status['training_dir_exists'] else 'does not exist'}")
        print(f"Datasets found: {len(status['datasets'])}")
        print(f"Trained models: {len(status['models'])}")
        
        if status['models']:
            print("\nTrained Models:")
            for model in status['models']:
                print(f"  {model['name']}: {model['size_mb']:.1f} MB ({model['path']})")
    
    if args.cache_status:
        show_cache_status()
    
    if args.stop:
        print(f"Stopping process {args.stop}...")
        stop_training_process(args.stop, args.force)
    
    if args.stop_all:
        processes = find_training_processes()
        if processes:
            print(f"Stopping {len(processes)} training process(es)...")
            for proc in processes:
                stop_training_process(proc['pid'], args.force)
        else:
            print("No training processes to stop.")
    
    if args.restart_enhanced:
        if not args.data_path:
            print("Error: --data-path is required when using --restart-enhanced")
            sys.exit(1)
        
        # Stop all current training processes
        processes = find_training_processes()
        if processes:
            print("Stopping current training processes...")
            for proc in processes:
                stop_training_process(proc['pid'], force=True)
            time.sleep(2)  # Wait for processes to stop
        
        # Start enhanced training
        print(f"Starting enhanced training for content type: {args.restart_enhanced}")
        
        enhanced_script = Path(__file__).parent / "train_enhanced.py"
        if not enhanced_script.exists():
            print(f"Error: Enhanced training script not found at {enhanced_script}")
            sys.exit(1)
        
        cmd = [
            sys.executable, str(enhanced_script),
            "--content_type", args.restart_enhanced,
            "--data_path", args.data_path,
            "--epochs", str(args.epochs)
        ]
        
        print(f"Running: {' '.join(cmd)}")
        
        try:
            # Start the new training process
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error starting enhanced training: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
