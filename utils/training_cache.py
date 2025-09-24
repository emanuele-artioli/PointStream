#!/usr/bin/env python3
"""
Training Data Cache Manager

This module handles caching of processed training data to avoid reprocessing
the same videos multiple times. It supports incremental dataset updates and
efficient model retraining with accumulated data.
"""

import json
import hashlib
import os
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Any, Set, Optional, Tuple
from datetime import datetime
import tempfile

from utils import config


class TrainingDataCache:
    """
    Manages caching of processed training data from videos.
    
    Features:
    - Tracks processed videos by hash to avoid reprocessing
    - Stores extracted metadata and object data
    - Supports incremental updates to training datasets
    - Enables fast model retraining on accumulated data
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the training data cache.
        
        Args:
            cache_dir: Directory to store cache data. If None, uses config default.
        """
        self.cache_dir = Path(cache_dir or config.get_str('cache', 'training_cache_dir', './training_cache'))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache structure:
        # cache_dir/
        #   ├── index.json          # Master index of all cached videos
        #   ├── videos/             # Individual video cache files
        #   │   ├── {video_hash}.json
        #   │   └── ...
        #   └── metadata/           # Compressed metadata files
        #       ├── {video_hash}/
        #       │   ├── scene_0001_metadata.pzm
        #       │   └── ...
        #       └── ...
        
        self.videos_dir = self.cache_dir / "videos"
        self.metadata_dir = self.cache_dir / "metadata"
        self.videos_dir.mkdir(exist_ok=True)
        self.metadata_dir.mkdir(exist_ok=True)
        
        self.index_file = self.cache_dir / "index.json"
        self.index = self._load_index()
        
        logging.info(f"Training cache initialized: {self.cache_dir}")
        logging.info(f"Cached videos: {len(self.index.get('videos', {}))}")
    
    def _load_index(self) -> Dict[str, Any]:
        """Load the cache index from disk."""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r') as f:
                    index = json.load(f)
                    # Ensure required structure
                    if 'videos' not in index:
                        index['videos'] = {}
                    if 'metadata' not in index:
                        index['metadata'] = {
                            'created': datetime.now().isoformat(),
                            'last_updated': datetime.now().isoformat(),
                            'total_videos': 0,
                            'total_objects': 0
                        }
                    return index
            except Exception as e:
                logging.warning(f"Failed to load cache index: {e}")
        
        # Create new index
        return {
            'videos': {},
            'metadata': {
                'created': datetime.now().isoformat(),
                'last_updated': datetime.now().isoformat(),
                'total_videos': 0,
                'total_objects': 0
            }
        }
    
    def _save_index(self):
        """Save the cache index to disk."""
        self.index['metadata']['last_updated'] = datetime.now().isoformat()
        with open(self.index_file, 'w') as f:
            json.dump(self.index, f, indent=2)
    
    def _get_video_hash(self, video_path: str) -> str:
        """
        Generate a unique hash for a video file.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            SHA256 hash of the video file
        """
        hash_sha256 = hashlib.sha256()
        with open(video_path, "rb") as f:
            # Read file in chunks to handle large videos
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def is_video_cached(self, video_path: str) -> bool:
        """
        Check if a video has already been processed and cached.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            True if video is cached, False otherwise
        """
        try:
            video_hash = self._get_video_hash(video_path)
            return video_hash in self.index['videos']
        except Exception as e:
            logging.error(f"Failed to check cache for {video_path}: {e}")
            return False
    
    def get_cached_videos(self) -> List[str]:
        """
        Get list of all cached video hashes.
        
        Returns:
            List of video hashes that are cached
        """
        return list(self.index['videos'].keys())
    
    def add_video_to_cache(self, video_path: str, metadata_dir: str, 
                          processing_stats: Dict[str, Any] = None) -> str:
        """
        Add a processed video to the cache.
        
        Args:
            video_path: Path to the original video file
            metadata_dir: Directory containing the processed metadata
            processing_stats: Optional statistics from processing
            
        Returns:
            The video hash used for caching
        """
        try:
            video_hash = self._get_video_hash(video_path)
            video_name = Path(video_path).name
            
            # Copy metadata files to cache
            cache_metadata_dir = self.metadata_dir / video_hash
            cache_metadata_dir.mkdir(exist_ok=True)
            
            metadata_path = Path(metadata_dir)
            if metadata_path.exists():
                # Copy all .pzm files
                for pzm_file in metadata_path.glob("*.pzm"):
                    shutil.copy2(pzm_file, cache_metadata_dir)
                
                # Copy any .json files
                for json_file in metadata_path.glob("*.json"):
                    shutil.copy2(json_file, cache_metadata_dir)
                
                # Copy the objects directory (essential for training)
                objects_dir = metadata_path / "objects"
                if objects_dir.exists():
                    cache_objects_dir = cache_metadata_dir / "objects"
                    shutil.copytree(objects_dir, cache_objects_dir, dirs_exist_ok=True)
            
            # Create video cache entry
            video_info = {
                'video_path': str(video_path),
                'video_name': video_name,
                'video_hash': video_hash,
                'cached_at': datetime.now().isoformat(),
                'metadata_dir': str(cache_metadata_dir),
                'processing_stats': processing_stats or {},
                'file_size': os.path.getsize(video_path),
                'metadata_files': [f.name for f in cache_metadata_dir.glob("*")]
            }
            
            # Save individual video cache file
            cache_file = self.videos_dir / f"{video_hash}.json"
            with open(cache_file, 'w') as f:
                json.dump(video_info, f, indent=2)
            
            # Update index
            self.index['videos'][video_hash] = video_info
            self.index['metadata']['total_videos'] = len(self.index['videos'])
            self._save_index()
            
            logging.info(f"Added video to cache: {video_name} -> {video_hash[:8]}")
            return video_hash
            
        except Exception as e:
            logging.error(f"Failed to cache video {video_path}: {e}")
            raise
    
    def get_video_metadata_dir(self, video_path: str) -> Optional[str]:
        """
        Get the cached metadata directory for a video.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Path to cached metadata directory, or None if not cached
        """
        try:
            video_hash = self._get_video_hash(video_path)
            if video_hash in self.index['videos']:
                return self.index['videos'][video_hash]['metadata_dir']
            return None
        except Exception as e:
            logging.error(f"Failed to get cached metadata for {video_path}: {e}")
            return None
    
    def get_uncached_videos(self, video_paths: List[str]) -> List[str]:
        """
        Filter a list of videos to return only those not yet cached.
        
        Args:
            video_paths: List of video file paths
            
        Returns:
            List of video paths that are not cached
        """
        uncached = []
        for video_path in video_paths:
            if not self.is_video_cached(video_path):
                uncached.append(video_path)
        return uncached
    
    def collect_all_training_data(self) -> Tuple[str, Dict[str, Any]]:
        """
        Collect all cached training data into a temporary directory.
        
        Returns:
            Tuple of (temp_directory_path, statistics)
        """
        # Create temporary directory for merged data
        temp_dir = tempfile.mkdtemp(prefix="pointstream_training_")
        temp_path = Path(temp_dir)
        
        stats = {
            'total_videos': 0,
            'total_metadata_files': 0,
            'videos_processed': []
        }
        
        # Copy all cached metadata to temp directory
        for video_hash, video_info in self.index['videos'].items():
            source_metadata_dir = Path(video_info['metadata_dir'])
            
            if source_metadata_dir.exists():
                # Copy metadata files
                for metadata_file in source_metadata_dir.glob("*"):
                    if metadata_file.is_file():  # Only copy files, not directories
                        dest_file = temp_path / metadata_file.name
                        
                        # Handle filename conflicts by prefixing with video hash
                        if dest_file.exists():
                            dest_file = temp_path / f"{video_hash[:8]}_{metadata_file.name}"
                        
                        shutil.copy2(metadata_file, dest_file)
                        stats['total_metadata_files'] += 1
                
                # Copy objects directory if it exists
                source_objects_dir = source_metadata_dir / "objects"
                if source_objects_dir.exists():
                    dest_objects_dir = temp_path / "objects"
                    dest_objects_dir.mkdir(exist_ok=True)
                    
                    # Copy all scene directories from this video
                    for scene_dir in source_objects_dir.iterdir():
                        if scene_dir.is_dir():
                            dest_scene_dir = dest_objects_dir / scene_dir.name
                            
                            # Handle scene directory conflicts by prefixing with video hash
                            if dest_scene_dir.exists():
                                dest_scene_dir = dest_objects_dir / f"{video_hash[:8]}_{scene_dir.name}"
                            
                            shutil.copytree(scene_dir, dest_scene_dir, dirs_exist_ok=True)
                
                stats['videos_processed'].append(video_info['video_name'])
                stats['total_videos'] += 1
        
        logging.info(f"Collected training data from {stats['total_videos']} cached videos")
        logging.info(f"Temporary training directory: {temp_dir}")
        
        return temp_dir, stats
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current cache.
        
        Returns:
            Dictionary with cache statistics
        """
        total_size = 0
        metadata_files = 0
        
        for video_hash, video_info in self.index['videos'].items():
            total_size += video_info.get('file_size', 0)
            metadata_files += len(video_info.get('metadata_files', []))
        
        return {
            'total_videos': len(self.index['videos']),
            'total_size_bytes': total_size,
            'total_size_mb': total_size / (1024 * 1024),
            'total_metadata_files': metadata_files,
            'cache_directory': str(self.cache_dir),
            'created': self.index['metadata']['created'],
            'last_updated': self.index['metadata']['last_updated']
        }
    
    def clear_cache(self):
        """
        Clear all cached data (use with caution).
        """
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.videos_dir.mkdir(exist_ok=True)
            self.metadata_dir.mkdir(exist_ok=True)
        
        self.index = self._load_index()
        logging.info("Training cache cleared")
    
    def remove_video_from_cache(self, video_path: str) -> bool:
        """
        Remove a specific video from the cache.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            True if video was removed, False if not found
        """
        try:
            video_hash = self._get_video_hash(video_path)
            
            if video_hash not in self.index['videos']:
                return False
            
            # Remove metadata directory
            cache_metadata_dir = self.metadata_dir / video_hash
            if cache_metadata_dir.exists():
                shutil.rmtree(cache_metadata_dir)
            
            # Remove video cache file
            cache_file = self.videos_dir / f"{video_hash}.json"
            if cache_file.exists():
                cache_file.unlink()
            
            # Update index
            del self.index['videos'][video_hash]
            self.index['metadata']['total_videos'] = len(self.index['videos'])
            self._save_index()
            
            logging.info(f"Removed video from cache: {video_hash[:8]}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to remove video from cache {video_path}: {e}")
            return False
