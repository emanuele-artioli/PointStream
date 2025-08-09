#!/usr/bin/env python3
"""
Cache management and testing script for PointStream SAM annotations.

This script allows you to:
1. Initialize the annotation cache system
2. Test cache functionality 
3. Manage cached annotations by content type
4. Simulate cache operations without running full training
"""
import argparse
import json
import hashlib
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional


class AnnotationCacheManager:
    """Manages SAM annotation caching organized by content type."""
    
    def __init__(self, cache_root: Path = None):
        self.cache_root = cache_root or Path("artifacts/annotations_cache")
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.cache_root / "cache_metadata.json"
        self.load_metadata()
    
    def load_metadata(self):
        """Load cache metadata from disk."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {
                "version": "1.0",
                "created": datetime.now().isoformat(),
                "content_types": {},
                "cache_stats": {
                    "total_size_bytes": 0,
                    "total_annotations": 0,
                    "last_cleanup": None
                }
            }
            self.save_metadata()
    
    def save_metadata(self):
        """Save cache metadata to disk."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def initialize_content_type(self, content_type: str, description: str = None):
        """Initialize a new content type in the cache."""
        content_dir = self.cache_root / "content_types" / content_type
        content_dir.mkdir(parents=True, exist_ok=True)
        
        if content_type not in self.metadata["content_types"]:
            self.metadata["content_types"][content_type] = {
                "created": datetime.now().isoformat(),
                "description": description or f"Annotations for {content_type} content",
                "annotation_count": 0,
                "total_size_bytes": 0,
                "last_updated": datetime.now().isoformat(),
                "cache_directory": str(content_dir)
            }
            self.save_metadata()
            print(f"✅ Initialized content type: {content_type}")
        else:
            print(f"ℹ️  Content type {content_type} already exists")
        
        return content_dir
    
    def get_content_type_stats(self, content_type: str = None) -> Dict:
        """Get statistics for a content type or all content types."""
        if content_type:
            if content_type in self.metadata["content_types"]:
                return self.metadata["content_types"][content_type]
            else:
                return {"error": f"Content type {content_type} not found"}
        else:
            # Return stats for all content types
            return {
                "total_content_types": len(self.metadata["content_types"]),
                "content_types": self.metadata["content_types"],
                "cache_stats": self.metadata["cache_stats"]
            }
    
    def simulate_annotation_cache(self, content_type: str, num_images: int = 10):
        """Simulate caching annotations for testing purposes."""
        content_dir = self.initialize_content_type(content_type)
        
        # Create a timestamp-based annotation set
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        annotation_set_dir = content_dir / f"simulated_{timestamp}"
        annotation_set_dir.mkdir(parents=True, exist_ok=True)
        
        # Create mock annotation files
        total_size = 0
        for i in range(num_images):
            # Create mock annotation file
            annotation_data = {
                "image_id": f"mock_image_{i:03d}",
                "annotations": [
                    {
                        "class": "mock_object",
                        "bbox": [100, 100, 200, 200],
                        "confidence": 0.95
                    }
                ],
                "created": datetime.now().isoformat(),
                "content_type": content_type
            }
            
            annotation_file = annotation_set_dir / f"annotation_{i:03d}.json"
            with open(annotation_file, 'w') as f:
                json.dump(annotation_data, f, indent=2)
            
            total_size += annotation_file.stat().st_size
        
        # Create mock dataset structure
        (annotation_set_dir / "train" / "images").mkdir(parents=True, exist_ok=True)
        (annotation_set_dir / "train" / "labels").mkdir(parents=True, exist_ok=True)
        (annotation_set_dir / "val" / "images").mkdir(parents=True, exist_ok=True)
        (annotation_set_dir / "val" / "labels").mkdir(parents=True, exist_ok=True)
        
        # Create mock data.yaml
        data_yaml = {
            "path": str(annotation_set_dir.absolute()),
            "train": "train/images",
            "val": "val/images",
            "names": {0: "mock_object"},
            "nc": 1
        }
        
        with open(annotation_set_dir / "data.yaml", 'w') as f:
            json.dump(data_yaml, f, indent=2)
        
        # Update metadata
        self.metadata["content_types"][content_type]["annotation_count"] += num_images
        self.metadata["content_types"][content_type]["total_size_bytes"] += total_size
        self.metadata["content_types"][content_type]["last_updated"] = datetime.now().isoformat()
        self.metadata["cache_stats"]["total_annotations"] += num_images
        self.metadata["cache_stats"]["total_size_bytes"] += total_size
        
        self.save_metadata()
        
        print(f"✅ Simulated {num_images} annotations for {content_type}")
        print(f"   📁 Created in: {annotation_set_dir}")
        print(f"   📊 Total size: {total_size / 1024:.1f} KB")
        
        return annotation_set_dir
    
    def list_cached_annotations(self, content_type: str = None):
        """List all cached annotation sets."""
        if content_type:
            content_types = [content_type] if content_type in self.metadata["content_types"] else []
        else:
            content_types = list(self.metadata["content_types"].keys())
        
        for ct in content_types:
            print(f"\n📂 Content Type: {ct}")
            content_info = self.metadata["content_types"][ct]
            print(f"   Description: {content_info['description']}")
            print(f"   Annotations: {content_info['annotation_count']}")
            print(f"   Size: {content_info['total_size_bytes'] / 1024:.1f} KB")
            print(f"   Last Updated: {content_info['last_updated']}")
            
            # List annotation sets in this content type
            content_dir = Path(content_info['cache_directory'])
            if content_dir.exists():
                annotation_sets = [d for d in content_dir.iterdir() if d.is_dir()]
                if annotation_sets:
                    print(f"   Annotation Sets:")
                    for ann_set in sorted(annotation_sets):
                        file_count = len(list(ann_set.glob("*.json")))
                        print(f"     - {ann_set.name}: {file_count} files")
                else:
                    print(f"     No annotation sets found")
    
    def cleanup_old_annotations(self, older_than_days: int = 30):
        """Remove annotation sets older than specified days."""
        cutoff_date = datetime.now() - timedelta(days=older_than_days)
        removed_count = 0
        
        for content_type in self.metadata["content_types"]:
            content_dir = Path(self.metadata["content_types"][content_type]["cache_directory"])
            
            if content_dir.exists():
                for annotation_set in content_dir.iterdir():
                    if annotation_set.is_dir():
                        # Check if annotation set is old
                        set_time = datetime.fromtimestamp(annotation_set.stat().st_mtime)
                        if set_time < cutoff_date:
                            print(f"🗑️  Removing old annotation set: {annotation_set}")
                            shutil.rmtree(annotation_set)
                            removed_count += 1
        
        # Update metadata
        self.metadata["cache_stats"]["last_cleanup"] = datetime.now().isoformat()
        self.save_metadata()
        
        print(f"✅ Removed {removed_count} old annotation sets")
        return removed_count
    
    def get_cache_size(self) -> Dict:
        """Calculate actual cache size on disk."""
        total_size = 0
        file_count = 0
        
        for file_path in self.cache_root.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
                file_count += 1
        
        return {
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "total_size_gb": total_size / (1024 * 1024 * 1024),
            "file_count": file_count
        }


def main():
    """Main function for cache management."""
    parser = argparse.ArgumentParser(description="Manage PointStream annotation cache")
    parser.add_argument("--init", type=str, help="Initialize a new content type")
    parser.add_argument("--description", type=str, help="Description for new content type")
    parser.add_argument("--simulate", type=str, help="Simulate annotations for content type")
    parser.add_argument("--num-images", type=int, default=10, help="Number of images to simulate")
    parser.add_argument("--list", type=str, nargs='?', const='all', help="List cached annotations (all or specific content type)")
    parser.add_argument("--stats", action="store_true", help="Show cache statistics")
    parser.add_argument("--cleanup", type=int, metavar="DAYS", help="Clean up annotations older than N days")
    parser.add_argument("--size", action="store_true", help="Calculate actual cache size")
    
    args = parser.parse_args()
    
    cache_manager = AnnotationCacheManager()
    
    if args.init:
        cache_manager.initialize_content_type(args.init, args.description)
    
    if args.simulate:
        cache_manager.simulate_annotation_cache(args.simulate, args.num_images)
    
    if args.list:
        content_type = None if args.list == 'all' else args.list
        cache_manager.list_cached_annotations(content_type)
    
    if args.stats:
        print("\n📊 CACHE STATISTICS")
        print("=" * 50)
        stats = cache_manager.get_content_type_stats()
        print(f"Total Content Types: {stats['total_content_types']}")
        print(f"Total Annotations: {stats['cache_stats']['total_annotations']}")
        print(f"Total Size: {stats['cache_stats']['total_size_bytes'] / 1024:.1f} KB")
        
        if stats['cache_stats']['last_cleanup']:
            print(f"Last Cleanup: {stats['cache_stats']['last_cleanup']}")
        else:
            print("Last Cleanup: Never")
    
    if args.cleanup:
        cache_manager.cleanup_old_annotations(args.cleanup)
    
    if args.size:
        print("\n💾 ACTUAL CACHE SIZE")
        print("=" * 30)
        size_info = cache_manager.get_cache_size()
        print(f"Files: {size_info['file_count']}")
        print(f"Size: {size_info['total_size_mb']:.2f} MB ({size_info['total_size_bytes']} bytes)")
    
    # Show basic info if no specific action requested
    if not any([args.init, args.simulate, args.list, args.stats, args.cleanup, args.size]):
        print("🎯 POINTSTREAM ANNOTATION CACHE MANAGER")
        print("=" * 45)
        
        cache_root = Path("artifacts/annotations_cache")
        if cache_root.exists():
            print(f"Cache Root: {cache_root}")
            stats = cache_manager.get_content_type_stats()
            print(f"Content Types: {stats['total_content_types']}")
            print(f"Total Annotations: {stats['cache_stats']['total_annotations']}")
            
            if stats['total_content_types'] > 0:
                print("\nAvailable Content Types:")
                for ct_name, ct_info in stats['content_types'].items():
                    print(f"  📂 {ct_name}: {ct_info['annotation_count']} annotations")
        else:
            print("Cache not initialized yet.")
        
        print("\nUsage Examples:")
        print("  python3 cache_manager.py --init sports --description 'Sports annotations'")
        print("  python3 cache_manager.py --simulate sports --num-images 20")
        print("  python3 cache_manager.py --list all")
        print("  python3 cache_manager.py --stats")


if __name__ == "__main__":
    main()
