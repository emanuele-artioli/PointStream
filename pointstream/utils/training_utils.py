"""
Utility functions for managing annotation cache and content-type-based model selection.
"""
import json
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

# Import config from the package
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from pointstream import config


def get_model_for_content_type(content_type: str) -> str:
    """
    Get the appropriate model path for a given content type.
    
    Args:
        content_type: The content type (e.g., 'sports', 'dance', 'automotive')
        
    Returns:
        Path to the model file
        
    Raises:
        ValueError: If content type is not supported
    """
    if content_type not in config.MODEL_REGISTRY:
        available_types = list(config.MODEL_REGISTRY.keys())
        raise ValueError(f"Unsupported content type '{content_type}'. Available types: {available_types}")
    
    model_path = config.MODEL_REGISTRY[content_type]
    
    # Check if the model file exists
    if not Path(model_path).exists() and content_type != "general":
        print(f"Warning: Model for content type '{content_type}' not found at {model_path}")
        print("Falling back to general model...")
        return config.MODEL_REGISTRY["general"]
    
    return model_path


def get_content_type_info(content_type: str) -> Dict:
    """
    Get information about a content type including its ontology.
    
    Args:
        content_type: The content type to get info for
        
    Returns:
        Dictionary with content type information
    """
    if content_type not in config.CONTENT_TYPE_ONTOLOGIES:
        return {
            "classes": ["unknown"],
            "description": f"Unknown content type: {content_type}",
            "supported": False
        }
    
    info = config.CONTENT_TYPE_ONTOLOGIES[content_type].copy()
    info["supported"] = True
    info["model_available"] = Path(config.MODEL_REGISTRY.get(content_type, "")).exists()
    
    return info


def list_available_content_types() -> List[Dict]:
    """
    List all available content types with their information.
    
    Returns:
        List of content type information dictionaries
    """
    content_types = []
    
    for content_type in config.MODEL_REGISTRY.keys():
        info = get_content_type_info(content_type)
        info["name"] = content_type
        info["model_path"] = config.MODEL_REGISTRY[content_type]
        content_types.append(info)
    
    return content_types


def get_cache_size() -> float:
    """
    Get the total size of the annotation cache in GB.
    
    Returns:
        Cache size in GB
    """
    cache_dir = config.ANNOTATIONS_CACHE_DIR
    if not cache_dir.exists():
        return 0.0
    
    total_size = 0
    for file_path in cache_dir.rglob('*'):
        if file_path.is_file():
            total_size += file_path.stat().st_size
    
    return total_size / (1024 ** 3)  # Convert to GB


def cleanup_old_cache_entries(older_than_days: int = None) -> int:
    """
    Remove cache entries older than the specified number of days.
    
    Args:
        older_than_days: Remove entries older than this many days
                        (uses config default if None)
    
    Returns:
        Number of entries removed
    """
    if older_than_days is None:
        older_than_days = config.CACHE_CONFIG["cleanup_older_than_days"]
    
    cache_dir = config.ANNOTATIONS_CACHE_DIR
    metadata_file = cache_dir / "cache_metadata.json"
    
    if not metadata_file.exists():
        return 0
    
    # Load metadata
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    # Calculate cutoff date
    cutoff_date = datetime.now() - timedelta(days=older_than_days)
    
    # Find entries to remove
    entries_to_remove = []
    for annotation_key, annotation_info in metadata["annotations"].items():
        created_date = datetime.fromisoformat(annotation_info["created"])
        if created_date < cutoff_date:
            entries_to_remove.append(annotation_key)
    
    # Remove old entries
    removed_count = 0
    for annotation_key in entries_to_remove:
        annotation_info = metadata["annotations"][annotation_key]
        annotation_path = Path(annotation_info["path"])
        
        # Remove the annotation file
        if annotation_path.exists():
            annotation_path.unlink()
            removed_count += 1
        
        # Remove from metadata
        del metadata["annotations"][annotation_key]
    
    # Update content type counts
    for content_type in metadata["content_types"]:
        metadata["content_types"][content_type]["annotation_count"] = sum(
            1 for ann in metadata["annotations"].values() 
            if ann["content_type"] == content_type
        )
    
    # Save updated metadata
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Removed {removed_count} old cache entries (older than {older_than_days} days)")
    return removed_count


def check_cache_health() -> Dict:
    """
    Check the health of the annotation cache and suggest cleanup if needed.
    
    Returns:
        Dictionary with cache health information
    """
    cache_size = get_cache_size()
    max_size = config.CACHE_CONFIG["max_size_gb"]
    
    health_info = {
        "size_gb": cache_size,
        "max_size_gb": max_size,
        "size_percentage": (cache_size / max_size) * 100 if max_size > 0 else 0,
        "needs_cleanup": cache_size > max_size,
        "status": "healthy"
    }
    
    if cache_size > max_size:
        health_info["status"] = "needs_cleanup"
        health_info["recommendation"] = f"Cache size ({cache_size:.2f} GB) exceeds maximum ({max_size} GB). Consider running cleanup."
    elif cache_size > max_size * 0.8:
        health_info["status"] = "warning"
        health_info["recommendation"] = f"Cache size ({cache_size:.2f} GB) is approaching maximum ({max_size} GB)."
    else:
        health_info["recommendation"] = "Cache is healthy."
    
    return health_info


def suggest_content_type(image_paths: List[Path]) -> str:
    """
    Suggest a content type based on image file names and paths.
    
    Args:
        image_paths: List of image file paths
        
    Returns:
        Suggested content type
    """
    # Simple heuristic based on common keywords in file names/paths
    path_text = " ".join([str(path).lower() for path in image_paths])
    
    content_type_keywords = {
        "sports": ["tennis", "basketball", "soccer", "football", "sport", "athlete", "game", "court", "field"],
        "dance": ["dance", "ballet", "choreography", "performance", "stage", "studio"],
        "automotive": ["car", "truck", "vehicle", "road", "highway", "auto", "motor", "driving"],
    }
    
    scores = {}
    for content_type, keywords in content_type_keywords.items():
        score = sum(1 for keyword in keywords if keyword in path_text)
        if score > 0:
            scores[content_type] = score
    
    if scores:
        suggested_type = max(scores, key=scores.get)
        confidence = scores[suggested_type] / len(image_paths)
        
        if confidence > 0.1:  # At least 10% of images match keywords
            return suggested_type
    
    return "general"


def print_training_summary(content_type: str, cache_stats: Dict, model_path: Path):
    """
    Print a summary of the training configuration and recommendations.
    
    Args:
        content_type: The content type being trained
        cache_stats: Cache statistics
        model_path: Path to the trained model
    """
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    
    print(f"Content Type: {content_type}")
    
    content_info = get_content_type_info(content_type)
    print(f"Classes: {', '.join(content_info['classes'])}")
    print(f"Description: {content_info['description']}")
    
    print(f"\nModel Location: {model_path}")
    
    print(f"\nCache Statistics:")
    print(f"  Total annotations: {cache_stats['total_annotations']}")
    print(f"  Content types: {cache_stats['content_types']}")
    
    if content_type in cache_stats['content_type_breakdown']:
        type_stats = cache_stats['content_type_breakdown'][content_type]
        print(f"  {content_type} annotations: {type_stats['count']}")
    
    cache_health = check_cache_health()
    print(f"\nCache Health: {cache_health['status'].upper()}")
    print(f"Cache Size: {cache_health['size_gb']:.2f} GB / {cache_health['max_size_gb']} GB")
    print(f"Recommendation: {cache_health['recommendation']}")
    
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("1. Update config.py MODEL_REGISTRY if this is a new content type")
    print("2. Test the model with: pointstream-server --content-type " + content_type)
    print("3. Consider running cache cleanup if needed: python -m pointstream.scripts.train_enhanced --clear-cache")
    print("="*60)
