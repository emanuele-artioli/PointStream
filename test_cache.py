#!/usr/bin/env python3
"""
Test script for the enhanced SAM annotation caching system.

This script tests the cache functionality without running actual YOLO training
to avoid interfering with other users on the shared server.
"""
import argparse
import os
import sys
from pathlib import Path
from datetime import datetime


def test_cache_integration():
    """Test the integration between training script and cache system."""
    print("🧪 TESTING SAM ANNOTATION CACHE INTEGRATION")
    print("=" * 50)
    
    # Test 1: Check if cache system is working
    cache_root = Path("artifacts/annotations_cache")
    if not cache_root.exists():
        print("❌ Cache directory not found")
        return False
    
    print("✅ Cache directory exists")
    
    # Test 2: Check metadata file
    metadata_file = cache_root / "cache_metadata.json"
    if not metadata_file.exists():
        print("❌ Cache metadata file not found")
        return False
    
    print("✅ Cache metadata file exists")
    
    # Test 3: Check content types
    content_types_dir = cache_root / "content_types"
    if not content_types_dir.exists():
        print("❌ Content types directory not found")
        return False
    
    content_types = [d.name for d in content_types_dir.iterdir() if d.is_dir()]
    print(f"✅ Found content types: {', '.join(content_types)}")
    
    # Test 4: Check if we can load cache manager
    try:
        import importlib.util
        import sys
        
        spec = importlib.util.spec_from_file_location("cache_manager", "cache_manager.py")
        cache_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cache_module)
        
        cache_manager = cache_module.AnnotationCacheManager()
        print("✅ Cache manager loaded successfully")
    except Exception as e:
        print(f"❌ Error loading cache manager: {e}")
        return False
    
    # Test 5: Check cache statistics
    try:
        stats = cache_manager.get_content_type_stats()
        print(f"✅ Cache stats: {stats['total_content_types']} content types, {stats['cache_stats']['total_annotations']} annotations")
    except Exception as e:
        print(f"❌ Error getting cache stats: {e}")
        return False
    
    return True


def test_training_workflow_simulation():
    """Simulate the enhanced training workflow without actually training."""
    print("\n🔄 TESTING ENHANCED TRAINING WORKFLOW")
    print("=" * 50)
    
    # Simulate content type selection
    content_type = "sports"
    print(f"📋 Selected content type: {content_type}")
    
    # Check if we have cached annotations
    cache_root = Path("artifacts/annotations_cache")
    content_cache_dir = cache_root / "content_types" / content_type
    
    if content_cache_dir.exists():
        annotation_sets = [d for d in content_cache_dir.iterdir() if d.is_dir()]
        if annotation_sets:
            latest_set = max(annotation_sets, key=lambda p: p.stat().st_mtime)
            print(f"✅ Found cached annotations: {latest_set}")
            
            # Check if it has the required structure
            data_yaml = latest_set / "data.yaml"
            if data_yaml.exists():
                print("✅ data.yaml found in cached annotations")
                print(f"   📁 Dataset path: {latest_set}")
                print("   🚀 Would skip annotation generation and use cached data")
                return True
            else:
                print("❌ data.yaml not found in cached annotations")
                return False
        else:
            print("ℹ️  No annotation sets found, would generate new annotations")
            return True
    else:
        print("ℹ️  Content type cache directory not found, would create new one")
        return True


def simulate_cache_workflow():
    """Simulate the complete cache workflow."""
    print("\n🎭 SIMULATING COMPLETE CACHE WORKFLOW")
    print("=" * 50)
    
    content_types = ["sports", "dance", "automotive"]
    
    for content_type in content_types:
        print(f"\n📂 Testing workflow for: {content_type}")
        
        # Simulate checking for cached annotations
        cache_root = Path("artifacts/annotations_cache")
        content_cache_dir = cache_root / "content_types" / content_type
        
        if content_cache_dir.exists():
            annotation_sets = [d for d in content_cache_dir.iterdir() if d.is_dir()]
            if annotation_sets:
                print(f"   ✅ Cache HIT: Found {len(annotation_sets)} annotation set(s)")
                for ann_set in annotation_sets:
                    file_count = len(list(ann_set.glob("*.json")))
                    print(f"      - {ann_set.name}: {file_count} annotations")
            else:
                print("   ⚠️  Cache MISS: No annotation sets found")
        else:
            print("   ⚠️  Cache MISS: Content type not initialized")
        
        # Simulate what would happen in training
        if content_cache_dir.exists() and any(content_cache_dir.iterdir()):
            print(f"   🚀 Would REUSE existing annotations for {content_type}")
            print(f"   ⏱️  Estimated time saved: ~15-30 minutes of SAM annotation")
        else:
            print(f"   🔄 Would GENERATE new annotations for {content_type}")
            print(f"   ⏱️  Estimated time: ~15-30 minutes for SAM annotation + training time")


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description="Test SAM annotation cache system")
    parser.add_argument("--integration", action="store_true", help="Test cache integration")
    parser.add_argument("--workflow", action="store_true", help="Test training workflow simulation")
    parser.add_argument("--simulate", action="store_true", help="Simulate complete cache workflow")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    
    args = parser.parse_args()
    
    if args.all or args.integration:
        success = test_cache_integration()
        if not success:
            print("\n❌ Cache integration test failed")
            return
    
    if args.all or args.workflow:
        success = test_training_workflow_simulation()
        if not success:
            print("\n❌ Training workflow test failed")
            return
    
    if args.all or args.simulate:
        simulate_cache_workflow()
    
    if not any([args.integration, args.workflow, args.simulate, args.all]):
        print("🎯 SAM ANNOTATION CACHE TESTING")
        print("=" * 40)
        print("Available tests:")
        print("  --integration  : Test cache system integration")
        print("  --workflow     : Test training workflow simulation")
        print("  --simulate     : Simulate complete cache workflow")
        print("  --all          : Run all tests")
        print()
        print("Example: python3 test_cache.py --all")
        
        # Run a quick status check
        print("\n📊 Quick Cache Status:")
        cache_root = Path("artifacts/annotations_cache")
        if cache_root.exists():
            content_types_dir = cache_root / "content_types"
            if content_types_dir.exists():
                content_types = [d.name for d in content_types_dir.iterdir() if d.is_dir()]
                print(f"Content types: {', '.join(content_types) if content_types else 'None'}")
            else:
                print("No content types initialized")
        else:
            print("Cache not initialized")


if __name__ == "__main__":
    main()
