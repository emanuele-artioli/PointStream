#!/usr/bin/env python3
"""
Test script for client reconstruction only
"""
import sys
import logging
from pathlib import Path

# Add current directory to Python path
sys.path.append('.')

from client.client import PointStreamClient, setup_logging

def main():
    setup_logging('INFO')
    
    logging.info("🧪 Testing PointStream client reconstruction...")
    
    # Initialize client
    client = PointStreamClient()
    
    # Test reconstruction with existing metadata (no training)
    metadata_dir = "./debug_metadata2"
    output_dir = "./test_output"
    
    logging.info(f"📂 Processing metadata from: {metadata_dir}")
    
    # Process with training disabled since models exist
    results = client.process_metadata(
        metadata_dir=metadata_dir,
        output_dir=output_dir,
        train_models=False  # Skip training, use existing models
    )
    
    logging.info("✅ Test completed!")
    logging.info(f"Results: {results.get('processed_scenes', 0)} scenes processed")

if __name__ == "__main__":
    main()
