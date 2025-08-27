#!/usr/bin/env python3
"""
PointStream Metadata Demuxer

This module decompresses metadata files that were compressed by the muxer.
It restores the original data structure for use by client components.
"""

import json
import gzip
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Any, Union
import numpy as np

try:
    from utils.decorators import track_performance
    from utils import config
except ImportError as e:
    logging.error(f"Failed to import PointStream utilities: {e}")
    raise


class MetadataDemuxer:
    """
    Decompresses and restores metadata files compressed by the PointStream muxer.
    
    The demuxer handles both binary (.pzm) and JSON+gzip (.json.gz) formats,
    and restores numpy arrays and other data structures for client use.
    """
    
    def __init__(self):
        """Initialize the metadata demuxer."""
        self.restore_numpy_arrays = config.get_bool('demuxer', 'restore_numpy_arrays', True)
        self.validate_homographies = config.get_bool('demuxer', 'validate_homographies', True)
        
        logging.info("📦 Metadata Demuxer initialized")
        logging.info(f"   Restore numpy arrays: {self.restore_numpy_arrays}")
        logging.info(f"   Validate homographies: {self.validate_homographies}")
    
    @track_performance
    def decompress_metadata_file(self, input_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Decompress a single metadata file.
        
        Args:
            input_path: Path to the compressed metadata file (.pzm or .json.gz)
            
        Returns:
            Decompressed metadata dictionary
        """
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Compressed metadata file not found: {input_path}")
        
        # Determine format based on file extension
        if input_path.suffix == '.pzm':
            metadata = self._load_binary_format(input_path)
        elif input_path.suffix == '.gz' and input_path.stem.endswith('.json'):
            metadata = self._load_json_gz_format(input_path)
        else:
            # Try to auto-detect format
            try:
                metadata = self._load_binary_format(input_path)
            except:
                try:
                    metadata = self._load_json_gz_format(input_path)
                except:
                    raise ValueError(f"Unable to determine format of compressed file: {input_path}")
        
        # Restore data structures
        restored_metadata = self._restore_metadata(metadata)
        
        logging.info(f"📦 Decompressed {input_path.name} successfully")
        
        return restored_metadata
    
    @track_performance
    def decompress_metadata_directory(self, input_dir: Union[str, Path]) -> Dict[str, Dict[str, Any]]:
        """
        Decompress all metadata files in a directory.
        
        Args:
            input_dir: Directory containing compressed metadata files
            
        Returns:
            Dictionary mapping filenames to decompressed metadata
        """
        input_dir = Path(input_dir)
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        # Find compressed metadata files
        compressed_files = list(input_dir.glob("*.pzm")) + list(input_dir.glob("*.json.gz"))
        if not compressed_files:
            logging.warning(f"No compressed metadata files found in {input_dir}")
            return {}
        
        logging.info(f"📦 Decompressing {len(compressed_files)} metadata files...")
        
        results = {}
        successful_files = 0
        failed_files = []
        
        for compressed_file in compressed_files:
            try:
                metadata = self.decompress_metadata_file(compressed_file)
                results[compressed_file.stem] = metadata
                successful_files += 1
                
            except Exception as e:
                logging.error(f"Failed to decompress {compressed_file.name}: {e}")
                failed_files.append(str(compressed_file))
        
        logging.info(f"📊 Decompression Summary:")
        logging.info(f"   Files processed: {successful_files}/{len(compressed_files)}")
        if failed_files:
            logging.warning(f"   Failed files: {failed_files}")
        
        return results
    
    def _load_binary_format(self, file_path: Path) -> Dict[str, Any]:
        """Load metadata from binary format (pickle + gzip)."""
        with gzip.open(file_path, 'rb') as f:
            return pickle.load(f)
    
    def _load_json_gz_format(self, file_path: Path) -> Dict[str, Any]:
        """Load metadata from JSON + gzip format."""
        with gzip.open(file_path, 'rt') as f:
            return json.load(f)
    
    def _restore_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Restore metadata structures and convert data types as needed.
        
        Args:
            metadata: Decompressed metadata dictionary
            
        Returns:
            Restored metadata with proper data types
        """
        restored = {}
        
        for key, value in metadata.items():
            if key == 'stitching_result' and isinstance(value, dict):
                restored[key] = self._restore_stitching_result(value)
            elif key == 'keypoints_result' and isinstance(value, dict):
                restored[key] = self._restore_keypoints_result(value)
            elif isinstance(value, dict):
                restored[key] = self._restore_metadata(value)
            elif isinstance(value, list):
                restored[key] = self._restore_list(value)
            else:
                restored[key] = value
        
        return restored
    
    def _restore_stitching_result(self, stitching_result: Dict[str, Any]) -> Dict[str, Any]:
        """Restore stitching result with proper homography arrays."""
        restored = {}
        
        for key, value in stitching_result.items():
            if key == 'homographies' and isinstance(value, list):
                restored[key] = self._restore_homographies(value)
            else:
                restored[key] = value
        
        return restored
    
    def _restore_homographies(self, homographies: List[Any]) -> List[np.ndarray]:
        """
        Restore homography matrices as numpy arrays.
        
        Args:
            homographies: List of homography matrices (as lists or arrays)
            
        Returns:
            List of 3x3 numpy arrays
        """
        restored_homographies = []
        
        for h in homographies:
            if isinstance(h, list) and len(h) == 3 and len(h[0]) == 3:
                # Convert 3x3 list to numpy array
                if self.restore_numpy_arrays:
                    h_array = np.array(h, dtype=np.float32)
                    
                    # Validate homography if enabled
                    if self.validate_homographies:
                        if not self._is_valid_homography(h_array):
                            logging.warning(f"Invalid homography detected: {h_array}")
                    
                    restored_homographies.append(h_array)
                else:
                    restored_homographies.append(h)
            else:
                # Handle edge cases or malformed data
                if self.restore_numpy_arrays:
                    if isinstance(h, list):
                        restored_homographies.append(np.array(h, dtype=np.float32))
                    else:
                        # Fallback: create identity matrix
                        logging.warning(f"Malformed homography, using identity: {h}")
                        restored_homographies.append(np.eye(3, dtype=np.float32))
                else:
                    restored_homographies.append(h)
        
        return restored_homographies
    
    def _restore_keypoints_result(self, keypoints_result: Dict[str, Any]) -> Dict[str, Any]:
        """Restore keypoints data structures."""
        restored = {}
        
        for key, value in keypoints_result.items():
            if isinstance(value, dict):
                restored[key] = self._restore_keypoints_result(value)
            elif isinstance(value, list) and key == 'keypoints':
                # Restore keypoints as numpy arrays if configured
                if self.restore_numpy_arrays:
                    restored_keypoints = []
                    for kp in value:
                        if isinstance(kp, list):
                            restored_keypoints.append(np.array(kp, dtype=np.float32))
                        else:
                            restored_keypoints.append(kp)
                    restored[key] = restored_keypoints
                else:
                    restored[key] = value
            else:
                restored[key] = value
        
        return restored
    
    def _restore_list(self, data: List[Any]) -> List[Any]:
        """Restore list data recursively."""
        restored = []
        for item in data:
            if isinstance(item, dict):
                restored.append(self._restore_metadata(item))
            elif isinstance(item, list):
                restored.append(self._restore_list(item))
            else:
                restored.append(item)
        return restored
    
    def _is_valid_homography(self, h: np.ndarray) -> bool:
        """
        Validate that a homography matrix is well-formed.
        
        Args:
            h: 3x3 homography matrix
            
        Returns:
            True if the homography appears valid
        """
        if h.shape != (3, 3):
            return False
        
        # Check if the matrix is not all zeros
        if np.allclose(h, 0):
            return False
        
        # Check if the bottom-right element is reasonable (usually close to 1)
        if abs(h[2, 2]) < 1e-6:
            return False
        
        # Check for reasonable determinant (should not be zero)
        det = np.linalg.det(h)
        if abs(det) < 1e-6:
            return False
        
        return True
    
    def get_scene_metadata(self, scene_file: Union[str, Path]) -> Dict[str, Any]:
        """
        Convenience method to load and decompress a single scene metadata file.
        
        Args:
            scene_file: Path to compressed or uncompressed scene metadata file
            
        Returns:
            Scene metadata dictionary ready for client processing
        """
        scene_file = Path(scene_file)
        
        # Handle different file types
        if scene_file.suffix in ['.pzm', '.gz']:
            # Compressed file - decompress it
            metadata = self.decompress_metadata_file(scene_file)
        elif scene_file.suffix == '.json':
            # Uncompressed JSON file - load directly
            with open(scene_file, 'r') as f:
                metadata = json.load(f)
            # Still apply restoration to handle any nested structures
            metadata = self._restore_metadata(metadata)
        else:
            raise ValueError(f"Unsupported file type: {scene_file.suffix}")
        
        # Unpack stitching_result into the top-level metadata for client compatibility
        if 'stitching_result' in metadata:
            stitching_data = metadata.pop('stitching_result')
            metadata.update(stitching_data)
        
        return metadata


def main():
    """CLI interface for the metadata demuxer."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="PointStream Metadata Demuxer - Decompress metadata files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Decompress a single metadata file
    python demuxer.py scene_0001_metadata.pzm
    
    # Decompress all metadata files in a directory
    python demuxer.py --directory ./compressed_metadata
    
    # Save decompressed output to specific file
    python demuxer.py scene_0001_metadata.pzm --output scene_0001_metadata.json
        """
    )
    
    parser.add_argument('input', nargs='?', help='Input compressed metadata file')
    parser.add_argument('--directory', '-d', help='Process all compressed files in directory')
    parser.add_argument('--output', '-o', help='Output file or directory')
    parser.add_argument('--format', choices=['json'], default='json',
                       help='Output format for decompressed data')
    
    args = parser.parse_args()
    
    if not args.input and not args.directory:
        parser.print_help()
        return
    
    # Initialize demuxer
    demuxer = MetadataDemuxer()
    
    try:
        if args.directory or (args.input and Path(args.input).is_dir()):
            # Directory mode
            input_dir = args.directory or args.input
            results = demuxer.decompress_metadata_directory(input_dir)
            
            # Save results if output directory specified
            if args.output:
                output_dir = Path(args.output)
                output_dir.mkdir(exist_ok=True)
                
                for filename, metadata in results.items():
                    output_file = output_dir / f"{filename}.json"
                    with open(output_file, 'w') as f:
                        json.dump(metadata, f, indent=2)
                
                print(f"✅ Decompressed {len(results)} files to {output_dir}")
            else:
                print(f"✅ Decompressed {len(results)} files (metadata loaded in memory)")
                
        else:
            # Single file mode
            metadata = demuxer.decompress_metadata_file(args.input)
            
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(metadata, f, indent=2)
                print(f"✅ Decompressed to {args.output}")
            else:
                print(f"✅ Decompressed {Path(args.input).name} (metadata loaded in memory)")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
