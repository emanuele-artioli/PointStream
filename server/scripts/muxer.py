#!/usr/bin/env python3
"""
PointStream Metadata Muxer

This module compresses metadata files to reduce size and network transfer time.
It removes redundant information, compacts data structures, and applies compression.
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


class MetadataMuxer:
    """
    Compresses and optimizes metadata files for efficient storage and transfer.
    
    The muxer performs several optimizations:
    1. Removes redundant and unnecessary data
    2. Compacts data structures 
    3. Applies compression algorithms
    4. Creates compact binary format
    """
    
    def __init__(self):
        """Initialize the metadata muxer."""
        self.compression_level = config.get_int('muxer', 'compression_level', 6)
        self.use_binary_format = config.get_bool('muxer', 'use_binary_format', True)
        self.remove_debug_data = config.get_bool('muxer', 'remove_debug_data', True)
        self.precision_digits = config.get_int('muxer', 'float_precision', 6)
        
        logging.info("🗜️  Metadata Muxer initialized")
        logging.info(f"   Compression level: {self.compression_level}")
        logging.info(f"   Binary format: {self.use_binary_format}")
        logging.info(f"   Float precision: {self.precision_digits} digits")
    
    @track_performance
    def compress_metadata_file(self, input_path: Union[str, Path], output_path: Union[str, Path] = None) -> Dict[str, Any]:
        """
        Compress a single metadata file by loading it and passing it to compress_metadata_object.
        
        Args:
            input_path: Path to the input JSON metadata file
            output_path: Path for the compressed output file (optional)
            
        Returns:
            Dictionary with compression results and statistics
        """
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input metadata file not found: {input_path}")
        
        # Determine output path
        if output_path is None:
            if self.use_binary_format:
                output_path = input_path.with_suffix('.pzm')  # PointStream Zipped Metadata
            else:
                output_path = input_path.with_suffix('.json.gz')
        
        # Load original metadata
        with open(input_path, 'r') as f:
            original_metadata = json.load(f)

        original_size = input_path.stat().st_size

        # Delegate to the object compression method
        return self.compress_metadata_object(original_metadata, output_path, original_size)

    @track_performance
    def compress_metadata_object(self, metadata_obj: Dict[str, Any], output_path: Union[str, Path], original_size: int = 0) -> Dict[str, Any]:
        """
        Compress a metadata dictionary from memory and save it.

        Args:
            metadata_obj: The metadata dictionary to compress
            output_path: Path for the compressed output file
            original_size: The original size in bytes (if known, for stats)

        Returns:
            Dictionary with compression results and statistics
        """
        output_path = Path(output_path)
        
        # Compress the metadata
        compressed_metadata = self._compress_metadata(metadata_obj)
        
        # Save compressed metadata
        if self.use_binary_format:
            compressed_size = self._save_binary_format(compressed_metadata, output_path)
        else:
            # Adjust output path for json.gz if needed
            if output_path.suffix != '.gz':
                output_path = output_path.with_suffix('.json.gz')
            compressed_size = self._save_json_gz_format(compressed_metadata, output_path)

        # Explicitly log file creation and output path
        if Path(output_path).exists():
            logging.info(f"✅ Muxed file created: {output_path} ({compressed_size} bytes)")
        else:
            logging.error(f"❌ Muxed file NOT created: {output_path}")

        # Calculate compression statistics
        if original_size > 0:
            compression_ratio = original_size / compressed_size if compressed_size > 0 else 0
            space_saved = original_size - compressed_size
            logging.info(f"📦 Compressed metadata: {original_size:,} → {compressed_size:,} bytes "
                        f"({compression_ratio:.1f}x compression, {space_saved:,} bytes saved)")
        else:
            compression_ratio = 0
            space_saved = 0
            logging.info(f"📦 Compressed metadata to {compressed_size:,} bytes")

        return {
            'output_path': str(output_path),
            'original_size': original_size,
            'compressed_size': compressed_size,
            'compression_ratio': compression_ratio,
            'space_saved': space_saved,
            'format': 'binary' if self.use_binary_format else 'json.gz'
        }
    
    @track_performance
    def compress_metadata_directory(self, input_dir: Union[str, Path], output_dir: Union[str, Path] = None) -> Dict[str, Any]:
        """
        Compress all metadata files in a directory.
        
        Args:
            input_dir: Directory containing JSON metadata files
            output_dir: Output directory for compressed files (optional)
            
        Returns:
            Dictionary with overall compression statistics
        """
        input_dir = Path(input_dir)
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        if output_dir is None:
            output_dir = input_dir / "compressed"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(exist_ok=True)
        
        # Find all metadata JSON files
        metadata_files = list(input_dir.glob("*_metadata.json"))
        if not metadata_files:
            logging.warning(f"No metadata files found in {input_dir}")
            return {'compressed_files': 0, 'total_space_saved': 0}
        
        logging.info(f"🗜️  Compressing {len(metadata_files)} metadata files...")
        
        total_original_size = 0
        total_compressed_size = 0
        successful_files = 0
        failed_files = []
        
        for metadata_file in metadata_files:
            try:
                # Determine output filename
                if self.use_binary_format:
                    output_file = output_dir / metadata_file.with_suffix('.pzm').name
                else:
                    output_file = output_dir / f"{metadata_file.stem}.json.gz"
                
                result = self.compress_metadata_file(metadata_file, output_file)
                
                total_original_size += result['original_size']
                total_compressed_size += result['compressed_size']
                successful_files += 1
                
            except Exception as e:
                logging.error(f"Failed to compress {metadata_file.name}: {e}")
                failed_files.append(str(metadata_file))
        
        # Calculate overall statistics
        overall_compression_ratio = total_original_size / total_compressed_size if total_compressed_size > 0 else 0
        total_space_saved = total_original_size - total_compressed_size
        
        logging.info(f"📊 Compression Summary:")
        logging.info(f"   Files processed: {successful_files}/{len(metadata_files)}")
        logging.info(f"   Total size reduction: {total_original_size:,} → {total_compressed_size:,} bytes")
        logging.info(f"   Overall compression: {overall_compression_ratio:.1f}x")
        logging.info(f"   Total space saved: {total_space_saved:,} bytes ({total_space_saved/1024/1024:.1f} MB)")
        
        return {
            'input_directory': str(input_dir),
            'output_directory': str(output_dir),
            'compressed_files': successful_files,
            'failed_files': failed_files,
            'total_original_size': total_original_size,
            'total_compressed_size': total_compressed_size,
            'overall_compression_ratio': overall_compression_ratio,
            'total_space_saved': total_space_saved
        }
    
    def _compress_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply compression optimizations to metadata structure.
        
        Args:
            metadata: Original metadata dictionary
            
        Returns:
            Compressed and optimized metadata
        """
        compressed = {}
        
        for key, value in metadata.items():
            # Remove debug and unnecessary data if configured
            if self.remove_debug_data and key in ['debug_info', 'processing_log', 'verbose_stats']:
                continue
            
            # Apply specific compression strategies
            if key == 'homographies' or (isinstance(value, dict) and 'homographies' in value):
                compressed[key] = self._compress_homographies(value)
            elif key == 'keypoints_result' or (isinstance(value, dict) and 'keypoints' in str(value)):
                compressed[key] = self._compress_keypoints(value)
            elif key == 'stitching_result':
                compressed[key] = self._compress_stitching_result(value)
            elif isinstance(value, dict):
                compressed[key] = self._compress_metadata(value)
            elif isinstance(value, list):
                compressed[key] = self._compress_list(value)
            elif isinstance(value, float):
                # Round floats to reduce precision
                compressed[key] = round(value, self.precision_digits)
            else:
                compressed[key] = value
        
        return compressed
    
    def _compress_homographies(self, data: Any) -> Any:
        """Compress homography matrices by reducing precision and removing redundant data."""
        if isinstance(data, dict) and 'homographies' in data:
            homographies = data['homographies']
            if isinstance(homographies, list):
                # Round homography values to reduce precision
                compressed_homographies = []
                for h in homographies:
                    if isinstance(h, list) and len(h) == 3 and len(h[0]) == 3:
                        # Round each value in the 3x3 matrix
                        compressed_h = [[round(val, self.precision_digits) for val in row] for row in h]
                        compressed_homographies.append(compressed_h)
                    else:
                        compressed_homographies.append(h)
                
                # Create compressed result
                result = data.copy()
                result['homographies'] = compressed_homographies
                return result
        
        return data
    
    def _compress_stitching_result(self, stitching_result: Dict[str, Any]) -> Dict[str, Any]:
        """Compress stitching result by optimizing homographies and removing redundant data."""
        if not isinstance(stitching_result, dict):
            return stitching_result
        
        compressed = {}
        for key, value in stitching_result.items():
            if key == 'homographies' and isinstance(value, list):
                # Compress homography matrices
                compressed_homographies = []
                for h in value:
                    if isinstance(h, list) and len(h) == 3:
                        # Round values and convert to more compact representation
                        compressed_h = [[round(val, self.precision_digits) for val in row] for row in h]
                        compressed_homographies.append(compressed_h)
                    else:
                        compressed_homographies.append(h)
                compressed[key] = compressed_homographies
            elif key in ['processing_time', 'debug_info'] and self.remove_debug_data:
                # Skip debug data
                continue
            else:
                compressed[key] = value
        
        return compressed
    
    def _compress_keypoints(self, data: Any) -> Any:
        """Compress keypoints data by reducing precision."""
        if isinstance(data, dict):
            compressed = {}
            for key, value in data.items():
                if key == 'keypoints' and isinstance(value, list):
                    # Round keypoint coordinates
                    compressed_keypoints = []
                    for kp in value:
                        if isinstance(kp, list) and len(kp) >= 2:
                            compressed_kp = [round(coord, 1) for coord in kp]  # 1 decimal for keypoints
                            compressed_keypoints.append(compressed_kp)
                        else:
                            compressed_keypoints.append(kp)
                    compressed[key] = compressed_keypoints
                elif isinstance(value, dict):
                    compressed[key] = self._compress_keypoints(value)
                else:
                    compressed[key] = value
            return compressed
        elif isinstance(data, list):
            return [self._compress_keypoints(item) for item in data]
        else:
            return data
    
    def _compress_list(self, data: List[Any]) -> List[Any]:
        """Compress list data recursively."""
        compressed = []
        for item in data:
            if isinstance(item, dict):
                compressed.append(self._compress_metadata(item))
            elif isinstance(item, list):
                compressed.append(self._compress_list(item))
            elif isinstance(item, float):
                compressed.append(round(item, self.precision_digits))
            else:
                compressed.append(item)
        return compressed
    
    def _save_binary_format(self, data: Dict[str, Any], output_path: Path) -> int:
        """Save compressed metadata in binary format using pickle + gzip."""
        with gzip.open(output_path, 'wb', compresslevel=self.compression_level) as f:
            pickle.dump(data, f)
        
        return output_path.stat().st_size
    
    def _save_json_gz_format(self, data: Dict[str, Any], output_path: Path) -> int:
        """Save compressed metadata in JSON + gzip format."""
        json_str = json.dumps(data, separators=(',', ':'))  # Compact JSON
        
        with gzip.open(output_path, 'wt', compresslevel=self.compression_level) as f:
            f.write(json_str)
        
        return output_path.stat().st_size


def main():
    """CLI interface for the metadata muxer."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="PointStream Metadata Muxer - Compress metadata files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Compress a single metadata file
    python muxer.py scene_0001_metadata.json
    
    # Compress all metadata files in a directory
    python muxer.py --directory ./test_output
    
    # Use JSON+gzip format instead of binary
    python muxer.py --format json scene_0001_metadata.json
        """
    )
    
    parser.add_argument('input', nargs='?', help='Input metadata file or directory')
    parser.add_argument('--directory', '-d', help='Process all metadata files in directory')
    parser.add_argument('--output', '-o', help='Output file or directory')
    parser.add_argument('--format', choices=['binary', 'json'], default='binary',
                       help='Output format: binary (pickle+gzip) or json (json+gzip)')
    parser.add_argument('--compression', '-c', type=int, default=6, choices=range(1, 10),
                       help='Compression level (1-9, higher = more compression)')
    
    args = parser.parse_args()
    
    if not args.input and not args.directory:
        parser.print_help()
        return
    
    # Configure muxer settings
    from utils.config import config
    config.set('muxer', 'compression_level', str(args.compression))
    config.set('muxer', 'use_binary_format', str(args.format == 'binary'))
    
    # Initialize muxer
    muxer = MetadataMuxer()
    
    try:
        if args.directory or (args.input and Path(args.input).is_dir()):
            # Directory mode
            input_dir = args.directory or args.input
            result = muxer.compress_metadata_directory(input_dir, args.output)
            print(f"✅ Compressed {result['compressed_files']} files")
            print(f"Space saved: {result['total_space_saved']:,} bytes")
        else:
            # Single file mode
            result = muxer.compress_metadata_file(args.input, args.output)
            print(f"✅ Compressed {Path(args.input).name}")
            print(f"Compression: {result['compression_ratio']:.1f}x")
            print(f"Space saved: {result['space_saved']:,} bytes")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
