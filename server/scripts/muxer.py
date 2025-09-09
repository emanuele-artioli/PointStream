#!/usr/bin/env python3
import json
import gzip
import pickle
from pathlib import Path
from typing import Dict, List, Any, Union
import numpy as np
from utils.decorators import track_performance
from utils import config

class MetadataMuxer:
    def __init__(self):
        self.compression_level = config.get_int('muxer', 'compression_level', 6)
        self.use_binary_format = config.get_bool('muxer', 'use_binary_format', True)
        self.remove_debug_data = config.get_bool('muxer', 'remove_debug_data', True)
        self.precision_digits = config.get_int('muxer', 'float_precision', 6)
    
    @track_performance
    def compress_metadata_file(self, input_path: Union[str, Path], output_path: Union[str, Path] = None) -> Dict[str, Any]:
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input metadata file not found: {input_path}")
        
        if output_path is None:
            if self.use_binary_format:
                output_path = input_path.with_suffix('.pzm')
            else:
                output_path = input_path.with_suffix('.json.gz')
        
        with open(input_path, 'r') as f:
            original_metadata = json.load(f)

        original_size = input_path.stat().st_size
        return self.compress_metadata_object(original_metadata, output_path, original_size)

    @track_performance
    def compress_metadata_object(self, metadata_obj: Dict[str, Any], output_path: Union[str, Path], original_size: int = 0) -> Dict[str, Any]:
        output_path = Path(output_path)
        compressed_metadata = self._compress_metadata(metadata_obj)
        
        if self.use_binary_format:
            compressed_size = self._save_binary_format(compressed_metadata, output_path)
        else:
            if output_path.suffix != '.gz':
                output_path = output_path.with_suffix('.json.gz')
            compressed_size = self._save_json_gz_format(compressed_metadata, output_path)

        if original_size > 0:
            compression_ratio = original_size / compressed_size if compressed_size > 0 else 0
            space_saved = original_size - compressed_size
        else:
            compression_ratio = 0
            space_saved = 0

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
        input_dir = Path(input_dir)
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        if output_dir is None:
            output_dir = input_dir / "compressed"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(exist_ok=True)
        
        metadata_files = list(input_dir.glob("*_metadata.json"))
        if not metadata_files:
            return {'compressed_files': 0, 'total_space_saved': 0}
        
        total_original_size = 0
        total_compressed_size = 0
        successful_files = 0
        failed_files = []
        
        for metadata_file in metadata_files:
            try:
                if self.use_binary_format:
                    output_file = output_dir / metadata_file.with_suffix('.pzm').name
                else:
                    output_file = output_dir / f"{metadata_file.stem}.json.gz"
                
                result = self.compress_metadata_file(metadata_file, output_file)
                
                total_original_size += result['original_size']
                total_compressed_size += result['compressed_size']
                successful_files += 1
            except Exception:
                failed_files.append(str(metadata_file))
        
        overall_compression_ratio = total_original_size / total_compressed_size if total_compressed_size > 0 else 0
        total_space_saved = total_original_size - total_compressed_size
        
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
        compressed = {}
        for key, value in metadata.items():
            if self.remove_debug_data and key in ['debug_info', 'processing_log', 'verbose_stats']:
                continue
            
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
                compressed[key] = round(value, self.precision_digits)
            else:
                compressed[key] = value
        return compressed
    
    def _compress_homographies(self, data: Any) -> Any:
        if isinstance(data, dict) and 'homographies' in data:
            homographies = data['homographies']
            if isinstance(homographies, list):
                compressed_homographies = []
                for h in homographies:
                    if isinstance(h, list) and len(h) == 3 and len(h[0]) == 3:
                        compressed_h = [[round(val, self.precision_digits) for val in row] for row in h]
                        compressed_homographies.append(compressed_h)
                    else:
                        compressed_homographies.append(h)
                result = data.copy()
                result['homographies'] = compressed_homographies
                return result
        return data
    
    def _compress_stitching_result(self, stitching_result: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(stitching_result, dict):
            return stitching_result
        compressed = {}
        for key, value in stitching_result.items():
            if key == 'homographies' and isinstance(value, list):
                compressed_homographies = []
                for h in value:
                    if isinstance(h, list) and len(h) == 3:
                        compressed_h = [[round(val, self.precision_digits) for val in row] for row in h]
                        compressed_homographies.append(compressed_h)
                    else:
                        compressed_homographies.append(h)
                compressed[key] = compressed_homographies
            elif key in ['processing_time', 'debug_info'] and self.remove_debug_data:
                continue
            else:
                compressed[key] = value
        return compressed
    
    def _compress_keypoints(self, data: Any) -> Any:
        if isinstance(data, dict):
            compressed = {}
            for key, value in data.items():
                if key == 'keypoints' and isinstance(value, list):
                    compressed_keypoints = []
                    for kp in value:
                        if isinstance(kp, list) and len(kp) >= 2:
                            compressed_kp = [round(coord, 1) for coord in kp]
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
        with gzip.open(output_path, 'wb', compresslevel=self.compression_level) as f:
            pickle.dump(data, f)
        return output_path.stat().st_size
    
    def _save_json_gz_format(self, data: Dict[str, Any], output_path: Path) -> int:
        json_str = json.dumps(data, separators=(',', ':'))
        with gzip.open(output_path, 'wt', compresslevel=self.compression_level) as f:
            f.write(json_str)
        return output_path.stat().st_size

def main():
    import argparse
    parser = argparse.ArgumentParser(description="PointStream Metadata Muxer")
    parser.add_argument('input', nargs='?', help='Input metadata file or directory')
    parser.add_argument('--directory', '-d', help='Process all metadata files in directory')
    parser.add_argument('--output', '-o', help='Output file or directory')
    parser.add_argument('--format', choices=['binary', 'json'], default='binary', help='Output format')
    parser.add_argument('--compression', '-c', type=int, default=6, choices=range(1, 10), help='Compression level (1-9)')
    args = parser.parse_args()
    
    if not args.input and not args.directory:
        parser.print_help()
        return
    
    from utils.config import config
    config.set('muxer', 'compression_level', str(args.compression))
    config.set('muxer', 'use_binary_format', str(args.format == 'binary'))
    
    muxer = MetadataMuxer()
    
    try:
        if args.directory or (args.input and Path(args.input).is_dir()):
            input_dir = args.directory or args.input
            muxer.compress_metadata_directory(input_dir, args.output)
        else:
            muxer.compress_metadata_file(args.input, args.output)
    except Exception as e:
        print(f"Error: {e}")
        return 1
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
