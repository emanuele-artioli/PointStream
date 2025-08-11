#!/usr/bin/env python3
"""
Scene Splitter Generator Demo

This script demonstrates how to use the VideoSceneSplitter as a generator
to yield scene frames one at a time for real-time processing pipelines.
"""

import sys
import time
from pathlib import Path

# Add the PointStream directory to the path
sys.path.append(str(Path(__file__).parent))

from video_scene_splitter import VideoSceneSplitter

def demo_scene_generator(video_path: str, config_path: str = None):
    """
    Demonstrate the scene generator functionality.
    
    Args:
        video_path: Path to input video
        config_path: Optional path to config file
    """
    print("=== Scene Splitter Generator Demo ===")
    print(f"Video: {video_path}")
    print(f"Config: {config_path or 'default'}")
    print()
    
    # Initialize splitter without encoding (just frame extraction)
    splitter = VideoSceneSplitter(
        input_video=video_path,
        enable_encoding=False,  # No encoding for performance
        config_file=config_path
    )
    
    print("Processing scenes in real-time...")
    print("(In production, frames would be passed to next processing stage)")
    print()
    
    scene_count = 0
    processing_times = []
    
    try:
        for scene_data in splitter.process_video_realtime_generator():
            
            if scene_data.get('status') == 'complete':
                # Final summary
                summary = scene_data['summary']
                print(f"\n{'='*50}")
                print(f"PROCESSING COMPLETE")
                print(f"{'='*50}")
                print(f"Total scenes processed: {summary['total_scenes']}")
                print(f"Total processing time: {summary['total_processing_time']:.3f}s")
                print(f"Average time per scene: {summary['average_processing_time_per_scene']:.3f}s")
                print(f"Processing speed: {summary['frames_per_second_processing']:.1f} fps")
                print(f"Real-time factor: {summary['real_time_factor']:.2f}x")
                
                if summary['real_time_factor'] >= 1.0:
                    print("✅ REAL-TIME CAPABLE: Processing is faster than video playback!")
                else:
                    print("⚠️  NOT REAL-TIME: Processing is slower than video playback")
                
                break
                
            elif scene_data.get('status') == 'error':
                print(f"❌ Error: {scene_data['error']}")
                continue
            
            else:
                # Process scene frames
                scene_count += 1
                frames = scene_data['frames']
                duration = scene_data['duration']
                start_time = scene_data['start_time']
                end_time = scene_data['end_time']
                
                print(f"Scene {scene_count:2d}: {start_time:6.2f}s-{end_time:6.2f}s "
                      f"({duration:5.2f}s, {len(frames):3d} frames)")
                
                # Simulate next processing stage
                process_start = time.time()
                
                # Here you would pass the frames to your next processing stage
                # For demo, we just check the frame properties
                if frames:
                    first_frame = frames[0]
                    print(f"   → Frame shape: {first_frame.shape}")
                    print(f"   → Frame dtype: {first_frame.dtype}")
                    
                    # Simulate some processing work (in real use, replace with actual processing)
                    # time.sleep(0.001)  # Uncomment to simulate processing time
                
                process_time = time.time() - process_start
                processing_times.append(process_time)
                
                print(f"   → Next stage processing: {process_time*1000:.1f}ms")
                print()
    
    finally:
        splitter.close()
    
    if processing_times:
        avg_next_stage_time = sum(processing_times) / len(processing_times)
        print(f"Average next-stage processing time: {avg_next_stage_time*1000:.1f}ms per scene")


def main():
    if len(sys.argv) < 2:
        print("Usage: python demo_generator.py <video_path> [config_path]")
        print()
        print("Example:")
        print("  python demo_generator.py /path/to/video.mp4")
        print("  python demo_generator.py /path/to/video.mp4 config_live_test.ini")
        sys.exit(1)
    
    video_path = sys.argv[1]
    config_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not Path(video_path).exists():
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)
    
    demo_scene_generator(video_path, config_path)


if __name__ == "__main__":
    main()
