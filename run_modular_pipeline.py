#!/usr/bin/env python3
"""
Modular PointStream Pipeline Runner
Allows stepwise execution, evaluation, and debugging of the pipeline.
"""

import argparse
import json
import time
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
import subprocess


class PipelineRunner:
    """Modular pipeline runner with stepwise execution and evaluation."""
    
    def __init__(self, workspace_dir: str = None):
        """Initialize the pipeline runner."""
        self.workspace_dir = Path(workspace_dir) if workspace_dir else Path(__file__).parent
        self.steps_completed = []
        self.step_results = {}
        self.timing_data = {}
        
        # Define pipeline steps
        self.pipeline_steps = [
            {
                "name": "cleanup",
                "description": "Clean up runtime files",
                "function": self._step_cleanup,
                "required": False
            },
            {
                "name": "test",
                "description": "Run unit tests",
                "function": self._step_test,
                "required": True
            },
            {
                "name": "server",
                "description": "Run server pipeline (analysis and compression)",
                "function": self._step_server,
                "required": True
            },
            {
                "name": "client",
                "description": "Run client pipeline (reconstruction)",
                "function": self._step_client,
                "required": True
            },
            {
                "name": "training",
                "description": "Run training demo",
                "function": self._step_training,
                "required": False
            },
            {
                "name": "evaluation",
                "description": "Run comprehensive evaluation",
                "function": self._step_evaluation,
                "required": True
            },
            {
                "name": "summary",
                "description": "Generate summary report",
                "function": self._step_summary,
                "required": False
            }
        ]
    
    def _run_command(self, command: List[str], description: str, timeout: int = 300) -> Dict[str, Any]:
        """Run a command and return results."""
        print(f"\n{'='*60}")
        print(f"Running: {description}")
        print(f"Command: {' '.join(command)}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.workspace_dir
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            success = result.returncode == 0
            
            print(f"Duration: {duration:.2f}s")
            print(f"Return code: {result.returncode}")
            print(f"Status: {'✓ SUCCESS' if success else '✗ FAILED'}")
            
            if result.stdout:
                print(f"STDOUT:\n{result.stdout}")
            
            if result.stderr and not success:
                print(f"STDERR:\n{result.stderr}")
            
            return {
                "success": success,
                "duration": duration,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }
            
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            print(f"ERROR: Command timed out after {timeout}s")
            return {
                "success": False,
                "duration": duration,
                "stdout": "",
                "stderr": f"Timeout after {timeout}s",
                "returncode": -1
            }
        except Exception as e:
            duration = time.time() - start_time
            print(f"ERROR: Failed to run command: {e}")
            return {
                "success": False,
                "duration": duration,
                "stdout": "",
                "stderr": str(e),
                "returncode": -1
            }
    
    def _step_cleanup(self, args: argparse.Namespace) -> Dict[str, Any]:
        """Step 1: Clean up runtime files."""
        if not args.force_cleanup and not args.all_steps:
            # Only run cleanup if explicitly requested
            return {"success": True, "skipped": True, "message": "Cleanup skipped (use --cleanup or --all)"}
        
        result = self._run_command(
            ["python", "cleanup_runtime_files.py", "--force"],
            "Cleanup runtime files"
        )
        
        return result
    
    def _step_test(self, args: argparse.Namespace) -> Dict[str, Any]:
        """Step 2: Run unit tests."""
        test_type = "fast" if not args.slow_tests else "all"
        
        result = self._run_command(
            ["python", "run_tests.py", "--type", test_type] + (["--verbose"] if args.verbose else []),
            f"Run {test_type} tests"
        )
        
        if not result["success"] and not args.continue_on_error:
            print("❌ Tests failed. Use --continue-on-error to proceed anyway.")
        
        return result
    
    def _step_server(self, args: argparse.Namespace) -> Dict[str, Any]:
        """Step 3: Run server pipeline."""
        video_path = args.video or "tests/data/DAVIS_stitched.mp4"
        
        result = self._run_command(
            ["python", "run_server.py", "--input-video", video_path],
            f"Server pipeline on {video_path}",
            timeout=600  # Longer timeout for processing
        )
        
        # Check for expected output
        if result["success"]:
            expected_output = self.workspace_dir / "artifacts" / "pipeline_output"
            if not expected_output.exists() or not any(expected_output.glob("*_final_results.json")):
                result["success"] = False
                result["stderr"] += "\nERROR: Expected server output files not found"
        
        return result
    
    def _step_client(self, args: argparse.Namespace) -> Dict[str, Any]:
        """Step 4: Run client pipeline."""
        # Find the latest server results
        output_dir = self.workspace_dir / "artifacts" / "pipeline_output"
        result_files = list(output_dir.glob("*_final_results.json"))
        
        if not result_files:
            return {
                "success": False,
                "duration": 0,
                "stdout": "",
                "stderr": "No server results found for client reconstruction",
                "returncode": -1
            }
        
        latest_result = max(result_files, key=lambda x: x.stat().st_mtime)
        
        result = self._run_command(
            ["python", "run_client.py", "--input-json", str(latest_result)],
            f"Client reconstruction from {latest_result.name}",
            timeout=300
        )
        
        # Check for expected output
        if result["success"]:
            video_name = latest_result.stem.replace("_final_results", "")
            expected_output = self.workspace_dir / f"{video_name}_reconstructed_scenes"
            if not expected_output.exists() or not any(expected_output.glob("scene_*.mp4")):
                result["success"] = False
                result["stderr"] += "\nERROR: Expected client output files not found"
        
        return result
    
    def _step_training(self, args: argparse.Namespace) -> Dict[str, Any]:
        """Step 5: Run training demo."""
        result = self._run_command(
            ["python", "train.py", "--demo", "--epochs", "1"],
            "Training demo (1 epoch)",
            timeout=300
        )
        
        return result
    
    def _step_evaluation(self, args: argparse.Namespace) -> Dict[str, Any]:
        """Step 6: Run comprehensive evaluation."""
        # Find reconstructed videos
        reconstructed_dirs = list(self.workspace_dir.glob("*_reconstructed_scenes"))
        
        if not reconstructed_dirs:
            return {
                "success": False,
                "duration": 0,
                "stdout": "",
                "stderr": "No reconstructed video directories found for evaluation",
                "returncode": -1
            }
        
        latest_dir = max(reconstructed_dirs, key=lambda x: x.stat().st_mtime)
        
        # Get original video path
        video_name = latest_dir.name.replace("_reconstructed_scenes", "")
        if video_name == "DAVIS_stitched":
            original_video = "tests/data/DAVIS_stitched.mp4"
        else:
            original_video = f"{video_name}.mp4"
        
        # Find corresponding JSON results
        json_results_pattern = f"{video_name}_final_results.json"
        json_results_path = self.workspace_dir / "artifacts" / "pipeline_output" / json_results_pattern
        
        if not json_results_path.exists():
            return {
                "success": False,
                "duration": 0,
                "stdout": "",
                "stderr": f"JSON results file not found: {json_results_path}",
                "returncode": -1
            }
        
        result = self._run_command(
            [
                "python", "run_evaluation.py", 
                "--original-video", original_video,
                "--json-results", str(json_results_path),
                "--reconstructed-dir", str(latest_dir),
                "--skip-fvd"  # Skip FVD for faster evaluation
            ],
            f"Evaluation of {latest_dir.name}",
            timeout=600  # Longer timeout for metrics calculation
        )
        
        return result
    
    def _step_summary(self, args: argparse.Namespace) -> Dict[str, Any]:
        """Step 7: Generate summary report."""
        result = self._run_command(
            ["python", "create_evaluation_summary.py"],
            "Generate evaluation summary",
            timeout=60
        )
        
        return result
    
    def run_step(self, step_name: str, args: argparse.Namespace) -> bool:
        """Run a single pipeline step."""
        step = next((s for s in self.pipeline_steps if s["name"] == step_name), None)
        if not step:
            print(f"ERROR: Unknown step '{step_name}'")
            return False
        
        print(f"\n🚀 Starting step: {step['name']} - {step['description']}")
        
        start_time = time.time()
        result = step["function"](args)
        end_time = time.time()
        
        # Store timing data
        self.timing_data[step_name] = {
            "duration": end_time - start_time,
            "start_time": start_time,
            "end_time": end_time
        }
        
        # Store step results
        self.step_results[step_name] = result
        
        if result.get("skipped"):
            print(f"⏭️  Step '{step_name}' skipped: {result.get('message', 'No reason given')}")
            return True
        elif result["success"]:
            print(f"✅ Step '{step_name}' completed successfully")
            self.steps_completed.append(step_name)
            return True
        else:
            print(f"❌ Step '{step_name}' failed")
            if step["required"] and not args.continue_on_error:
                print(f"   Required step failed. Use --continue-on-error to proceed.")
                return False
            elif not step["required"]:
                print(f"   Optional step failed, continuing...")
                return True
            else:
                print(f"   Continuing despite failure (--continue-on-error enabled)")
                return True
    
    def run_pipeline(self, args: argparse.Namespace) -> bool:
        """Run the complete pipeline or selected steps."""
        if args.steps:
            # Run specific steps
            steps_to_run = args.steps
        elif args.all_steps:
            # Run all steps
            steps_to_run = [step["name"] for step in self.pipeline_steps]
        else:
            # Run default steps (excluding cleanup and optional steps)
            steps_to_run = [step["name"] for step in self.pipeline_steps 
                          if step["required"] and step["name"] != "cleanup"]
        
        print(f"📋 Pipeline steps to run: {', '.join(steps_to_run)}")
        
        overall_success = True
        
        for step_name in steps_to_run:
            success = self.run_step(step_name, args)
            if not success:
                overall_success = False
                break
        
        # Generate final report
        self._generate_final_report()
        
        return overall_success
    
    def _generate_final_report(self):
        """Generate a final report of the pipeline run."""
        print(f"\n{'='*80}")
        print("PIPELINE EXECUTION REPORT")
        print(f"{'='*80}")
        
        total_time = sum(timing["duration"] for timing in self.timing_data.values())
        
        print(f"Total execution time: {total_time:.2f}s")
        print(f"Steps completed: {len(self.steps_completed)}")
        print(f"Steps attempted: {len(self.step_results)}")
        
        print(f"\nStep-by-step results:")
        for step_name, result in self.step_results.items():
            duration = self.timing_data.get(step_name, {}).get("duration", 0)
            status = "✅ SUCCESS" if result["success"] else "❌ FAILED"
            if result.get("skipped"):
                status = "⏭️  SKIPPED"
            
            print(f"  {step_name:12} | {status:10} | {duration:6.2f}s")
        
        # Save detailed report
        report_data = {
            "execution_time": time.time(),
            "total_duration": total_time,
            "steps_completed": self.steps_completed,
            "step_results": self.step_results,
            "timing_data": self.timing_data
        }
        
        report_file = self.workspace_dir / "pipeline_execution_report.json"
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\nDetailed report saved to: {report_file}")


def main():
    parser = argparse.ArgumentParser(description="Modular PointStream Pipeline Runner")
    
    # Get available step names
    runner = PipelineRunner()
    available_steps = [step["name"] for step in runner.pipeline_steps]
    
    # Step selection
    parser.add_argument(
        "--steps", 
        nargs="+", 
        choices=available_steps,
        help="Specific steps to run"
    )
    parser.add_argument(
        "--all-steps", 
        action="store_true", 
        help="Run all pipeline steps including optional ones"
    )
    
    # Video input
    parser.add_argument(
        "--video", 
        help="Path to input video (default: tests/data/DAVIS_stitched.mp4)"
    )
    
    # Execution options
    parser.add_argument(
        "--continue-on-error", 
        action="store_true", 
        help="Continue pipeline execution even if a step fails"
    )
    parser.add_argument(
        "--force-cleanup", 
        action="store_true", 
        help="Force cleanup step even if not running all steps"
    )
    parser.add_argument(
        "--slow-tests", 
        action="store_true", 
        help="Run slow tests including VMAF and FVD"
    )
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true", 
        help="Verbose output"
    )
    
    # Workspace
    parser.add_argument(
        "--workspace", 
        help="Workspace directory (default: current directory)"
    )
    
    args = parser.parse_args()
    
    # Show available steps if no action specified
    if not args.steps and not args.all_steps:
        # Set default behavior
        print("Running default pipeline (core steps only)")
        print("Use --all-steps to run all steps, or --steps to specify particular steps")
    
    runner = PipelineRunner(args.workspace)
    
    print("🎯 PointStream Modular Pipeline Runner")
    print(f"Workspace: {runner.workspace_dir}")
    
    success = runner.run_pipeline(args)
    
    if success:
        print("\n🎉 Pipeline completed successfully!")
        sys.exit(0)
    else:
        print("\n💥 Pipeline failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
