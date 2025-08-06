#!/usr/bin/env python3
"""
Test runner for PointStream pipeline.
Runs comprehensive tests before pipeline execution to ensure system integrity.
"""

import subprocess
import sys
import time
from pathlib import Path
import argparse


def run_command(command, description="", timeout=300):
    """
    Run a command with timeout and error handling.
    
    Args:
        command (list): Command to run
        description (str): Description of the command
        timeout (int): Timeout in seconds
    
    Returns:
        tuple: (success, output, error)
    """
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(command)}")
    print(f"{'='*60}")
    
    try:
        start_time = time.time()
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=Path(__file__).parent
        )
        end_time = time.time()
        
        print(f"Duration: {end_time - start_time:.2f}s")
        print(f"Return code: {result.returncode}")
        
        if result.stdout:
            print(f"STDOUT:\n{result.stdout}")
        
        if result.stderr:
            print(f"STDERR:\n{result.stderr}")
        
        return result.returncode == 0, result.stdout, result.stderr
        
    except subprocess.TimeoutExpired:
        print(f"ERROR: Command timed out after {timeout}s")
        return False, "", f"Timeout after {timeout}s"
    except Exception as e:
        print(f"ERROR: Failed to run command: {e}")
        return False, "", str(e)


def check_test_dependencies():
    """Check if required dependencies for testing are available."""
    print("\nChecking test dependencies...")
    
    required_packages = [
        ('pytest', 'pytest'),
        ('opencv-python', 'cv2'),
        ('numpy', 'numpy')
    ]
    missing_packages = []
    
    for package_name, import_name in required_packages:
        try:
            __import__(import_name)
            print(f"✓ {package_name}")
        except ImportError:
            missing_packages.append(package_name)
            print(f"✗ {package_name} (missing)")
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Install them with: pip install " + ' '.join(missing_packages))
        return False
    
    return True


def run_tests(test_type="fast", verbose=False):
    """
    Run tests based on the specified type.
    
    Args:
        test_type (str): Type of tests to run ('fast', 'all', 'integration')
        verbose (bool): Whether to run tests in verbose mode
    
    Returns:
        bool: True if all tests passed
    """
    if not check_test_dependencies():
        return False
    
    # Base pytest command
    cmd = ["python", "-m", "pytest"]
    
    if verbose:
        cmd.append("-v")
    
    # Add specific test markers based on type
    if test_type == "fast":
        cmd.extend(["-m", "not slow and not integration"])
        description = "Fast unit tests"
    elif test_type == "all":
        description = "All tests (including slow ones)"
    elif test_type == "integration":
        cmd.extend(["-m", "integration"])
        description = "Integration tests only"
    else:
        print(f"Unknown test type: {test_type}")
        return False
    
    # Add test directory
    cmd.append("tests/")
    
    # Set timeout based on test type
    timeout = 120 if test_type == "fast" else 600
    
    success, stdout, stderr = run_command(cmd, description, timeout)
    
    if success:
        print(f"\n✓ {description} PASSED")
    else:
        print(f"\n✗ {description} FAILED")
    
    return success


def run_code_quality_checks():
    """Run code quality checks (linting, formatting)."""
    print("\nRunning code quality checks...")
    
    checks = [
        {
            "cmd": ["python", "-m", "flake8", "pointstream/", "--max-line-length=100", "--ignore=E203,W503"],
            "description": "Flake8 linting",
            "optional": True
        },
        {
            "cmd": ["python", "-c", "import pointstream; print('✓ Package imports successfully')"],
            "description": "Package import test",
            "optional": False
        }
    ]
    
    all_passed = True
    
    for check in checks:
        success, _, _ = run_command(
            check["cmd"], 
            check["description"], 
            timeout=30
        )
        
        if not success:
            if check["optional"]:
                print(f"WARNING: Optional check '{check['description']}' failed")
            else:
                print(f"ERROR: Required check '{check['description']}' failed")
                all_passed = False
    
    return all_passed


def main():
    parser = argparse.ArgumentParser(description="Run PointStream tests")
    parser.add_argument(
        "--type", 
        choices=["fast", "all", "integration"], 
        default="fast",
        help="Type of tests to run"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Run tests in verbose mode"
    )
    parser.add_argument(
        "--skip-quality-checks",
        action="store_true",
        help="Skip code quality checks"
    )
    
    args = parser.parse_args()
    
    print("PointStream Test Runner")
    print("="*40)
    
    # Run code quality checks
    if not args.skip_quality_checks:
        quality_passed = run_code_quality_checks()
        if not quality_passed:
            print("\n❌ Code quality checks failed")
            sys.exit(1)
    
    # Run tests
    tests_passed = run_tests(args.type, args.verbose)
    
    if tests_passed:
        print("\n✅ All tests passed! Pipeline is ready to run.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please fix issues before running pipeline.")
        sys.exit(1)


if __name__ == "__main__":
    main()
