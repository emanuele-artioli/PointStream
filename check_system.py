#!/usr/bin/env python3
"""
PointStream System Status Check

Verifies that all components are properly installed and compatible.
Run this after installation or when experiencing issues.
"""

import sys
import subprocess
import importlib
from pathlib import Path
import platform


def check_python_version():
    """Check Python version compatibility."""
    print("🐍 Python Version Check")
    print(f"   Python version: {sys.version}")
    version_info = sys.version_info
    
    if version_info >= (3, 10):
        print("   ✅ Python version is compatible")
        return True
    else:
        print("   ❌ Python 3.10+ required")
        return False


def check_package_import(package_name, min_version=None):
    """Check if a package can be imported and meets version requirements."""
    try:
        module = importlib.import_module(package_name)
        version = getattr(module, '__version__', 'unknown')
        
        if min_version and hasattr(module, '__version__'):
            from packaging import version as pkg_version
            if pkg_version.parse(version) >= pkg_version.parse(min_version):
                print(f"   ✅ {package_name}: {version}")
                return True
            else:
                print(f"   ⚠️  {package_name}: {version} (requires >={min_version})")
                return False
        else:
            print(f"   ✅ {package_name}: {version}")
            return True
            
    except ImportError as e:
        print(f"   ❌ {package_name}: Not installed ({e})")
        return False


def check_cuda_setup():
    """Check CUDA and GPU setup."""
    print("\n🔥 CUDA & GPU Check")
    
    # Check nvidia-smi
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            driver_line = [line for line in lines if 'Driver Version:' in line]
            if driver_line:
                print(f"   ✅ NVIDIA Driver: {driver_line[0].split('Driver Version:')[1].split()[0]}")
            else:
                print("   ✅ NVIDIA Driver: Available")
        else:
            print("   ❌ nvidia-smi not available")
            return False
    except FileNotFoundError:
        print("   ❌ nvidia-smi not found")
        return False
    
    # Check PyTorch CUDA
    try:
        import torch
        print(f"   PyTorch version: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   CUDA version (PyTorch): {torch.version.cuda}")
            print(f"   GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
            
            # Test CUDA operation
            try:
                x = torch.randn(2, 3).cuda()
                y = torch.randn(2, 3).cuda()
                z = x + y
                print("   ✅ CUDA operations working")
                return True
            except Exception as e:
                print(f"   ❌ CUDA operation failed: {e}")
                return False
        else:
            print("   ⚠️  CUDA not available in PyTorch")
            return False
    except ImportError:
        print("   ❌ PyTorch not installed")
        return False


def check_mmlab_setup():
    """Check MMlab components."""
    print("\n🔬 MMlab Components Check")
    
    mmlab_packages = [
        ('mmcv', '2.1.0'),
        ('mmdet', '3.2.0'),
        ('mmpose', '1.3.2'),
    ]
    
    all_ok = True
    for package, min_ver in mmlab_packages:
        if not check_package_import(package, min_ver):
            all_ok = False
    
    return all_ok


def check_core_packages():
    """Check core PointStream dependencies."""
    print("\n📦 Core Packages Check")
    
    core_packages = [
        ('numpy', '1.20.0'),
        ('opencv-python', '4.8.0'),  # This imports as cv2
        ('matplotlib', '3.7.0'),
        ('scikit-image', '0.21.0'),  # This imports as skimage
        ('PIL', None),  # Pillow imports as PIL
        ('transformers', '4.36.2'),
        ('ultralytics', '8.0.0'),
    ]
    
    all_ok = True
    for package, min_ver in core_packages:
        # Handle special import names
        import_name = package
        if package == 'opencv-python':
            import_name = 'cv2'
        elif package == 'scikit-image':
            import_name = 'skimage'
        
        if not check_package_import(import_name, min_ver):
            all_ok = False
    
    return all_ok


def check_pointstream_package():
    """Check PointStream package installation."""
    print("\n🎯 PointStream Package Check")
    
    try:
        import pointstream
        print(f"   ✅ PointStream installed: {getattr(pointstream, '__version__', 'development')}")
        
        # Check key modules
        key_modules = [
            'pointstream.config',
            'pointstream.pipeline.stage_01_analyzer',
            'pointstream.models.yolo_handler',
            'pointstream.evaluation.evaluator',
        ]
        
        all_ok = True
        for module in key_modules:
            try:
                importlib.import_module(module)
                print(f"   ✅ {module}")
            except ImportError as e:
                print(f"   ❌ {module}: {e}")
                all_ok = False
        
        return all_ok
        
    except ImportError as e:
        print(f"   ❌ PointStream not installed: {e}")
        return False


def check_file_structure():
    """Check important file structure."""
    print("\n📁 File Structure Check")
    
    base_dir = Path(__file__).parent
    important_paths = [
        'pointstream',
        'tests',
        'artifacts',
        'tests/data',
        'pyproject.toml',
        'requirements.txt',
        'environment.yml',
    ]
    
    all_ok = True
    for path in important_paths:
        full_path = base_dir / path
        if full_path.exists():
            print(f"   ✅ {path}")
        else:
            print(f"   ⚠️  {path} not found")
            if path in ['pointstream', 'pyproject.toml']:
                all_ok = False
    
    return all_ok


def check_test_data():
    """Check test data availability."""
    print("\n🎥 Test Data Check")
    
    base_dir = Path(__file__).parent
    test_video = base_dir / "tests" / "data" / "DAVIS_stitched.mp4"
    
    if test_video.exists():
        size_mb = test_video.stat().st_size / (1024 * 1024)
        print(f"   ✅ Test video: {test_video} ({size_mb:.1f} MB)")
        return True
    else:
        print(f"   ⚠️  Test video not found: {test_video}")
        return False


def main():
    """Run comprehensive system check."""
    print("🔍 PointStream System Status Check")
    print("=" * 50)
    
    checks = [
        ("Python Version", check_python_version),
        ("Core Packages", check_core_packages),
        ("CUDA Setup", check_cuda_setup),
        ("MMlab Components", check_mmlab_setup),
        ("PointStream Package", check_pointstream_package),
        ("File Structure", check_file_structure),
        ("Test Data", check_test_data),
    ]
    
    results = []
    
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"   ❌ Error during {check_name}: {e}")
            results.append((check_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for check_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status} {check_name}")
    
    print(f"\nOverall: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n🎉 All checks passed! System is ready for PointStream.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} issues found. See output above for details.")
        print("\nFor help, check:")
        print("   - README.md installation instructions")
        print("   - CUDA troubleshooting section")
        print("   - Run with verbose conda/pip output")
        return 1


if __name__ == "__main__":
    sys.exit(main())
