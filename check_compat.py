"""
Compatibility checker for RS Pipeline dependencies.
Helps identify and fix version conflicts.
"""

import sys
import subprocess
from packaging import version


def get_package_version(package_name):
    """Get installed package version."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", package_name],
            capture_output=True,
            text=True
        )
        for line in result.stdout.split('\n'):
            if line.startswith('Version:'):
                return line.split(':')[1].strip()
    except:
        return None


def check_compatibility():
    """Check all package versions and compatibility."""
    print("=" * 70)
    print("RS PIPELINE COMPATIBILITY CHECK")
    print("=" * 70)
    
    packages = {
        'torch': {'min': '2.0.0', 'recommended': '2.6.0+'},
        'vllm': {'min': '0.6.3', 'recommended': '0.11.0+'},
        'transformers': {'min': '4.40.0', 'recommended': 'latest'},
        'opencv-python': {'min': '4.5.0', 'recommended': '4.9.0+'},
        'pillow': {'min': '9.0.0', 'recommended': '10.0.0+'},
        'qwen-vl-utils': {'min': None, 'recommended': 'latest'},
    }
    
    results = []
    all_ok = True
    
    print("\n📦 Package Versions:\n")
    
    for pkg, reqs in packages.items():
        installed = get_package_version(pkg)
        
        if installed is None:
            print(f"✗ {pkg:20s} NOT INSTALLED")
            results.append((pkg, None, 'missing'))
            all_ok = False
        else:
            status = "✓"
            issue = None
            
            if reqs['min']:
                try:
                    if version.parse(installed.split('+')[0]) < version.parse(reqs['min']):
                        status = "⚠"
                        issue = f"outdated (min: {reqs['min']})"
                        all_ok = False
                except:
                    pass
            
            print(f"{status} {pkg:20s} {installed:15s} ", end="")
            if issue:
                print(f"← {issue}")
            else:
                print()
            
            results.append((pkg, installed, issue))
    
    # Check specific compatibility issues
    print("\n🔍 Compatibility Checks:\n")
    
    torch_ver = get_package_version('torch')
    vllm_ver = get_package_version('vllm')
    
    if torch_ver and vllm_ver:
        torch_major = torch_ver.split('.')[0]
        torch_minor = torch_ver.split('.')[1]
        vllm_major = vllm_ver.split('.')[0]
        vllm_minor = vllm_ver.split('.')[1]
        
        # Check known incompatibilities
        if vllm_ver.startswith('0.11') or vllm_ver.startswith('0.10'):
            if not torch_ver.startswith('2.4') and not torch_ver.startswith('2.5'):
                print("⚠ vLLM 0.11.x typically requires PyTorch 2.4+")
                all_ok = False
        
        if vllm_ver.startswith('0.6'):
            print("⚠ vLLM 0.6.3 may not support Qwen3-VL-30B models")
            print("  → Upgrade: pip install vllm>=0.11.0")
            all_ok = False
    
    # Check CUDA
    print("\n🎮 GPU Check:\n")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  GPU count: {torch.cuda.device_count()}")
        else:
            print("✗ CUDA not available")
            all_ok = False
    except ImportError:
        print("✗ PyTorch not installed")
        all_ok = False
    
    # Recommendations
    print("\n" + "=" * 70)
    if all_ok:
        print("✓ ALL CHECKS PASSED - Ready to use!")
    else:
        print("⚠ ISSUES FOUND - See recommendations below:")
        print("\n💡 Recommended Actions:\n")
        
        for pkg, ver, issue in results:
            if issue == 'missing':
                print(f"  pip install {pkg}")
            elif issue:
                print(f"  pip install --upgrade {pkg}")
        
        if vllm_ver and vllm_ver.startswith('0.6'):
            print("\n  For Qwen3-VL support:")
            print("  pip uninstall -y vllm")
            print("  pip install vllm>=0.11.0")
    
    print("=" * 70)
    return all_ok


if __name__ == "__main__":
    check_compatibility()
