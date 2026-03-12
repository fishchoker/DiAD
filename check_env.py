
import sys
import importlib.metadata
import torch

def check_dependencies():
    """
    Checks for the presence and version of specified Python packages.
    """
    print("\n--- Checking Critical Dependencies ---")
    
    # List of required packages and their minimum versions
    required_packages = {
        "transformers": "4.20.0",
        "safetensors": "0.3.0",
        "pytorch_lightning": "2.0.0",
        "open_clip": "2.7.0",
        "xformers": "0.0.20",
        "opencv-python": "4.5.0"  # Note: package name is opencv-python, but imported as cv2
    }
    
    all_ok = True
    for package, min_version in required_packages.items():
        try:
            # Check if the package is installed and get its version
            version = importlib.metadata.version(package)
            
            # Simple version comparison
            if version >= min_version:
                print(f"✅ {package} ({version}) is installed and meets version requirements (>= {min_version}).")
            else:
                print(f"⚠️ {package} ({version}) is installed but version is older than required (>= {min_version}).")
                all_ok = False
                
        except importlib.metadata.PackageNotFoundError:
            print(f"❌ {package} is not installed. Please install it by running: pip install {package}>={min_version}")
            all_ok = False
            
    if all_ok:
        print("\n✅ All critical dependencies are installed and up-to-date.")
    else:
        print("\n❌ Some dependencies are missing or outdated. Please install/update them as suggested above.")
        
    return all_ok

def check_pytorch_cuda():
    """
    Checks PyTorch version, CUDA availability, and GPU details.
    """
    print("--- Checking PyTorch and CUDA ---")
    
    try:
        # Check PyTorch version
        pt_version = torch.__version__
        print(f"PyTorch Version: {pt_version}")
        if not pt_version.startswith("2.6"):
            print(f"⚠️ PyTorch version is {pt_version}, but 2.6.x is recommended for optimal RTX 5090 support.")
        else:
            print("✅ PyTorch version is 2.6.x.")

        # Check CUDA availability
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            gpu_name = torch.cuda.get_device_name(0)
            compute_cap = torch.cuda.get_device_capability(0)
            
            print(f"✅ CUDA is available.")
            print(f"   - CUDA Version (from PyTorch): {cuda_version}")
            print(f"   - GPU Detected: {gpu_name}")
            print(f"   - Compute Capability: {compute_cap[0]}.{compute_cap[1]}")

            # Check for Blackwell architecture (Compute Capability 12.x for 50-series)
            if compute_cap[0] >= 12:
                print("✅ GPU Architecture: Blackwell (RTX 50-series) detected.")
            elif compute_cap[0] >= 9:
                print("✅ GPU Architecture: Hopper/Ada detected.")
            else:
                print("ℹ️ GPU Architecture: Older generation detected.")

            # Test a basic CUDA operation
            try:
                tensor = torch.randn(3, 3).cuda()
                _ = tensor * tensor
                print("✅ Basic CUDA tensor operations are working correctly.")
            except Exception as e:
                print(f"❌ A test CUDA operation failed: {e}")
        else:
            print("❌ CUDA is not available. Check your NVIDIA drivers and ensure you have a CUDA-enabled PyTorch build.")
            
    except ImportError:
        print("❌ PyTorch is not installed. Please install it to proceed.")
        return

def check_python_version():
    """
    Checks if the Python version is compatible.
    """
    print("--- Checking System Information ---")
    
    py_version = sys.version.split()[0]
    print(f"Python Version: {py_version}")
    
    if sys.version_info < (3, 10):
        print("❌ Python version is older than 3.10. PyTorch 2.6 and other modern libraries may require a newer version.")
    else:
        print("✅ Python version is 3.10 or newer.")

def main():
    """
    Main function to run all environment checks.
    """
    print("="*50)
    print("RTX 5090 Environment Compatibility Check")
    print("="*50)
    
    check_python_version()
    check_pytorch_cuda()
    check_dependencies()
    
    print("\n" + "="*50)
    print("Check complete.")

if __name__ == "__main__":
    main()
