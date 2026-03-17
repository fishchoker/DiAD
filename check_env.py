import os
import sys
import subprocess

def run_cmd(cmd):
    print(f"\n$ {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(result.stdout)
        if result.stderr:
            print("⚠️ STDERR:")
            print(result.stderr)
    except Exception as e:
        print(f"❌ Failed: {e}")

print("="*60)
print("🐍 Python Environment Info")
print("="*60)
print("Python:", sys.version)
print("Executable:", sys.executable)

print("\n" + "="*60)
print("📦 Conda / Pip Packages")
print("="*60)

run_cmd("which python")
run_cmd("pip --version")
run_cmd("conda info --envs")

print("\n--- pip list (torch相关) ---")
run_cmd("pip list | grep -E 'torch|mkl|intel|openmp'")

print("\n--- conda list (关键库) ---")
run_cmd("conda list | grep -E 'torch|mkl|intel|openmp'")

print("\n" + "="*60)
print("🔥 Torch Import Test")
print("="*60)

try:
    import torch
    print("✅ torch imported successfully!")
    print("torch version:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
except Exception as e:
    print("❌ torch import FAILED:")
    print(e)

print("\n" + "="*60)
print("🔍 Dynamic Library Check (libtorch_cpu.so)")
print("="*60)

try:
    import torch
    torch_path = os.path.dirname(torch.__file__)
    lib_path = os.path.join(torch_path, "lib", "libtorch_cpu.so")

    print("libtorch_cpu.so path:", lib_path)

    if os.path.exists(lib_path):
        run_cmd(f"ldd {lib_path} | grep -E 'iomp|mkl|not found'")
    else:
        print("❌ libtorch_cpu.so not found")
except Exception as e:
    print("❌ Cannot check libtorch:", e)

print("\n" + "="*60)
print("🌐 LD_LIBRARY_PATH")
print("="*60)

print(os.environ.get("LD_LIBRARY_PATH", "Not Set"))

print("\n" + "="*60)
print("🧠 Intel Runtime Check")
print("="*60)

run_cmd("ldconfig -p | grep -E 'iomp|itt|mkl'")

print("\n" + "="*60)
print("✅ Done")
print("="*60)