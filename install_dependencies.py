# Install the correct version of Pytorch Geometric.
import subprocess
import sys
import torch

def format_pytorch_version(version):
  return version.split('+')[0]

TORCH_version = torch.__version__
TORCH = format_pytorch_version(TORCH_version)

def format_cuda_version(version):
  return 'cu' + version.replace('.', '')

CUDA_version = torch.version.cuda
CUDA = format_cuda_version(CUDA_version)

subprocess.check_call([sys.executable, "-m", "pip", "install","-r", "requirements.txt"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "torch-scatter", "-f", f"https://data.pyg.org/whl/torch-{TORCH}+{CUDA}.html"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "torch-sparse", "-f", f"https://data.pyg.org/whl/torch-{TORCH}+{CUDA}.html"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "torch-cluster", "-f", f"https://data.pyg.org/whl/torch-{TORCH}+{CUDA}.html"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "torch-spline-conv", "-f", f"https://data.pyg.org/whl/torch-{TORCH}+{CUDA}.html"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "torch-geometric"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "torch-geometric"])