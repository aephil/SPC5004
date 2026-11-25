#!/usr/bin/env bash
# Install what the lec18 scripts need. Works on NVIDIA (CUDA) or AMD (ROCm) machines.
set -euo pipefail
cd "$(dirname "$0")"
# Keep an existing GPU build of PyTorch (most cloud images have one). Only install it if it's missing.
if ! python -c "import torch" 2>/dev/null; then
  echo "PyTorch not found: installing the default (NVIDIA CUDA) build"
  pip install torch torchvision
fi
python -c "import torch; assert torch.cuda.is_available(), 'PyTorch cannot see a GPU'; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'hip', getattr(torch.version, 'hip', None))"
python -c "import torchvision" 2>/dev/null || { echo "torchvision is missing: install the build that matches this PyTorch (see pytorch.org), then rerun"; exit 1; }
pip install -r requirements.txt
# 8-bit Adam is optional; the scripts test it and fall back to plain AdamW.
pip install "bitsandbytes>=0.48" || echo "bitsandbytes did not install; continuing without 8-bit Adam"
python -c "import torch; print('torch still', torch.__version__)"   # check pip did not replace the GPU build
