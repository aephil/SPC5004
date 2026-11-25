#!/usr/bin/env bash
# Install what the lec19-22 jobs need (NVIDIA / CUDA). Keeps the machine's own PyTorch.
set -euo pipefail
cd "$(dirname "$0")"
if ! python -c "import torch" 2>/dev/null; then
  echo "PyTorch not found: installing the default (CUDA) build"
  pip install torch torchvision
fi
python -c "import torch; assert torch.cuda.is_available(), 'PyTorch cannot see a GPU'; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
pip install -r requirements.txt
# Only for the pose ControlNet; the job skips pose if this is missing. It needs torchvision, and installing
# torchvision from PyPI could replace the GPU build of PyTorch, so only go ahead if torchvision is already there.
if python -c "import torchvision" 2>/dev/null; then
  pip install "controlnet_aux==0.0.10" || echo "controlnet_aux did not install: the pose example will be skipped"
  # controlnet_aux pulls in opencv-python, which needs libGL and often fails on headless servers: swap it back
  pip uninstall -y opencv-python >/dev/null 2>&1 || true
  pip install --force-reinstall --no-deps opencv-python-headless
else
  echo "torchvision not installed: skipping controlnet_aux, so the pose example will be skipped"
fi
python -c "import torch; print('torch still', torch.__version__)"   # check pip did not replace the GPU build
accelerate config default >/dev/null 2>&1 || true
