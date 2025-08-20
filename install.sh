#!/bin/bash

if [ "$(python3 --version | grep -c '3\.10')" -lt 1 ]; then
  echo "You must use Python version == 3.10 for this script. If not, your installation will fail or may not work properly."
  exit 1
fi

pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121

pip install opencv-python ultralytics "numpy<2.0"
