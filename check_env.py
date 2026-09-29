import PIL
import cv2
import flask
import sys
import timm
import torch

print("=" * 60)
print("  PNEUMONIA DIAGNOSTIC SYSTEM - ENVIRONMENT CHECK")
print("=" * 60)
print(f"[✓] Python Version   : {sys.version.split()[0]}")
print(f"[✓] PyTorch Version  : {torch.__version__}")
print(f"[✓] OpenCV Version   : {cv2.__version__}")
print(f"[✓] Flask Version    : {flask.__version__}")
print(f"[✓] TIMM (ViT) Ver   : {timm.__version__}")
print(f"[✓] Pillow Version   : {PIL.__version__}")
print("=" * 60)

# Verify PyTorch tensor creation
x = torch.rand(2, 3)
print("[✓] PyTorch Tensor Allocation Test: PASSED")

print("=" * 60)
print("CONFIRMATION SUCCESSFUL: All dependencies ready!")
print("=" * 60)