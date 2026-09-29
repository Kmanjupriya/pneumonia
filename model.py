import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image

# ---------------------------------------------------------
# 1. CLAHE Preprocessing Function
# ---------------------------------------------------------
def apply_clahe(image_path):
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE)
    to enhance lung field visibility in chest X-rays.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image from path: {image_path}")
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_img = clahe.apply(img)
    
    # Convert single channel grayscale to 3-channel RGB PIL Image
    rgb_img = cv2.cvtColor(enhanced_img, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(rgb_img)

# ---------------------------------------------------------
# 2. Hybrid DenseNet-121 + Transformer Architecture
# ---------------------------------------------------------
class HybridPneumoniaModel(nn.Module):
    def __init__(self, num_classes=2):
        super(HybridPneumoniaModel, self).__init__()
        
        # Base Backbone: DenseNet-121
        densenet = models.densenet121(weights=None)
        self.densenet = densenet
        self.features = densenet.features
        
        feature_dim = 1024
        
        # Renamed to match state_dict: transformer_layer
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True
        )
        
        # Pooling & Classification Head renamed to match state_dict: classifier
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        feat_map = self.features(x)  # [Batch, 1024, 7, 7]
        
        b, c, h, w = feat_map.shape
        feat_seq = feat_map.view(b, c, h * w).permute(0, 2, 1)
        
        # Global Attention Processing via single layer
        trans_out = self.transformer_layer(feat_seq)  # [Batch, 49, 1024]
        
        trans_out = trans_out.permute(0, 2, 1).view(b, c, h, w)
        
        pooled = self.global_pool(trans_out).view(b, -1)
        logits = self.classifier(pooled)
        return logits

# ---------------------------------------------------------
# 3. Grad-CAM Explainability Module (Fixed Dimensions)
# ---------------------------------------------------------
def generate_gradcam(model, input_tensor, original_image_tensor, target_category=None):
    model.eval()
    
    # Target DenseNet-121 final convolutional feature block
    target_layer = model.densenet.features.denseblock4
    
    gradients = []
    activations = []

    def save_gradient(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def save_activation(module, input, output):
        activations.append(output)

    # Register hooks
    handle_forward = target_layer.register_forward_hook(save_activation)
    handle_backward = target_layer.register_full_backward_hook(save_gradient)

    # Forward pass
    output = model(input_tensor)
    if target_category is None:
        target_category = torch.argmax(output, dim=1).item()

    # Backward pass
    model.zero_grad()
    loss = output[0, target_category]
    loss.backward()

    # Extract gradients and feature map activations
    grads = gradients[0].cpu().data.numpy()[0]
    acts = activations[0].cpu().data.numpy()[0]

    # Global Average Pooling of Gradients for Channel Importance Weights
    weights = np.mean(grads, axis=(1, 2))
    cam = np.zeros(acts.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * acts[i, :, :]

    # Apply ReLU activation and normalize heatmap
    cam = np.maximum(cam, 0)
    if np.max(cam) > 0:
        cam = cam / np.max(cam)

    # Remove hooks
    handle_forward.remove()
    handle_backward.remove()

    # Resize heatmap to standard 224x224 shape
    cam = cv2.resize(cam, (224, 224))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Format original input tensor back into 224x224 RGB uint8 image
    orig = original_image_tensor.squeeze().cpu().detach().numpy()
    if orig.ndim == 3:
        orig = np.transpose(orig, (1, 2, 0))
    
    # Scale intensity values to 0-255 uint8 format
    orig = (orig - orig.min()) / (orig.max() - orig.min() + 1e-8)
    orig = np.uint8(255 * orig)

    # Force identical sizing and 3-channel structure before blending
    orig = cv2.resize(orig, (224, 224))
    heatmap = cv2.resize(heatmap, (224, 224))

    # Blend original preprocessed X-ray with Grad-CAM heatmap
    overlay = cv2.addWeighted(orig, 0.6, heatmap, 0.4, 0)

    return overlay, target_category

if __name__ == '__main__':
    print("Model architecture & Grad-CAM module loaded successfully.")