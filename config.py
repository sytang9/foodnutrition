"""Configuration constants for Nutrition5k XAI application."""

import torch

# Image processing
IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Model and targets
TARGET_COLS = ['calories', 'mass', 'fat', 'carb', 'protein']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Counterfactual parameters
CF_STEPS = 200
CF_LR = 0.02
CF_LAM_L2 = 10.0
CF_LAM_TV = 0.05
CF_GRAD_CLIP = 1.0

# Grad-CAM parameters
GRADCAM_DOWN_SIZE = 28
GRADCAM_Q_LOW = 60.0
GRADCAM_Q_HIGH = 99.7
GRADCAM_GAMMA = 0.45

# Visualization
OVERLAY_ALPHA = 0.35
HEATMAP_CMAP = 'jet'
