# Nutrition5k XAI: Food Image Analysis with Explainable AI

A Streamlit web application is available at [https://foodnutrition7.streamlit.app/](https://foodnutrition7.streamlit.app/) for analyzing food images and generating **explainable AI** explanations using Grad-CAM and counterfactuals.
The trained model is available at [https://huggingface.co/closear/foodnutritionperdish](https://huggingface.co/closear/foodnutritionperdish).

## Features

### 1. Grad-CAM Feature Importance
- Upload a food image
- Get predictions for all 5 nutrition outputs (calories, mass, fat, carb, protein)
- Select any output and visualize feature importance with Grad-CAM heatmaps
- Understand which image regions drive predictions

### 2. Counterfactual Analysis
- Upload a food image
- Specify a target calorie value
- Generate a minimal pixel-space counterfactual showing what image changes would alter the prediction
- View original, counterfactual, heatmap overlay, and masked visualization
- Adjustable threshold to highlight high-sensitivity regions

## Setup & Installation

### Requirements
- Python 3.8+
- CUDA (optional, for GPU acceleration)

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/nutrition5k-xai.git
cd nutrition5k-xai
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download model checkpoint
The model checkpoint `convnext_tiny_nutrition5k_best.pt` (~100MB) should be placed in the repository root.

If using Git LFS:
```bash
git lfs install
git lfs pull
```

Or download manually from [Hugging Face] and place in the root directory.

### 4. Run locally
```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

## Model Details

- **Architecture:** ConvNeXt-Tiny (ImageNet pretrained)
- **Dataset:** Nutrition5k (overhead + side-view food images)
- **Targets:** 5D regression (calories, mass, fat, carb, protein)
- **Loss:** Huber loss on z-scored targets
- **Training:** Early stopping on validation calorie MAE per dish
- **Best checkpoint:** Saved from epoch with lowest per-dish validation error

## XAI Methods

### Grad-CAM (Gradient-weighted Class Activation Mapping)
- Gradient-based feature importance
- Shows which regions of the image influence the prediction
- Implemented via backpropagation through the final convolutional layer

### Counterfactuals (CEM-style)
- Gradient-based pixel-space optimization
- Finds minimal image changes to shift predictions toward a target value
- Regularization: L2 (stay close to original) + Total Variation (smooth changes)
- Stability: Sigmoid parameterization to keep pixels in [0,1]

## File Structure

```
nutrition5k-xai/
├── app.py                              # Main Streamlit application
├── config.py                           # Configuration constants
├── utils.py                            # Helper functions (XAI, visualization)
├── requirements.txt                    # Python dependencies
├── .gitignore                          # Git ignore patterns
├── convnext_tiny_nutrition5k_best.pt   # Model checkpoint (use Git LFS)
├── nutrition5k_main.ipynb              # Original Jupyter notebook (optional)
└── README.md                           # This file
```

## Usage Example

### Tab 1: Grad-CAM
1. Upload a food image
2. View predictions for all 5 nutrition outputs
3. Select "Calories" (or another output)
4. Click "Generate Grad-CAM"
5. View heatmap showing calorie-driving regions

### Tab 2: Counterfactuals
1. Upload a food image
2. Observe current calorie prediction
3. Drag the slider to set a target (e.g., 350 calories)
4. Adjust heatmap threshold (0.6 is default)
5. Click "Generate Counterfactual"
6. View 4 panels: original, overlay, counterfactual, masked image





