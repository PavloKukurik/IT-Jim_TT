# Tetst Task IT-Jim | Task_№1

_Pavlo Kukurik_

## Task 1 — DL (MANDATORY)

This repository contains my solution to the mandatory DL task from It-Jim's 2025 internship challenge. The goal was to build a binary classifier that can distinguish images with visible artifacts from those without, based on a dataset of AI-generated faces.

---

### Problem Description

Some AI-generated face images contain subtle or obvious visual artifacts — things like distorted features, incorrect facial anatomy, or unnatural textures. The objective is to automatically detect these images using a binary classification model.

---

### Project Structure

```
artifact_classifier/
├── data/
│   ├── train/                  # training images (with class labels in filename)
│   └── test/                   # test images (no labels)
│
├── notebooks/
│   └── 01_train_baseline.ipynb  # full pipeline with training and validation
│
├── src/
│   ├── train.py               # CLI training script (baseline)
│   ├── inference.py           # script to run predictions on test set
│
├── outputs/
│   ├── models/                # saved .h5 model files
│   └── results/               # prediction outputs (csv)
│
├── requirements.txt
└── README.md
```

---

### Approach

- Used MobileNetV2 with ImageNet weights as a feature extractor
- Implemented two-phase training:
  1. Feature extractor frozen — only classification head trained
  2. Fine-tuned last 20 layers of the base model at lower learning rate
- Handled class imbalance with:
  - Class weighting during training
  - Light augmentation (flip, rotation, brightness)
- Focused on the micro F1 score as the main evaluation metric

All core logic and results are available in the notebook.

---

### Results

- Final model: `MobileNetV2` (fine-tuned)
- **Micro F1 score**: 0.9056 (on validation split)

Confusion matrix on validation:
```
              Pred 0   Pred 1
True 0         14        13
True 1         21       312
```

- **Average confidence on test predictions**: 0.799
- This indicates that the model was generally confident in its predictions.

To support this, a histogram of prediction confidences was plotted (see figure below). It shows a strong bias toward high-confidence outputs, especially for class 1 predictions.

📊 Suggested visualizations to include:
- Histogram of `confidence` values for all predictions
- (Optional) Overlayed histograms or KDEs by predicted class (0 vs 1)

---

### How to Run

### Install dependencies:
```bash
pip install -r requirements.txt
```

### Run inference:
```bash
python3 src/inference.py
```

Predictions will be saved to `outputs/results/test_predictions.csv`.

