# Tetst Task IT-Jim | Task_№1

_Pavlo Kukurik_

[Git](https://github.com/PavloKukurik/IT-Jim_TT/tree/master)


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

---

### Results

- Final model: `MobileNetV2` (fine-tuned)
- **Micro F1 score**: 0.9056 (on validation split)
- **Average confidence on test predictions**: 0.799
- This indicates that the model was generally confident in its predictions.


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

