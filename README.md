# 🧠 Segmentation Evaluation with Dice Coefficient for segmentation models

This project demonstrates how to compute evaluation metrics like **Dice Coefficient**, **Jaccard Index (IoU)**, and **Accuracy** for segmentation models using a synthetic dataset of binary masks.

## 📌 Overview

We simulate a simple medical-image-style segmentation task using circular binary masks. A logistic regression model is trained to classify pixels as foreground (1) or background (0), and segmentation performance is evaluated using common metrics.

### 🎯 Key Features
- Synthetic binary mask generation
- Pixel-wise logistic regression model
- Dice Coefficient & Jaccard Index calculation
- Accuracy score
- Visual comparison of ground truth and predicted masks

---

## 📁 Files

- `segmentation_metrics.py` — Main script with data generation, training, evaluation, and visualization.
- `README.md` — This file.
- Output: Visualization of predicted vs ground truth masks.

---

## ⚙️ Requirements

Install dependencies using:

```bash
pip install numpy matplotlib scikit-learn
