# Segmentation Evaluation Using Dice Coefficient and IoU

This project shows how to evaluate **image segmentation models** using metrics like **Dice Coefficient**, **Jaccard Index (IoU)**, and **Accuracy**, with a simple synthetic dataset made of circular binary masks.

---

## 📌 Overview

We simulate a basic segmentation task—like detecting objects in medical images—using randomly generated circles in images. A **Logistic Regression** model is trained to classify each pixel as either foreground (1) or background (0). Then, we evaluate how well it performs using standard segmentation metrics.

---

## 🚀 Features

- ✅ Generate synthetic binary masks (circle shapes)
- 📊 Train a pixel-wise **logistic regression** model
- 📐 Compute key evaluation metrics:
  - **Dice Coefficient**
  - **Jaccard Index (Intersection over Union)**
  - **Accuracy**
- 🖼️ Visual comparison of ground truth vs predicted masks

---

## 📁 Files Included

- `segmentation_metrics.py` – Python script for data creation, training, evaluation, and visualization
- `README.md` – Project description and instructions
- 📷 Output: Image showing the ground truth and predicted mask side by side

---

## ⚙️ Installation & Requirements

Make sure you have Python installed, then install the required libraries:

```bash
pip install numpy matplotlib scikit-learn
