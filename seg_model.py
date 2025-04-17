import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Create synthetic segmentation-like dataset (binary masks)
def generate_segmentation_data(n_samples=200, img_size=32):
    X = []
    y = []
    for _ in range(n_samples):
        img = np.zeros((img_size, img_size))
        mask = np.zeros((img_size, img_size))
        cx, cy = np.random.randint(8, 24, size=2)
        radius = np.random.randint(5, 10)
        for i in range(img_size):
            for j in range(img_size):
                if (i - cx) ** 2 + (j - cy) ** 2 < radius ** 2:
                    mask[i, j] = 1
        noise = np.random.rand(img_size, img_size) * 0.3
        img = mask + noise
        X.append(img.flatten())
        y.append(mask.flatten())
    return np.array(X), np.array(y)

# 2. Generate and split data
X, y = generate_segmentation_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Reshape: each pixel is a sample with intensity as feature
X_train_pixels = X_train.reshape(-1, 1)
y_train_pixels = y_train.reshape(-1)
X_test_pixels = X_test.reshape(-1, 1)
y_test_pixels = y_test.reshape(-1)

# 4. Initialize and train logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_pixels, y_train_pixels)

# 5. Predict
y_pred_pixels = model.predict(X_test_pixels)

# 6. Evaluation metrics
def dice_coefficient(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred).sum()
    return 2. * intersection / (y_true.sum() + y_pred.sum() + 1e-8)

def jaccard_index(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()
    return intersection / (union + 1e-8)

# 7. Evaluate model
dice = dice_coefficient(y_test_pixels, y_pred_pixels)
jaccard = jaccard_index(y_test_pixels, y_pred_pixels)
acc = accuracy_score(y_test_pixels, y_pred_pixels)

print(f"🎯 Dice Coefficient: {dice:.4f}")
print(f"📏 Jaccard Index (IoU): {jaccard:.4f}")
print(f"✅ Accuracy: {acc:.4f}")

# 8. Visualize one example (from test set)
img_size = 32
sample_index = 0

true_mask = y_test[sample_index].reshape((img_size, img_size))
start = sample_index * img_size * img_size
end = (sample_index + 1) * img_size * img_size
pred_mask = y_pred_pixels[start:end].reshape((img_size, img_size))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
ax1.imshow(true_mask, cmap='gray')
ax1.set_title('Ground Truth Mask')
ax1.axis('off')

ax2.imshow(pred_mask, cmap='gray')
ax2.set_title('Predicted Mask')
ax2.axis('off')

plt.tight_layout()
plt.show()