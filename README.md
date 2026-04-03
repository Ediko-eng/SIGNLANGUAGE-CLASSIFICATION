# 🤟 Hand Sign Classification — KNN

*A Machine Learning system designed to recognize and classify Sign Language hand gestures (ASL) using the K-Nearest Neighbors (KNN) algorithm.*

---

## 📖 Project Overview
This project focuses on the automated recognition of hand signs. It transforms raw visual data (images of hand gestures) into mathematical arrays to train a model capable of predicting the correct alphabet or sign. By leveraging computer vision and the **K-Nearest Neighbors (KNN)** algorithm, the system identifies patterns in hand shapes to bridge the communication gap for the deaf and hard-of-hearing community.

## 🚀 Key Features
* **Sign Language Preprocessing:** Standardizes hand gesture images to a uniform 400x400 resolution for consistency.
* **Array Transformation:** Converts complex RGB visual data into structured NumPy arrays for mathematical analysis.
* **KNN Classification:** Uses proximity-based learning to classify signs based on their similarity to known training examples.
* **Performance Analysis:** Features a detailed **Confusion Matrix (Matriz Konfuzaun)** in Tetum to evaluate the model's accuracy across different hand signs.

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Core Libraries:** * `OpenCV (cv2)`: Image processing and resizing.
  * `NumPy`: Numerical matrix operations.
  * `Scikit-Learn`: KNN Model implementation and metrics.
  * `Seaborn/Matplotlib`: Confusion matrix visualization.

## 📁 Repository Structure
* **`Resizing.ipynb`**: Handles the batch processing and resizing of raw hand sign images.
* **`Transform_image_to_array.ipynb`**: Converts images to numerical features, trains the KNN model, and generates the confusion matrix.
* **`matriz_konfuzaun_knn_output.png`**: The final evaluation result showing the model's performance on sign classification.

---

## ➗ Mathematical Foundations

This machine learning model operates based on three primary mathematical steps:

### 1. Data Normalization (Scaling)
To ensure that light intensity doesn't bias the model, pixel values are scaled from the range $[0, 255]$ to a range of $[0, 1]$.
$$X_{norm} = \frac{X - X_{min}}{X_{max} - X_{min}} \implies X_{norm} = \frac{Pixel}{255}$$

### 2. Euclidean Distance (Similarity Measure)
The "Machine Learning" logic happens by calculating how "close" a new hand sign is to the existing signs in the database. We use the **Euclidean Distance** formula:
$$d(p, q) = \sqrt{\sum_{i=1}^{n} (q_i - p_i)^2}$$
*Where $p$ and $q$ are the pixel vectors of two different hand signs.*

### 3. K-Nearest Neighbors Logic
The classification $Y$ for a new hand sign is determined by the majority vote of its $K$ closest neighbors:
$$\hat{Y} = \text{mode}\{y_i \mid i \in N_k(x)\}$$

---

## 📊 Evaluation Results

The model performance is visualized through a **Confusion Matrix**, which allows us to see exactly which signs are being recognized correctly and which ones are being confused.

![Matriz Konfuzaun](matriz_konfuzaun_knn_output.png)
*Figure 1: Accuracy visualization for Hand Sign Classification (Tetum Version).*

### Performance Summary
| Metric | Formula | Description |
| :--- | :--- | :--- |
| **Accuracy** | $\frac{TP+TN}{Total}$ | Overall correctness of the sign recognition. |
| **Precision** | $\frac{TP}{TP+FP}$ | How many "Sign A" predictions were actually "Sign A". |
| **Recall** | $\frac{TP}{TP+FN}$ | How many of the actual "Sign A" gestures were caught by the model. |

## ⚙️ How to Run
1. Install dependencies: `pip install opencv-python numpy scikit-learn seaborn matplotlib`
2. Run `Resizing.ipynb` to prepare your dataset.
3. Run `Transform_image_to_array.ipynb` to train and test the KNN classifier.

---
**Author:** Edmilson Fabio Valente  
**Department:** Informatic Engineering — UNTL  
**Focus:** Computer Vision & Machine Learning
