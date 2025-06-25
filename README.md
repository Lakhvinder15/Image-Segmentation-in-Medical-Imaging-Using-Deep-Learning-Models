# Image Classification for Brain Tumor Diagnosis Using Deep Learning

## Overview

This project applies deep learning techniques to classify brain MRI images into three categories: **glioma**, **meningioma**, and **tumor**. Originally intended for image segmentation, the task was redefined as a **multi-class image classification problem** due to the absence of segmentation masks. The work demonstrates how artificial intelligence (AI) can be used to support accurate and scalable medical diagnostics.

---

## Objectives

- Build a Convolutional Neural Network (CNN) for brain tumor classification.
- Preprocess and prepare real-world medical imaging data.
- Evaluate model performance using standard classification metrics.
- Illustrate the potential of AI in clinical diagnostic support.

---

## Key Contributions

### 🔁 Problem Reformulation
- Adapted the project from segmentation to classification based on dataset limitations.
- Maintained clinical relevance by focusing on categorizing tumor types.

### 🧠 Data Preparation
- Dataset: Labeled MRI scans from [Kaggle](https://www.kaggle.com/datasets/orvile/brain-cancer-mri-dataset)
- Preprocessing: Image resizing (128x128), normalization, and train-test split (80:20)
- Ensured class balance using stratified sampling.

### 🧱 Model Architecture
- Frameworks: TensorFlow and Keras
- Model: Custom CNN with convolutional, pooling, ReLU, and Softmax layers
- Chosen for efficiency and interpretability over large pre-trained models

### 📈 Training & Evaluation
- Trained over 10 epochs
- Achieved ~80% test accuracy
- Evaluation metrics: Accuracy, precision, recall, F1-score, confusion matrix
- Visualized sample predictions with color-coded labels (green = correct, red = incorrect)

### ⚙️ Practical Impact
- Demonstrates how CNNs can aid radiologists in diagnosis and decision-making
- Provides a scalable solution for high-throughput medical image classification

### 🚧 Challenges & Future Work
- Limitation: Lack of segmentation masks restricted project to classification
- Future scope:
  - Incorporate pixel-level segmentation using annotated datasets
  - Integrate advanced architectures (e.g., U-Net, ResNet)
  - Extend to other clinical imaging tasks like surgical planning or anomaly detection

---

## Tools & Technologies

- **Language:** Python
- **Libraries:** TensorFlow, Keras, NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn
- **Dataset:** [Brain Tumor MRI Dataset (Kaggle)](https://www.kaggle.com/datasets/orvile/brain-cancer-mri-dataset)

---

## Conclusion

This project highlights the power of deep learning in medical imaging. A custom CNN model was successfully developed to classify brain tumor types from MRI scans with high accuracy. The approach is lightweight, scalable, and suitable for integration into clinical workflows, marking a step toward AI-assisted healthcare.

---

## Author

**Lakhvinder Kaur**  
Programme: Computer Vision and Artificial Intelligence 
