# Multi-Modal Thoracic Pathology Screening Using Deep Learning
## 📌 Project Overview

This project implements a multi-modal deep learning model for automated screening of thoracic pathologies using chest X-ray images combined with structured clinical metadata. The goal is to improve diagnostic performance by leveraging both visual and non-visual patient information, reflecting real-world clinical decision-making.

The model is trained and evaluated on the NIH Chest X-ray dataset, treating each pathology as an independent binary classification task.

## 🎯 Objectives

Develop a robust multi-label classification model for thoracic disease detection

Combine image features and clinical data using a late-fusion architecture

Ensure generalization through patient-wise data splitting

Evaluate performance using accuracy, loss, and ROC-AUC metrics

## 🗂 Dataset

Source: NIH Chest X-ray Dataset

Data Types:

Chest X-ray images

Clinical metadata (e.g., age, gender, view position)

Labels: Multiple thoracic pathologies per image (multi-label)

Due to class imbalance in the dataset, appropriate preprocessing and class-weighted loss were applied.

## 🏗 Model Architecture

The project uses a late fusion multi-modal architecture:

**Image Branch:**

Convolutional Neural Network (CNN) for feature extraction from X-ray images

**Clinical Branch:**

Fully connected neural network for structured clinical features

**Fusion Layer:**

Concatenation of image and clinical embeddings

**Output Layer:**

Sigmoid-activated neurons for multi-label prediction

This design allows the model to learn complementary information from both modalities.

**⚙️ Training Configuration**

Loss Function: Binary Cross-Entropy (class-weighted)

Optimizer: Adam

Epochs: 15

**Evaluation Metrics:**

Binary Accuracy

Training & Validation Loss

ROC Curves and AUC scores

A patient-wise train–validation split was used to prevent data leakage and ensure realistic evaluation.

**📈 Results**

Training and validation accuracy increase steadily and remain closely aligned

Loss curves decrease smoothly without divergence, indicating stable learning

ROC curves demonstrate effective discrimination for selected pathologies

The model shows strong generalization and avoids overfitting, validating the effectiveness of the multi-modal approach.

## 🧪 Evaluation

Given the imbalanced nature of medical datasets, performance is assessed using:

Binary accuracy (interpreted cautiously)

Binary cross-entropy loss

ROC-AUC metrics for selected pathologies

These metrics together provide a reliable measure of clinical relevance.

## 🚀 How to Run

Clone the repository

Install required dependencies:
pip install -r requirements.txt

🔍 Key Contributions

Implementation of a multi-modal late fusion model

Proper handling of class imbalance

Patient-wise data splitting for clinical validity

Comprehensive evaluation using accuracy, loss, and ROC curves

## 🧠 Future Improvements

External dataset validation

Comparison with additional architectures

Threshold optimization per pathology

Deployment using Streamlit or Flask for real-time inference

## 👤 Author

Tess Kamau
Economics & Finance | Data Science | Machine Learning

📍 Kenya
