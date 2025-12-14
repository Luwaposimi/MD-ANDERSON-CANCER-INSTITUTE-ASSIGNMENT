Project Overview

This project implements an Artificial Neural Network (ANN) using TensorFlow and Keras to classify breast cancer tumors as Benign or Malignant. The model is trained on a structured clinical dataset and demonstrates how deep learning can support medical decision-making through accurate and automated cancer diagnosis.

Dataset
File: data.csv
Target Variable: diagnosis
0 → Benign
1 → Malignant

Features: Numerical measurements of cell nuclei characteristics
Preprocessing Steps:
Removal of irrelevant columns (id, Unnamed: 32)
Label encoding of diagnosis
Feature scaling using StandardScaler

Project Pipeline
Data Collection
Loads the breast cancer dataset from a local CSV file.
Data Preprocessing
Cleans and encodes the dataset
Standardizes numerical features

Splits data into training and test sets with stratification
Model Building
Fully connected ANN with multiple hidden layers
ReLU activation functions and Dropout for regularization
Sigmoid output layer for binary classification

Model Training
Optimizer: Adam
Loss Function: Binary Cross-Entropy
Trained for 100 epochs with validation split

Model Evaluation
Accuracy, Precision, Recall, and F1-Score
Classification report and confusion matrix
Model Persistence
Saves trained model and scaler for reuse

Visualization
Training accuracy and loss curves
Confusion matrix heatmap
Evaluation metrics summary table

Model Architecture

Input Layer: 30 features

Hidden Layers:

Dense (64 neurons) + ReLU + Dropout (0.2)

Dense (32 neurons) + ReLU + Dropout (0.2)

Dense (16 neurons) + ReLU + Dropout (0.2)

Output Layer: Dense (1 neuron) + Sigmoid

Model Evaluation Metrics

The model is evaluated on a held-out test dataset using:

Accuracy

Precision

Recall

F1-Score

These metrics are particularly important in healthcare applications where minimizing false negatives is critical.

Output Files

All outputs are saved in the model_output/ directory:

model_output/
├── breast_cancer_ann_model.h5
├── scaler.pkl
├── confusion_matrix.png
├── training_accuracy.png
├── training_loss.png
├── evaluation_metrics_table.png

Installation Requirements

Install the required dependencies using:

pip install pandas numpy scikit-learn tensorflow matplotlib seaborn joblib

How to Run the Project

Place the dataset (data.csv) in the project root directory

Install all dependencies

Run the Python script:

python breast_cancer_ann.py
