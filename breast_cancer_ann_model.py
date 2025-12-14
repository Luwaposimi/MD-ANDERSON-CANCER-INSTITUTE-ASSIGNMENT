
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, classification_report, confusion_matrix)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

# DATA COLLECTION
print("Loading dataset...")


df = pd.read_csv('data.csv')  

# STEP 2: DATA PREPROCESSING 
print("Preprocessing data...")

df.drop(['id', 'Unnamed: 32'], axis=1, inplace=True, errors='ignore')

df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

print(f"Total missing values after cleaning: {df.isnull().sum().sum()}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")

# STEP 3: MODEL BUILDING 
print("Building the Artificial Neural Network...")

model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# STEP 4: TRAINING 
print("Training the model...")

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# STEP 5: EVALUATION 
print("Evaluating on test set...")

y_pred_prob = model.predict(X_test, verbose=0)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n=== Model Performance on Test Set ===")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Benign', 'Malignant']))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# STEP 6: SAVE MODEL & CREATE OUTPUT FOLDER 
output_dir = 'model_output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

model.save(os.path.join(output_dir, 'breast_cancer_ann_model.h5'))
joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))
print(f"\nModel saved to: {output_dir}/breast_cancer_ann_model.h5")
print(f"Scaler saved to: {output_dir}/scaler.pkl")

#  STEP 7: GENERATE AND SAVE PLOTS AS PNG 
print("Generating and saving visualization plots...")

# 1. Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues',
            xticklabels=['Benign', 'Malignant'],
            yticklabels=['Benign', 'Malignant'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
plt.close()

# 2. Training Accuracy
plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'training_accuracy.png'), dpi=300, bbox_inches='tight')
plt.close()

# 3. Training Loss
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'training_loss.png'), dpi=300, bbox_inches='tight')
plt.close()

# 4. NEW: Evaluation Metrics Table as PNG
metrics_data = {
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
    'Value': [f'{accuracy:.4f}', f'{precision:.4f}', f'{recall:.4f}', f'{f1:.4f}']
}
metrics_df = pd.DataFrame(metrics_data)

fig, ax = plt.subplots(figsize=(6, 3))  # Compact size for table
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=metrics_df.values, colLabels=metrics_df.columns, cellLoc='center', loc='center')

# Styling
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 2)  # Make rows taller
table[(0, 0)].set_facecolor('#4CAF50')  # Header green
table[(0, 1)].set_facecolor('#4CAF50')
table[(0, 0)].get_text().set_color('white')
table[(0, 1)].get_text().set_color('white')

plt.title('Model Evaluation Metrics', fontsize=16, pad=30)
plt.savefig(os.path.join(output_dir, 'evaluation_metrics_table.png'), dpi=300, bbox_inches='tight')
plt.close()

print("   - evaluation_metrics_table.png (NEW: Metrics table image)")

print("All PNG plots saved successfully in the 'model_output' folder!")

print("\n=== Project Completed Successfully ===")