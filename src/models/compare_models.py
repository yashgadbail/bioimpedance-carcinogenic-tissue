import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
from src.models.nn_model import BioImpedanceNN

# Paths
DATA_PATH = r"e:\bioimpedance-carcinogenic-tissue\data\data.csv"
MODEL_DIR = r"e:\bioimpedance-carcinogenic-tissue\saved_models"
PLOT_DIR = r"e:\bioimpedance-carcinogenic-tissue\notebooks\plots"

def load_artifacts():
    # Load RF artifacts
    rf_model = joblib.load(os.path.join(MODEL_DIR, 'random_forest_model.joblib'))
    rf_scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.joblib'))
    rf_encoder = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.joblib'))

    # Load NN artifacts
    nn_scaler = joblib.load(os.path.join(MODEL_DIR, 'nn_scaler.joblib'))
    nn_encoder = joblib.load(os.path.join(MODEL_DIR, 'nn_label_encoder.joblib'))
    
    # Load NN Model
    # We need to respect the input/output dims used during training.
    # Assuming standard run: 9 features, 6 classes
    input_dim = 9 
    output_dim = len(nn_encoder.classes_)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    nn_model = BioImpedanceNN(input_dim, output_dim).to(device)
    nn_model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'nn_model.pth')))
    nn_model.eval()

    return rf_model, rf_scaler, rf_encoder, nn_model, nn_scaler, nn_encoder

def evaluate_rf(model, scaler, X, y):
    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled)
    acc = accuracy_score(y, preds)
    return acc, preds

def evaluate_nn(model, scaler, X, y):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_scaled = scaler.transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        outputs = model(X_tensor)
        _, preds = torch.max(outputs, 1)
    
    preds = preds.cpu().numpy()
    acc = accuracy_score(y, preds)
    return acc, preds

def compare_models():
    print("Loading Data...")
    df = pd.read_csv(DATA_PATH)
    X = df.drop('Class', axis=1).values
    y_str = df['Class'].values
    
    rf_model, rf_scaler, rf_encoder, nn_model, nn_scaler, nn_encoder = load_artifacts()
    
    # Encode y for comparison
    # Note: RF and NN might have slightly different encodings if not perfectly consistent, 
    # but based on the code they use LabelEncoder on the same string set.
    y_encoded = rf_encoder.transform(y_str)

    print("\n--- Evaluating Random Forest ---")
    rf_acc, rf_preds = evaluate_rf(rf_model, rf_scaler, X, y_encoded)
    print(f"Random Forest Accuracy: {rf_acc:.4f}")

    print("\n--- Evaluating Neural Network ---")
    nn_acc, nn_preds = evaluate_nn(nn_model, nn_scaler, X, y_encoded)
    print(f"Neural Network Accuracy: {nn_acc:.4f}")

    # Visualization
    models = ['Random Forest', 'Neural Network']
    accuracies = [rf_acc, nn_acc]

    plt.figure(figsize=(8, 6))
    sns.barplot(x=models, y=accuracies, palette='viridis')
    plt.title('Model Comparison: Accuracy')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.0)
    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.01, f"{v:.2f}", ha='center')
    
    save_path = os.path.join(PLOT_DIR, 'model_comparison.png')
    plt.savefig(save_path)
    print(f"\nComparison plot saved to {save_path}")

    # Detailed report
    print("\n--- Random Forest Report ---")
    print(model_performance_report(y_encoded, rf_preds, rf_encoder.classes_))

    print("\n--- Neural Network Report ---")
    print(model_performance_report(y_encoded, nn_preds, nn_encoder.classes_))

    # Generate Cole-Cole Plot for NN
    print("\n--- Generating Cole-Cole Classification Plot ---")
    # Convert preds to class names
    y_pred_labels = nn_encoder.inverse_transform(nn_preds)
    # y_encoded was created using rf_encoder in this script, but should be same as nn_encoder if classes are same.
    # To be safe, let's use the original y_str for true labels
    y_true_labels = df['Class'].values
    
    from src.processing.eda_cole_cole import plot_cole_cole_comparison
    plot_cole_cole_comparison(df, y_true_labels, y_pred_labels, nn_encoder.classes_, title_suffix="_NN")

def model_performance_report(y_true, y_pred, class_names):
    # Just return report string
    from sklearn.metrics import classification_report
    return classification_report(y_true, y_pred, target_names=class_names)

if __name__ == "__main__":
    compare_models()
