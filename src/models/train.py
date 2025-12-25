import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import os
import seaborn as sns
import matplotlib.pyplot as plt

# Paths
DATA_PATH = r"e:\bioimpedance-carcinogenic-tissue\data\data.csv"
MODEL_DIR = r"e:\bioimpedance-carcinogenic-tissue\saved_models"
CONFUSION_MATRIX_PATH = r"e:\bioimpedance-carcinogenic-tissue\notebooks\plots\confusion_matrix.png"

# Ensure directories exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(os.path.dirname(CONFUSION_MATRIX_PATH), exist_ok=True)

def train_model():
    # 1. Load Data
    df = pd.read_csv(DATA_PATH)
    print(f"Data Loaded: {df.shape}")

    # 2. Preprocessing
    X = df.drop('Class', axis=1)
    y = df['Class']

    # Label Encoding
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Save the label encoder
    joblib.dump(le, os.path.join(MODEL_DIR, 'label_encoder.joblib'))
    print(f"Classes: {le.classes_}")

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.joblib'))

    # 3. Model Training
    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    # 4. Evaluation
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH)
    print(f"Confusion Matrix saved to {CONFUSION_MATRIX_PATH}")

    # 5. Save Model
    model_path = os.path.join(MODEL_DIR, 'random_forest_model.joblib')
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_model()
