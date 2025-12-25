import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Set style
sns.set(style="whitegrid")

# Paths
DATA_PATH = r"e:\bioimpedance-carcinogenic-tissue\data\data.csv"
OUTPUT_DIR = r"e:\bioimpedance-carcinogenic-tissue\notebooks\plots"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data():
    """Load and return the dataset."""
    df = pd.read_csv(DATA_PATH)
    print("Data loaded successfully.")
    print(f"Shape: {df.shape}")
    return df

def basic_inspection(df):
    """Print basic info about the dataset."""
    print("\n--- Basic Info ---")
    print(df.info())
    print("\n--- Missing Values ---")
    print(df.isnull().sum())
    print("\n--- Class Distribution ---")
    print(df['Class'].value_counts())

def plot_distributions(df):
    """Generate boxplots for each feature by class."""
    features = [col for col in df.columns if col != 'Class']
    
    for feature in features:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Class', y=feature, data=df, palette='Set3')
        plt.title(f'Distribution of {feature} by Tissue Class')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'boxplot_{feature}.png'))
        plt.close()
    print(f"Boxplots saved to {OUTPUT_DIR}")

def plot_correlation(df):
    """Generate and save correlation heatmap."""
    # Encode class to numeric for correlation if needed, or just drop it
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    
    plt.figure(figsize=(12, 10))
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'correlation_matrix.png'))
    plt.close()
    print(f"Correlation matrix saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    df = load_data()
    basic_inspection(df)
    plot_distributions(df)
    plot_correlation(df)
