# Bioimpedance-Based Tissue Classification System

**A Machine Learning & Physics-Informed Approach to Non-Invasive Cancer Detection**

![Bioimpedance Analysis](https://images.unsplash.com/photo-1576086213369-97a306d36557?auto=format&fit=crop&w=1000&q=80)

## 📖 Overview
This project develops a robust system for the automated classification of breast tissue types (including **Carcinoma**, **Fibro-adenoma**, and **Adipose**) using **Bioelectrical Impedance Analysis (BIA)**. By analyzing the electrical properties of tissue (Resistance, Reactance, Phase Angle), we provide a rapid, radiation-free alternative to traditional biopsy and mammography.

The system combines:
1.  **Machine Learning**: Random Forest and Neural Networks for high-accuracy classification.
2.  **Physics-Informed Analysis**: **Cole-Cole plots** to validate model decisions against biophysical reality.
3.  **Web Interface**: A Flask-based dashboard for real-time analysis and "Live Simulation".
4.  **Comprehensive Documentation**: A detailed **32-page Project Report** (LaTeX).

---

## 🌟 Key Features

*   **Multi-Class Classification**: Distinguishes between 6 tissue types:
    *   `car`: Carcinoma (Malignant)
    *   `fad`: Fibro-adenoma (Benign)
    *   `mas`: Mastopathy (Benign)
    *   `gla`: Glandular (Healthy)
    *   `con`: Connective (Healthy)
    *   `adi`: Adipose (Healthy)
*   **Physics Visualization**: Automatically generates **Cole-Cole Arcs** to visualize the impedance spectrum and decision boundaries.
*   **Live Simulation Mode**: Simulates streaming data input to demonstrate real-time diagnostic capabilities.
*   **Interactive Web App**: Modern, dark-themed UI built with Flask and HTML5.

---

## 📊 Results & Performance

We evaluated our models on a dataset of 106 samples. The **Random Forest** classifier was selected for deployment due to its superior performance on tabular bioimpedance data.

| Metric | Random Forest | Neural Network |
| :--- | :--- | :--- |
| **Accuracy** | **94%** | 76% |
| **Precision** | **0.95** | 0.76 |
| **Recall (Sensitivity)** | **0.94** | 0.76 |

### 🔬 Physics-Informed Validation (Cole-Cole Plots)
Beyond raw accuracy, we validated the **Neural Network's** understanding of physics by reconstructing **Cole-Cole plots**.
*   **Adipose Tissue**: Correctly identified as having **High Resistance** (large arcs to the right) due to low water content.
*   **Carcinoma**: Correctly identified as having **Low Resistance** (small arcs to the left) due to high vascularization and cell breakdown.
*   *Interpretation*: This confirms the models are learning true biophysical properties, not just statistical noise.

---

## 🚀 Installation & Usage

### Prerequisites
*   Python 3.8 or higher
*   Git

### 1. Clone the Repository
```bash
git clone https://github.com/yashgadbail/bioimpedance-carcinogenic-tissue.git
cd bioimpedance-carcinogenic-tissue
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Application
Start the Flask web server:
```bash
python src/web/app.py
```

### 4. Access the Dashboard
Open your browser and navigate to:
```
http://127.0.0.1:5000
```
*   **Home**: Project overview and "Live Analysis" button.
*   **Analysis**: Input parameters manually or use "Simulate Data" to see the model in action.
*   **Report**: View the generated Project Report and download the PDF.

---

## 📂 Project Structure

```bash
e:\bioimpedance-carcinogenic-tissue\
├── data/                   # Original dataset (CSV)
├── final_project_report/   # 📄 NEW: Detailed 32-page LaTeX Project Report
│   ├── chapters/           # Individual TeX chapters
│   ├── images/             # Generated plots (Cole-Cole, Confusion Matrix)
│   └── main.tex            # Main LaTeX driver
├── notebooks/              # Jupyter Notebooks for EDA & Plotting
├── report/                 # Original/Draft Report
├── saved_models/           # 🤖 Trained Models (.joblib)
├── src/                    # Source Code
│   ├── models/             # Training scripts (RF, NN, Comparison)
│   ├── processing/         # Feature extraction & Cole-Cole logic
│   └── web/                # Flask Application (Templates, Static)
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## 📚 Detailed Report
A comprehensive **32-page Project Report** is available in the `final_project_report/` directory. It includes:
*   Theoretical background on Bioimpedance (dispersion regions, Cole-Cole equation).
*   Detailed methodology (PINNs proxy, Random Forest split criteria).
*   Extensive error analysis and future scope.

## 🤝 Contributors
*   **Yash Gadbail** - *Department of Scientific Computing, Modeling & Simulation*

## 📝 License
This project is licensed under the MIT License.
