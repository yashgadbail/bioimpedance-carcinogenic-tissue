# Bioimpedance Analysis & Tissue Classification

A Machine Learning powered web application for analyzing bioimpedance measurements to detect carcinogenic tissue abnormalities. This project combines conventional Machine Learning (Random Forest) and Deep Learning (MLP) approaches with a modern Web Interface for real-time analysis.

![Bioimpedance Analysis](https://images.unsplash.com/photo-1576086213369-97a306d36557?auto=format&fit=crop&w=1000&q=80) 

## 🌟 Features

*   **Machine Learning Classification**: Accurately classifies 6 types of breast tissue (Carcinoma, Fibro-adenoma, Mastopathy, Glandular, Connective, Adipose).
*   **Web Interface**: Clean, dark-themed UI for easy parameter input and instant prediction.
*   **Live Simulation**: Visualizes streaming bioimpedance data ($I_0$, $PA_{500}$) in real-time using simulated sensors.
*   **Comprehensive Report**: Embedded PDF viewer to access the detailed project report and LaTeX source code.

## 🛠️ Technology Stack

*   **Frontend**: HTML5, CSS3, JavaScript, Chart.js
*   **Backend**: Python, Flask
*   **ML/AI**: Scikit-Learn (Random Forest), PyTorch (Neural Network), Pandas, Seaborn
*   **Documentation**: LaTeX

## 📂 Project Structure

```
├── data/               # Dataset (CSV)
├── notebooks/          # Exploratory Data Analysis (EDA)
├── report/             # LaTeX Project Report & PDF
├── saved_models/       # Trained models & scalers
├── src/                # Source Code
│   ├── models/         # Training scripts (RF & NN)
│   ├── processing/     # Data processing utils
│   └── web/            # Flask Web Application
│       ├── static/     # CSS styles
│       ├── templates/  # HTML templates
│       └── app.py      # Application entry point
├── requirements.txt    # Dependencies
└── README.md           # This file
```

## 🚀 Getting Started

### Prerequisites

*   Python 3.8+
*   Pip

### Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/yourusername/bioimpedance-analysis.git
    cd bioimpedance-analysis
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application

1.  **Start the Flask Server**
    ```bash
    python src/web/app.py
    ```

2.  **Access the Web App**
    Open your browser and navigate to:
    `http://127.0.0.1:5000`

## 📊 Model Performance

*   **Random Forest**: ~94% Accuracy (Selected for deployment due to robustness on small data)
*   **Neural Network**: ~76% Accuracy (Used for comparative analysis)

## 📄 Project Report

The project includes a detailed LaTeX report covering the theoretical background (Cole-Cole model), methodology, and error analysis. You can view it directly in the web app under the "View & Download Report" section.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License.
