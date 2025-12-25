from flask import Flask, request, render_template, jsonify, send_file, Response, stream_with_context
import joblib
import numpy as np
import os
import time
import json
import random

app = Flask(__name__)

# Load Model and Artifacts
MODEL_DIR = r"e:\bioimpedance-carcinogenic-tissue\saved_models"
MODEL_PATH = os.path.join(MODEL_DIR, "random_forest_model.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.joblib")

model = None
scaler = None
encoder = None

def load_artifacts():
    global model, scaler, encoder
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        encoder = joblib.load(ENCODER_PATH)
        print("Artifacts loaded successfully.")
    else:
        print("Model artifacts not found. Please train the model first.")

# Manually load since before_first_request is deprecated in newer Flask
load_artifacts()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/simulation')
def simulation():
    return render_template('simulation.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        # Get data from form keys that match our feature names
        feature_names = ['I0', 'PA500', 'HFS', 'DA', 'Area', 'A/DA', 'Max.IP', 'DR', 'P']
        
        # Extract features in correct order
        features = []
        for name in feature_names:
            # Handle potential frontend key mismatches (e.g. ADA -> A/DA, MaxIP -> Max.IP)
            val = request.form.get(name)
            if val is None:
                # Fallback mapping
                if name == 'Max.IP': val = request.form.get('MaxIP')
                if name == 'A/DA': val = request.form.get('ADA')
            
            if val is None:
                return jsonify({'error': f'Missing feature: {name}'}), 400
            features.append(float(val))

        # Reshape for single sample
        features_array = np.array(features).reshape(1, -1)
        
        # Scale features
        features_scaled = scaler.transform(features_array)
        
        # Predict
        # Get probability/confidence
        probas = model.predict_proba(features_scaled)[0]
        prediction_index = np.argmax(probas)
        confidence = probas[prediction_index]
        
        # Decode class label
        prediction_class = encoder.inverse_transform([prediction_index])[0]
        
        return jsonify({
            'class': prediction_class,
            'confidence': f"{confidence*100:.1f}%"
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

def generate_random_data():
    """Generates a random bioimpedance sample based on approximate class means."""
    classes = ['car', 'fad', 'mas', 'gla', 'con', 'adi']
    # Base values for 'car' (approx)
    base = { 
        'I0': 400, 'PA500': 0.2, 'HFS': 0.1, 'DA': 150, 
        'Area': 5000, 'A/DA': 30, 'Max.IP': 70, 'DR': 150, 'P': 450 
    }
    
    while True:
        # drift values slightly
        for k in base:
            change = random.uniform(-0.1, 0.1) * base[k]
            base[k] += change
            # Keep positive
            if base[k] < 0: base[k] = 10

        # Create feature vector
        features = [
            base['I0'], base['PA500'], base['HFS'], base['DA'], 
            base['Area'], base['A/DA'], base['Max.IP'], base['DR'], base['P']
        ]
        
        # Predict on this simulated data
        if model and scaler and encoder:
            feat_arr = np.array(features).reshape(1, -1)
            feat_scaled = scaler.transform(feat_arr)
            idx = model.predict(feat_scaled)[0]
            pred_class = encoder.inverse_transform([idx])[0]
        else:
            pred_class = "loading..."

        json_data = json.dumps({
            'I0': base['I0'],
            'PA500': base['PA500'],
            'predicted_class': pred_class
        })
        yield f"data: {json_data}\n\n"
        time.sleep(1)

@app.route('/stream')
def stream():
    return Response(stream_with_context(generate_random_data()), mimetype='text/event-stream')

@app.route('/report')
def report_page():
    try:
        report_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'report', 'main.tex')
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return render_template('report.html', latex_content=content)
    except Exception as e:
        return f"Error reading report file: {str(e)}", 500

@app.route('/download_report')
def download_report():
    try:
        report_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'report', 'main.pdf')
        if os.path.exists(report_path):
            return send_file(report_path, as_attachment=True)
        else:
            return "Report PDF not found. Please compile the LaTeX report and place 'main.pdf' in the 'report' directory.", 404
    except Exception as e:
        return str(e), 500

@app.route('/view_pdf')
def view_pdf():
    try:
        report_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'report', 'main.pdf')
        if os.path.exists(report_path):
            return send_file(report_path, mimetype='application/pdf')
        else:
            return "Report PDF not found.", 404
    except Exception as e:
        return str(e), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
