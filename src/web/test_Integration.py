import pytest
import sys
import os
import io

# Add project root to path to allow importing src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.web.app import app

def test_home_page():
    """Test the home page loads."""
    client = app.test_client()
    response = client.get('/')
    assert response.status_code == 200
    assert b"Bioimpedance Tissue Analysis" in response.data

def test_prediction():
    """Test the prediction endpoint."""
    client = app.test_client()
    
    # Sample data (based on 'car' class in data.csv)
    data = {
        'I0': 524.79,
        'PA500': 0.187,
        'HFS': 0.032,
        'DA': 228.8,
        'Area': 6843.59,
        'ADA': 29.91,
        'MaxIP': 60.2,
        'DR': 220.73,
        'P': 556.82
    }
    
    response = client.post('/predict', data=data)
    
    assert response.status_code == 200
    json_data = response.get_json()
    assert 'class' in json_data
    assert 'confidence' in json_data
    assert isinstance(json_data['class'], str)

def test_missing_data():
    """Test prediction with missing data."""
    client = app.test_client()
    response = client.post('/predict', data={})
    assert response.status_code == 400

if __name__ == "__main__":
    # If run directly, just run a quick manual check
    print("Running manual test...")
    try:
        test_home_page()
        print("Home page test passed!")
        test_prediction()
        print("Prediction test passed!")
        test_missing_data()
        print("Error handling mechanism passed!")
        print("All tests passed.")
    except AssertionError as e:
        print(f"Test failed: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
