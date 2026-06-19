"""
Simple test client for the Kidney Disease Prediction API.
Run this script to test if your API is working correctly.
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint."""
    print("Testing Health Check...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            print("✅ API is online!")
            print(json.dumps(response.json(), indent=2))
            return True
        else:
            print(f"❌ API returned status code {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API. Is it running?")
        return False

def test_prediction():
    """Test prediction endpoint."""
    print("\nTesting Prediction...")
    
    payload = {
        "patient": {
            "name": "Test Patient",
            "age": 65,
            "sex": "male"
        },
        "lab_values": {
            "creatinine": 2.5,
            "acr": 45.0,
            "blood_urea": 60.0,
            "hemoglobin": 11.5
        }
    }
    
    try:
        start_time = time.time()
        response = requests.post(f"{BASE_URL}/predict", json=payload)
        duration = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Prediction successful! ({duration:.2f}s)")
            print(f"  Prediction: {'Positive' if result['prediction'] else 'Negative'}")
            print(f"  Probability: {result['probability']:.2%}")
            print(f"  Stage: {result['gfr_stage']}")
            print(f"  Risk Level: {result['risk_level']}")
        else:
            print(f"❌ Prediction failed: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def test_staging():
    """Test staging endpoint."""
    print("\nTesting Staging...")
    
    payload = {
        "creatinine": 1.2,
        "age": 45,
        "is_female": False,
        "acr": 15.0
    }
    
    try:
        response = requests.post(f"{BASE_URL}/stage", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Staging successful!")
            print(f"  eGFR: {result['egfr']}")
            print(f"  Stage: {result['gfr_stage']}")
            print(f"  Description: {result['description']}")
        else:
            print(f"❌ Staging failed: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print(f"Connecting to {BASE_URL}...")
    
    if test_health():
        test_staging()
        test_prediction()
    else:
        print("\n⚠️ Please run the API server first:")
        print("   python api.py")
