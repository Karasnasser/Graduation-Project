import urllib.request
import json

payload = {
    "patient": {"name": "test", "age": 50, "sex": "male"},
    "lab_values": {"creatinine": 1.5, "acr": 30.0, "blood_urea": 25.0, "hemoglobin": 13.0}
}
req = urllib.request.Request('http://127.0.0.1:8000/predict', method='POST')
req.add_header('Content-Type', 'application/json')
data = json.dumps(payload).encode('utf-8')

try:
    with urllib.request.urlopen(req, data=data) as response:
        print("Status:", response.status)
        print("Response:", response.read().decode('utf-8'))
except urllib.error.HTTPError as e:
    print("Error HTTP Status:", e.code)
    print("Response:", e.read().decode('utf-8'))
except Exception as e:
    print("Error connecting:", e)
