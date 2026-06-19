import json
import urllib.request

url = "http://localhost:8000/predict/whatif"
data = {
    "baseline": {"age": 60, "sex": "male", "sc": 2.5, "bp": 160, "al": 2, "dm": "no"},
    "modified": {"age": 60, "sex": "male", "sc": 1.8, "bp": 125, "al": 0, "dm": "no"}
}

req = urllib.request.Request(url, data=json.dumps(data).encode('utf-8'), headers={'Content-Type': 'application/json'})
try:
    with urllib.request.urlopen(req) as response:
        print("Status", response.status)
        print(response.read().decode())
except urllib.error.HTTPError as e:
    print("HTTPError", e.code)
    print(e.read().decode())
except Exception as e:
    print("Error:", e)
