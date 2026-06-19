import os
import google.generativeai as genai
from pathlib import Path

# Load env manually
env_path = Path(".env")
if env_path.exists():
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ[key.strip()] = val.strip()

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key or api_key == "PASTE_YOUR_API_KEY_HERE":
    print("No valid API key found.")
    exit(1)

genai.configure(api_key=api_key)

print("Available models supporting generateContent:")
print("-" * 50)
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(f"- {m.name}")
