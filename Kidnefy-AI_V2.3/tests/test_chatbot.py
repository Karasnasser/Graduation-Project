"""
Quick Test Script for RAG Chatbot
Run this after setting your GEMINI_API_KEY in .env file
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Manually load .env file to avoid python-dotenv dependency issues
env_path = Path(".env")
if env_path.exists():
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ[key.strip()] = val.strip()

# Check API key
api_key = os.environ.get("GEMINI_API_KEY", "")
if not api_key or api_key == "PASTE_YOUR_API_KEY_HERE":
    print("=" * 50)
    print("\n[ERROR] API Key not set!")
    print("Please open the '.env' file in this folder and replace:")
    print("  GEMINI_API_KEY=PASTE_YOUR_API_KEY_HERE")
    print("with your actual Google Gemini API key.")
    print("\nGet a free key here: https://aistudio.google.com/")
    print("=" * 50)
    sys.exit(1)

print("=" * 50)
print("[OK] API Key found! Initializing medical chatbot...")
print("=" * 50)

try:
    from src.rag import GeminiRAG
    
    rag = GeminiRAG()
    
    if rag.model is None:
        print("\n[ERROR] Gemini model failed to initialize.")
        print("Check if your API key is valid.")
        sys.exit(1)
    
    print("\n[SUCCESS] Chatbot ready!")
    print("\nType your medical question (or 'exit' to quit):")
    print("-" * 50)
    
    while True:
        question = input("\nYou: ").strip()
        
        if question.lower() in ['exit', 'quit', 'q']:
            print("\nGoodbye!")
            break
        
        if not question:
            continue
        
        print("\nAI: Thinking...")
        result = rag.ask(question, include_sources=True)
        
        if result.get("error"):
            print(f"\n[Error]: {result['answer']}")
        else:
            print(f"\n{result['answer']}")
            
            if result.get("sources"):
                sources_list = [s["source"] if isinstance(s, dict) else str(s) for s in result["sources"]]
                print(f"\n[Sources: {', '.join(sources_list)}]")

except Exception as e:
    print(f"\n[ERROR] {e}")
