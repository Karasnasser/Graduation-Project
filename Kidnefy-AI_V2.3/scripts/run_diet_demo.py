"""
Demo Script: Smart Diet Planner
================================
Tests the AI-powered diet planner with a simulated Stage 4 CKD patient
who has high potassium, hypertension, and diabetes.
"""

import sys
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from src.rag.diet_planner import SmartDietPlanner

def main():
    print("=" * 70)
    print("  🥗 Kidnefy-AI: Smart Diet Planner Demo")
    print("=" * 70)

    # Simulate a high-risk patient
    patient = {
        "age":          58,
        "weight":       85,        # kg
        "egfr":         22.0,      # Stage G4 CKD
        "stage":        "G4",
        "potassium":    5.9,       # HIGH → must restrict bananas, tomatoes, etc.
        "sodium":       148.0,     # Slightly HIGH
        "diabetes":     "yes",
        "hypertension": "yes",
    }

    print("\n[Patient Profile]")
    for k, v in patient.items():
        print(f"  {k:20}: {v}")

    print("\n[Calling Gemini AI to generate 7-day meal plan...]")
    print("-" * 70)

    planner = SmartDietPlanner()

    if not planner.is_active:
        print("❌ ERROR: GEMINI_API_KEY is not set in your .env file!")
        print("   Please add: GEMINI_API_KEY=your_key_here")
        return

    result = planner.generate_diet_plan(patient)

    print("\n" + result)
    print("\n" + "=" * 70)
    print("✅ Demo completed successfully!")

if __name__ == "__main__":
    main()
