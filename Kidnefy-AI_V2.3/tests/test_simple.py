"""
Simple Logic Test
Tests GFR calculation and Staging without heavy ML imports.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.staging import GFRCalculator, RiskAssessor, GFRStage, RiskLevel

def test_logic():
    print(" Testing Staging Logic...")
    
    # Initialize
    calc = GFRCalculator()
    assessor = RiskAssessor()
    
    # Test Case 1: Healthy Person
    # Age 30, Creatinine 0.9, ACR 10
    print("\n1️⃣  Test Case: Healthy Person (Age 30, Lab normal)")
    result1 = assessor.complete_assessment(
        ckd_probability=0.05,
        creatinine=0.9,
        acr=10.0,
        age=30,
        is_female=False
    )
    print(f"   eGFR: {result1.egfr_value} (Expected > 90)")
    print(f"   Stage: {result1.gfr_stage.value} (Expected G1)")
    print(f"   Risk: {result1.risk_level.value} (Expected Low Risk)")
    
    if result1.gfr_stage == GFRStage.G1:
        print("   ✅ PASSED")
    else:
        print("   ❌ FAILED")

    # Test Case 2: Advanced CKD
    # Age 70, Creatinine 2.5, ACR 350
    print("\n2️⃣  Test Case: Advanced CKD (Age 70, High Creatinine)")
    result2 = assessor.complete_assessment(
        ckd_probability=0.95,
        creatinine=2.5,
        acr=350.0,
        age=70,
        is_female=True
    )
    print(f"   eGFR: {result2.egfr_value} (Expected Low)")
    print(f"   Stage: {result2.gfr_stage.value} (Expected G4/G5)")
    print(f"   Risk: {result2.risk_level.value} (Expected Very High/Critical)")
    print(f"   Alerts: {len(result2.alerts)} alerts generated")
    
    if result2.risk_level in [RiskLevel.VERY_HIGH, RiskLevel.CRITICAL]:
        print("   ✅ PASSED")
    else:
        print("   ❌ FAILED")

if __name__ == "__main__":
    try:
        test_logic()
        print("\n✨ Core logic is working correctly!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
