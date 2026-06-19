import pandas as pd
import numpy as np

print("=" * 60)
print("DATASET ANALYSIS: Why 99% accuracy?")
print("=" * 60)

# 1. Main CKD dataset
df = pd.read_csv("data/raw/kidney_disease.csv")
print(f"\n1. kidney_disease.csv: {df.shape[0]} rows x {df.shape[1]} cols")
print(f"   Columns: {list(df.columns)}")

# 2. Updated CKD with stages
df2 = pd.read_csv("data/raw/updated_ckd_dataset_with_stages.csv")
print(f"\n2. updated_ckd_dataset_with_stages.csv: {df2.shape[0]} rows x {df2.shape[1]} cols")
print(f"   Columns: {list(df2.columns[:20])}")

# 3. CKD stages
df3 = pd.read_csv("data/raw/ckd_stages_dataset.csv")
print(f"\n3. ckd_stages_dataset.csv: {df3.shape[0]} rows x {df3.shape[1]} cols")
print(f"   Columns: {list(df3.columns)}")

# 4. Check serum creatinine correlation with target
print("\n" + "=" * 60)
print("KEY INSIGHT: Serum Creatinine (sc) vs Target")
print("=" * 60)
df_clean = df.copy()
df_clean = df_clean.replace('?', np.nan)
df_clean['sc_num'] = pd.to_numeric(df_clean['sc'], errors='coerce')
df_clean['target'] = df_clean['classification'].str.strip().map({'ckd': 1, 'notckd': 0})

ckd = df_clean[df_clean['target'] == 1]['sc_num'].dropna()
notckd = df_clean[df_clean['target'] == 0]['sc_num'].dropna()
print(f"   CKD patients    -> sc mean: {ckd.mean():.2f}, median: {ckd.median():.2f}")
print(f"   Non-CKD patients -> sc mean: {notckd.mean():.2f}, median: {notckd.median():.2f}")
print(f"   Separation factor: {ckd.mean() / notckd.mean():.1f}x")

corr = df_clean['sc_num'].corr(df_clean['target'])
print(f"   Correlation with target: {corr:.4f}")

# 5. Check bu
df_clean['bu_num'] = pd.to_numeric(df_clean['bu'], errors='coerce')
ckd_bu = df_clean[df_clean['target'] == 1]['bu_num'].dropna()
notckd_bu = df_clean[df_clean['target'] == 0]['bu_num'].dropna()
print(f"\n   BU - CKD patients    -> mean: {ckd_bu.mean():.2f}")
print(f"   BU - Non-CKD patients -> mean: {notckd_bu.mean():.2f}")
print(f"   BU Separation factor: {ckd_bu.mean() / notckd_bu.mean():.1f}x")

# 6. Dataset size analysis
print(f"\n" + "=" * 60)
print("DATASET SIZE PROBLEM")
print("=" * 60)
print(f"   Total samples: {df.shape[0]}")
print(f"   This is VERY SMALL for ML!")
print(f"   Typical medical ML datasets: 5,000 - 100,000+ samples")
print(f"   With only 400 samples, even weak features can seem perfect")
