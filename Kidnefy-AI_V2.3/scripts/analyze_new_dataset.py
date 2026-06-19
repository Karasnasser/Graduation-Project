import pandas as pd
import numpy as np

df = pd.read_csv('data/raw/diabetic_nephropathy2_dataset.csv')

print("=" * 60)
print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
print("=" * 60)

print("\nColumns:")
for i, col in enumerate(df.columns, 1):
    print(f"  {i}. {col} ({df[col].dtype})")

print("\nFirst 5 rows:")
print(df.head(5).to_string())

print("\nDescribe (numeric):")
print(df.describe().to_string())

print("\nNull counts:")
for col in df.columns:
    nulls = df[col].isnull().sum()
    if nulls > 0:
        print(f"  {col}: {nulls} nulls ({nulls/len(df)*100:.1f}%)")

print("\nUnique values per column:")
for col in df.columns:
    n = df[col].nunique()
    if n <= 20:
        print(f"  {col}: {n} unique -> {list(df[col].unique()[:10])}")
    else:
        print(f"  {col}: {n} unique (sample: {list(df[col].dropna().unique()[:5])})")

# Check for target column
print("\nPotential target columns:")
for col in df.columns:
    if df[col].nunique() <= 10:
        print(f"  {col}: {dict(df[col].value_counts())}")
