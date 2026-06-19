"""Quick look at the first dataset only."""
import pandas as pd
df = pd.read_csv('data/raw/ckd_stages_dataset.csv')
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"\nFirst 3 rows:\n{df.head(3).to_string()}")
for c in df.columns:
    if any(k in c.lower() for k in ['stage', 'class', 'target', 'label', 'gfr', 'grade', 'ckd']):
        print(f"\n>>> '{c}' - Unique ({df[c].nunique()}): {sorted(df[c].dropna().unique())}")
        print(f"    Value counts:\n{df[c].value_counts().to_string()}")
print(f"\nMissing:\n{df.isnull().sum().to_string()}")
