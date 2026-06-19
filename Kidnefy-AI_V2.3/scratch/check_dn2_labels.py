import pandas as pd
df = pd.read_csv("data/raw/diabetic_nephropathy2_dataset.csv")
print(pd.crosstab(df['CKD_stage'], df['risk_level']))
print("\nAverage eGFR and UACR by risk level:")
print(df.groupby('risk_level')[['eGFR', 'UACR']].mean())
