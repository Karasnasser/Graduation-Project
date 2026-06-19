import pandas as pd
import numpy as np

try:
    df_dn = pd.read_excel("data/raw/Diabetic_Nephropathy_v1.xlsx")
    print("DN 1 shape:", df_dn.shape)
    print("DN 1 columns:", df_dn.columns.tolist()[:10])
except Exception as e:
    print("Could not load DN 1:", e)

try:
    df_dn2 = pd.read_csv("data/raw/diabetic_nephropathy2_dataset.csv")
    print("\nDN 2 shape:", df_dn2.shape)
    print("DN 2 target columns distribution:")
    if 'risk_level' in df_dn2.columns:
        print(df_dn2['risk_level'].value_counts(dropna=False))
    if 'CKD_stage' in df_dn2.columns:
        print(df_dn2['CKD_stage'].value_counts(dropna=False))
except Exception as e:
    print("Could not load DN 2:", e)
