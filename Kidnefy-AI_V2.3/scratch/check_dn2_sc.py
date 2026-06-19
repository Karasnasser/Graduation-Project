import pandas as pd
df = pd.read_csv("data/raw/diabetic_nephropathy2_dataset.csv")
print("serum_creatinine (sc) stats in dn2:")
print(df['serum_creatinine'].describe())
