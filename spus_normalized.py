import pandas as pd
df = pd.read_csv("SPUS_cleaned.csv")
numeric_cols = ['open', 'high', 'low', 'close', 'volume']
#conversion
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

#applying min-max normalization 
df_normalized = df.copy()

df_normalized[numeric_cols] = (df[numeric_cols] - df[numeric_cols].min()) / (
    df[numeric_cols].max() - df[numeric_cols].min())

#saving the normalized data
df_normalized.to_csv("SPUS_normalized.csv", index=False)

print(df_normalized[numeric_cols].min())
print(df_normalized[numeric_cols].max())
