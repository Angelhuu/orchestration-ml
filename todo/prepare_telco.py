from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

raw_path = ROOT / "data" / "telco_churn_raw.csv"
clean_path = ROOT / "data" / "telco_churn_clean.csv"

df = pd.read_csv(raw_path)

df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

df = df.dropna(subset=["TotalCharges", "Churn"])
df = df.drop(columns=["customerID"])

df.to_csv(clean_path, index=False)

print("CSV prepared:", df.shape)
print(df["Churn"].value_counts())
print("Saved to:", clean_path)