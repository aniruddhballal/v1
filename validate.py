# validate.py
import json
import pandas as pd

df = pd.read_csv("data/finance/creditcard.csv")
with open("output/creditcard_anomalies.json") as f:
    explanations = json.load(f)

flagged_rows = [e["row"] for e in explanations]
flagged = df.iloc[flagged_rows]

total_flagged = len(flagged)
fraud_caught = flagged["Class"].sum()
fraud_in_dataset = df["Class"].sum()
random_baseline = total_flagged * (fraud_in_dataset / len(df))

print(f"Flagged:              {total_flagged} rows")
print(f"Fraud in flagged:     {int(fraud_caught)}")
print(f"Random baseline:      {random_baseline:.1f} (expected if flagging randomly)")
print(f"Lift over random:     {fraud_caught / random_baseline:.1f}×")
print(f"Fraud recall:         {fraud_caught / fraud_in_dataset * 100:.1f}% of all fraud cases caught")