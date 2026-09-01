"""
Creates a subject-level train/val/test split (70/15/15), stratified by each
subject's dominant label, so no patient's studies appear in more than one split.
"""
import pandas as pd
from sklearn.model_selection import train_test_split

INPUT = "data/processed/final_dataset.csv"

df = pd.read_csv(INPUT)

subject_label = (
    df.groupby("subject_id")["final_label"]
    .agg(lambda x: x.value_counts().idxmax())
    .reset_index()
)

train_subj, temp_subj = train_test_split(
    subject_label, test_size=0.30, stratify=subject_label["final_label"], random_state=42
)
val_subj, test_subj = train_test_split(
    temp_subj, test_size=0.50, stratify=temp_subj["final_label"], random_state=42
)

split_map = {}
split_map.update({s: "train" for s in train_subj["subject_id"]})
split_map.update({s: "val" for s in val_subj["subject_id"]})
split_map.update({s: "test" for s in test_subj["subject_id"]})

df["split"] = df["subject_id"].map(split_map)

print("=== Subjects per split ===")
print(subject_label["subject_id"].groupby(df.drop_duplicates("subject_id").set_index("subject_id")["split"]).count() if False else
      pd.Series({"train": len(train_subj), "val": len(val_subj), "test": len(test_subj)}))

print("\n=== Studies (samples) per split ===")
print(df["split"].value_counts())

print("\n=== Label distribution per split (%) ===")
print(pd.crosstab(df["split"], df["final_label"], normalize="index").round(3) * 100)

df.to_csv(INPUT, index=False)
for name in ["train", "val", "test"]:
    df[df["split"] == name].to_csv(f"data/processed/{name}.csv", index=False)

print("\nSaved: data/processed/final_dataset.csv (with split column), train.csv, val.csv, test.csv")
