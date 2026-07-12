import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

df = pd.read_csv("Black Friday Dataset.csv")

# ---------------------------------------------------------------
# Customer-level Frequency & Monetary aggregation
# (Recency is not computable: dataset has no date/timestamp field,
#  it is a single Black-Friday-event snapshot, not a transaction log
#  over time. We proceed with FM segmentation and say so explicitly.)
# ---------------------------------------------------------------
cust = df.groupby("User_ID").agg(
    Frequency=("Purchase", "count"),
    Monetary=("Purchase", "sum"),
    Avg_Purchase=("Purchase", "mean"),
    Gender=("Gender", "first"),
    Age=("Age", "first"),
    Occupation=("Occupation", "first"),
    City_Category=("City_Category", "first"),
).reset_index()

print("Customer-level table shape:", cust.shape)
print(cust[["Frequency", "Monetary", "Avg_Purchase"]].describe())

# Score Frequency and Monetary into quartiles (1 = lowest, 4 = highest)
cust["F_Score"] = pd.qcut(cust["Frequency"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
cust["M_Score"] = pd.qcut(cust["Monetary"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
cust["FM_Score"] = cust["F_Score"] + cust["M_Score"]

def segment(row):
    f, m = row["F_Score"], row["M_Score"]
    if f >= 4 and m >= 4:
        return "Champions"
    elif f >= 3 and m >= 3:
        return "Loyal high-value"
    elif f <= 2 and m >= 3:
        return "Big spenders (infrequent)"
    elif f >= 3 and m <= 2:
        return "Frequent low-value"
    elif f <= 2 and m <= 2:
        return "At risk / low engagement"
    else:
        return "Standard"

cust["Segment"] = cust.apply(segment, axis=1)

print("\nSegment counts:")
print(cust["Segment"].value_counts())

print("\nSegment summary (avg frequency, avg monetary, customer count, % of revenue):")
seg_summary = cust.groupby("Segment").agg(
    Customers=("User_ID", "count"),
    Avg_Frequency=("Frequency", "mean"),
    Avg_Monetary=("Monetary", "mean"),
    Total_Monetary=("Monetary", "sum"),
).sort_values("Total_Monetary", ascending=False)
seg_summary["Pct_of_Revenue"] = (seg_summary["Total_Monetary"] / cust["Monetary"].sum() * 100).round(1)
seg_summary["Pct_of_Customers"] = (seg_summary["Customers"] / len(cust) * 100).round(1)
print(seg_summary)

cust.to_csv("customer_fm_segments.csv", index=False)
seg_summary.to_csv("fm_segment_summary.csv")

# ---------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

seg_order = seg_summary.index.tolist()
sns.countplot(data=cust, y="Segment", order=seg_order, ax=axes[0], color="#B54834")
axes[0].set_title("Customer Count by Segment")
axes[0].set_xlabel("Number of Customers")
axes[0].set_ylabel("")

seg_summary["Total_Monetary"].reindex(seg_order).plot(kind="barh", ax=axes[1], color="#2F5D8A")
axes[1].invert_yaxis()
axes[1].set_title("Total Revenue Contribution by Segment")
axes[1].set_xlabel("Total Purchase ($)")
axes[1].set_ylabel("")

plt.tight_layout()
plt.savefig("charts/09_fm_segments.png", dpi=120)
plt.close()

# Scatter: Frequency vs Monetary colored by segment
fig, ax = plt.subplots(figsize=(8, 6))
palette = dict(zip(seg_order, sns.color_palette("Set2", len(seg_order))))
sns.scatterplot(data=cust, x="Frequency", y="Monetary", hue="Segment", hue_order=seg_order,
                 palette=palette, alpha=0.6, ax=ax)
ax.set_title("Customer Segments: Frequency vs Monetary Value")
ax.set_xlabel("Frequency (number of transactions)")
ax.set_ylabel("Monetary (total purchase $)")
plt.tight_layout()
plt.savefig("charts/10_fm_scatter.png", dpi=120)
plt.close()

print("\nSaved: customer_fm_segments.csv, fm_segment_summary.csv, charts/09_fm_segments.png, charts/10_fm_scatter.png")
