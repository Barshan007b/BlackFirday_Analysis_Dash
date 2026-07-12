import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
OUT = "/home/claude/charts"
import os
os.makedirs(OUT, exist_ok=True)

pd.set_option("display.width", 140)

# ---------------------------------------------------------------
# STEP 1: WALKTHROUGH OF THE DATASET
# ---------------------------------------------------------------
print("="*80)
print("STEP 1: DATASET WALKTHROUGH")
print("="*80)

df = pd.read_csv("Black Friday Dataset.csv")

print("\nShape:", df.shape)
print("\nColumns:", list(df.columns))

print("\n--- df.info() ---")
df.info()

print("\n--- Missing values ---")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct})
print(missing_df)

print("\n--- Basic stats on Purchase ---")
print(df["Purchase"].describe())

# ---------------------------------------------------------------
# STEP 2: ANALYZING COLUMNS
# ---------------------------------------------------------------
print("\n" + "="*80)
print("STEP 2: ANALYZING COLUMNS")
print("="*80)

# Drop the two sparse columns as described in the walkthrough
df_clean = df.drop(columns=["Product_Category_2", "Product_Category_3"])
print("\nDropped Product_Category_2 and Product_Category_3 (too sparse).")
print("New shape:", df_clean.shape)

key_cols = ["Gender", "Age", "Marital_Status", "Product_Category_1"]
for col in key_cols:
    print(f"\nUnique values in {col} ({df_clean[col].nunique()} unique):")
    print(sorted(df_clean[col].unique(), key=lambda x: str(x)))

print("\nUser_ID unique count:", df_clean["User_ID"].nunique())
print("Product_ID unique count:", df_clean["Product_ID"].nunique())
print("City_Category unique values:", df_clean["City_Category"].unique())
print("Stay_In_Current_City_Years unique values:", df_clean["Stay_In_Current_City_Years"].unique())
print("Occupation unique count:", df_clean["Occupation"].nunique())

# ---------------------------------------------------------------
# STEP 3: ANALYZING GENDER
# ---------------------------------------------------------------
print("\n" + "="*80)
print("STEP 3: ANALYZING GENDER")
print("="*80)

gender_counts = df_clean["Gender"].value_counts()
gender_pct = (gender_counts / len(df_clean) * 100).round(2)
print("\nTransaction counts by gender:\n", gender_counts)
print("\nPercentage split:\n", gender_pct)

gender_purchase = df_clean.groupby("Gender")["Purchase"].agg(["sum", "mean", "count"])
gender_purchase = gender_purchase.rename(columns={"sum": "total_purchase", "mean": "avg_purchase", "count": "num_transactions"})
print("\nPurchase amount by gender:\n", gender_purchase)

# Unique customers by gender (dedupe on User_ID)
unique_customers_gender = df_clean.drop_duplicates("User_ID")["Gender"].value_counts()
print("\nUnique customers by gender:\n", unique_customers_gender)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
gender_counts.plot(kind="bar", ax=axes[0], color=["#4C72B0", "#DD8452"])
axes[0].set_title("Transaction Count by Gender")
axes[0].set_xlabel("Gender")
axes[0].set_ylabel("Number of Transactions")

gender_purchase["total_purchase"].plot(kind="bar", ax=axes[1], color=["#4C72B0", "#DD8452"])
axes[1].set_title("Total Purchase Amount by Gender")
axes[1].set_xlabel("Gender")
axes[1].set_ylabel("Total Purchase")
plt.tight_layout()
plt.savefig(f"{OUT}/03_gender_analysis.png", dpi=120)
plt.close()

# ---------------------------------------------------------------
# STEP 4: ANALYZING AGE & MARITAL STATUS
# ---------------------------------------------------------------
print("\n" + "="*80)
print("STEP 4: ANALYZING AGE & MARITAL STATUS")
print("="*80)

age_order = ["0-17", "18-25", "26-35", "36-45", "46-50", "51-55", "55+"]
df_clean["Age"] = pd.Categorical(df_clean["Age"], categories=age_order, ordered=True)

age_purchase = df_clean.groupby("Age")["Purchase"].agg(["sum", "mean", "count"]).rename(
    columns={"sum": "total_purchase", "mean": "avg_purchase", "count": "num_transactions"}
)
print("\nPurchase by age group:\n", age_purchase)

marital_purchase = df_clean.groupby("Marital_Status")["Purchase"].agg(["sum", "mean", "count"]).rename(
    columns={"sum": "total_purchase", "mean": "avg_purchase", "count": "num_transactions"}
)
marital_purchase.index = marital_purchase.index.map({0: "Unmarried", 1: "Married"})
print("\nPurchase by marital status:\n", marital_purchase)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
age_purchase["total_purchase"].plot(kind="bar", ax=axes[0], color="#55A868")
axes[0].set_title("Total Purchase by Age Group")
axes[0].set_xlabel("Age Group")
axes[0].set_ylabel("Total Purchase")

marital_purchase["total_purchase"].plot(kind="bar", ax=axes[1], color=["#C44E52", "#8172B2"])
axes[1].set_title("Total Purchase by Marital Status")
axes[1].set_xlabel("Marital Status")
axes[1].set_ylabel("Total Purchase")
plt.tight_layout()
plt.savefig(f"{OUT}/04_age_marital_analysis.png", dpi=120)
plt.close()

# ---------------------------------------------------------------
# STEP 5: MULTI COLUMN ANALYSIS (Age + Marital Status + Gender)
# ---------------------------------------------------------------
print("\n" + "="*80)
print("STEP 5: MULTI COLUMN ANALYSIS")
print("="*80)

multi = df_clean.groupby(["Age", "Gender", "Marital_Status"], observed=True)["Purchase"].agg(
    ["sum", "mean", "count"]
).rename(columns={"sum": "total_purchase", "mean": "avg_purchase", "count": "num_transactions"})
print("\nAge x Gender x Marital Status purchase breakdown:\n", multi)

# Pie chart: distribution of transactions across age groups
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
age_counts = df_clean["Age"].value_counts().sort_index()
axes[0].pie(age_counts, labels=age_counts.index, autopct="%1.1f%%", startangle=90,
            colors=sns.color_palette("pastel", len(age_counts)))
axes[0].set_title("Transaction Share by Age Group")

# Bar plot: avg purchase by age, split by gender
sns.barplot(data=df_clean, x="Age", y="Purchase", hue="Gender", estimator="mean",
            errorbar=None, order=age_order, ax=axes[1])
axes[1].set_title("Average Purchase by Age Group and Gender")
axes[1].set_ylabel("Average Purchase")
plt.tight_layout()
plt.savefig(f"{OUT}/05_multicolumn_analysis.png", dpi=120)
plt.close()

# Extra: age x marital status combo bar
fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(data=df_clean, x="Age", y="Purchase", hue="Marital_Status", estimator="mean",
            errorbar=None, order=age_order, ax=ax)
ax.set_title("Average Purchase by Age Group and Marital Status")
ax.legend(title="Marital Status", labels=["Unmarried", "Married"])
plt.tight_layout()
plt.savefig(f"{OUT}/05b_age_marital_combo.png", dpi=120)
plt.close()

# ---------------------------------------------------------------
# STEP 6: OCCUPATION AND PRODUCTS ANALYSIS
# ---------------------------------------------------------------
print("\n" + "="*80)
print("STEP 6: OCCUPATION AND PRODUCTS ANALYSIS")
print("="*80)

occ_purchase = df_clean.groupby("Occupation")["Purchase"].agg(["sum", "mean", "count"]).rename(
    columns={"sum": "total_purchase", "mean": "avg_purchase", "count": "num_transactions"}
).sort_values("total_purchase", ascending=False)
print("\nPurchase by Occupation (sorted by total, top 10):\n", occ_purchase.head(10))

top_products = df_clean["Product_ID"].value_counts().head(10)
print("\nTop 10 most purchased Product_IDs:\n", top_products)

top_categories = df_clean["Product_Category_1"].value_counts().head(10)
print("\nTop 10 Product_Category_1 by transaction count:\n", top_categories)

# Which product category is most popular per occupation (top category per occupation)
occ_cat = df_clean.groupby(["Occupation", "Product_Category_1"], observed=True).size().reset_index(name="count")
top_cat_per_occ = occ_cat.loc[occ_cat.groupby("Occupation")["count"].idxmax()].sort_values("Occupation")
print("\nMost popular Product_Category_1 per Occupation:\n", top_cat_per_occ.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
occ_purchase["total_purchase"].sort_values(ascending=False).plot(kind="bar", ax=axes[0], color="#4C72B0")
axes[0].set_title("Total Purchase Amount by Occupation")
axes[0].set_xlabel("Occupation Code")
axes[0].set_ylabel("Total Purchase")

top_categories.plot(kind="bar", ax=axes[1], color="#DD8452")
axes[1].set_title("Top 10 Product Categories by Transaction Count")
axes[1].set_xlabel("Product Category")
axes[1].set_ylabel("Number of Transactions")
plt.tight_layout()
plt.savefig(f"{OUT}/06_occupation_products.png", dpi=120)
plt.close()

# ---------------------------------------------------------------
# STEP 7: COMBINING GENDER & MARITAL STATUS
# ---------------------------------------------------------------
print("\n" + "="*80)
print("STEP 7: COMBINING GENDER & MARITAL STATUS")
print("="*80)

gm_purchase = df_clean.groupby(["Gender", "Marital_Status"])["Purchase"].agg(["sum", "mean", "count"]).rename(
    columns={"sum": "total_purchase", "mean": "avg_purchase", "count": "num_transactions"}
)
print("\nGender x Marital Status purchase breakdown:\n", gm_purchase)

gm_customers = df_clean.drop_duplicates("User_ID").groupby(["Gender", "Marital_Status"]).size()
print("\nUnique customer count by Gender x Marital Status:\n", gm_customers)

fig, ax = plt.subplots(figsize=(8, 6))
sns.countplot(data=df_clean, x="Gender", hue="Marital_Status", ax=ax)
ax.set_title("Transaction Count by Gender and Marital Status")
ax.legend(title="Marital Status", labels=["Unmarried", "Married"])
plt.tight_layout()
plt.savefig(f"{OUT}/07_gender_marital_countplot.png", dpi=120)
plt.close()

print("\nAll charts saved to", OUT)
print("\nDONE.")