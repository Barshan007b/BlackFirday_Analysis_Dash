import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib

sns.set_theme(style="whitegrid")

df = pd.read_csv("Black Friday Dataset.csv")
df["Product_Category_2"] = df["Product_Category_2"].fillna(0)
df["Product_Category_3"] = df["Product_Category_3"].fillna(0)

FEATURES = ["Gender", "Age", "Occupation", "City_Category",
            "Stay_In_Current_City_Years", "Marital_Status",
            "Product_Category_1", "Product_Category_2", "Product_Category_3"]
TARGET = "Purchase"

X = df[FEATURES]
y = df[TARGET]

CATEGORICAL = ["Gender", "Age", "City_Category", "Stay_In_Current_City_Years"]
NUMERIC = ["Occupation", "Marital_Status", "Product_Category_1", "Product_Category_2", "Product_Category_3"]

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ],
    remainder="passthrough",
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

results = {}

# --- Linear Regression baseline ---
lin_pipe = Pipeline([("prep", preprocess), ("model", LinearRegression())])
lin_pipe.fit(X_train, y_train)
pred_lin = lin_pipe.predict(X_test)
results["Linear Regression"] = {
    "R2": r2_score(y_test, pred_lin),
    "MAE": mean_absolute_error(y_test, pred_lin),
    "RMSE": np.sqrt(mean_squared_error(y_test, pred_lin)),
}

# --- Random Forest ---
rf_pipe = Pipeline([("prep", preprocess), ("model", RandomForestRegressor(
    n_estimators=60, max_depth=12, min_samples_leaf=10, random_state=42, n_jobs=-1))])
rf_pipe.fit(X_train, y_train)
pred_rf = rf_pipe.predict(X_test)
results["Random Forest"] = {
    "R2": r2_score(y_test, pred_rf),
    "MAE": mean_absolute_error(y_test, pred_rf),
    "RMSE": np.sqrt(mean_squared_error(y_test, pred_rf)),
}

print("Model comparison:")
for name, m in results.items():
    print(f"  {name}: R2={m['R2']:.3f}  MAE={m['MAE']:.1f}  RMSE={m['RMSE']:.1f}")

pd.DataFrame(results).T.round(3).to_csv("model_comparison.csv")

# --- Feature importance from Random Forest ---
ohe = rf_pipe.named_steps["prep"].named_transformers_["cat"]
ohe_names = ohe.get_feature_names_out(CATEGORICAL)
all_names = list(ohe_names) + NUMERIC
importances = rf_pipe.named_steps["model"].feature_importances_
feat_imp = pd.DataFrame({"feature": all_names, "importance": importances}).sort_values("importance", ascending=False)
print("\nTop 15 feature importances (Random Forest):")
print(feat_imp.head(15).to_string(index=False))
feat_imp.to_csv("feature_importance.csv", index=False)

fig, ax = plt.subplots(figsize=(9, 7))
top_feat = feat_imp.head(15)
sns.barplot(data=top_feat, y="feature", x="importance", color="#2F5D8A", ax=ax)
ax.set_title("Top 15 Feature Importances (Random Forest)")
ax.set_xlabel("Importance")
ax.set_ylabel("")
plt.tight_layout()
plt.savefig("charts/12_feature_importance.png", dpi=120)
plt.close()

# --- Predicted vs actual scatter (Random Forest, sample for readability) ---
sample_idx = np.random.RandomState(42).choice(len(y_test), size=3000, replace=False)
fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(y_test.values[sample_idx], pred_rf[sample_idx], alpha=0.25, s=12, color="#B54834")
lims = [y_test.min(), y_test.max()]
ax.plot(lims, lims, color="#1E1B16", linewidth=1, linestyle="--")
ax.set_xlabel("Actual Purchase ($)")
ax.set_ylabel("Predicted Purchase ($)")
ax.set_title(f"Random Forest: Predicted vs Actual (R\u00b2 = {results['Random Forest']['R2']:.3f})")
plt.tight_layout()
plt.savefig("charts/13_pred_vs_actual.png", dpi=120)
plt.close()

# Save the trained model + a small metadata file describing the input schema
joblib.dump(rf_pipe, "purchase_predictor.joblib")
print("\nSaved: model_comparison.csv, feature_importance.csv, purchase_predictor.joblib")
print("Saved charts: 12_feature_importance.png, 13_pred_vs_actual.png")
