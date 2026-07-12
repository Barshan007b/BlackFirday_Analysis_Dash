import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib

st.set_page_config(page_title="Black Friday Sales Explorer", layout="wide", page_icon="\U0001F6CD\uFE0F")

# ---------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Black Friday Dataset.csv")
    age_order = ["0-17", "18-25", "26-35", "36-45", "46-50", "51-55", "55+"]
    df["Age"] = pd.Categorical(df["Age"], categories=age_order, ordered=True)
    return df

@st.cache_data
def load_segments():
    return pd.read_csv("customer_fm_segments.csv")

@st.cache_data
def load_rules():
    return pd.read_csv("market_basket_rules.csv")

@st.cache_resource
def load_model():
    return joblib.load("purchase_predictor.joblib")

df = load_data()
segments = load_segments()
rules = load_rules()

st.title("Black Friday sales explorer")
st.caption("537,577 transactions \u00b7 5,891 customers \u00b7 interactive exploration, segmentation, market basket analysis, and purchase prediction")

page = st.sidebar.radio(
    "Section",
    ["Overview & filters", "Customer segmentation (FM)", "Market basket analysis",
     "Purchase prediction", "Business recommendations"],
)

# =================================================================
# PAGE 1: OVERVIEW & FILTERS
# =================================================================
if page == "Overview & filters":
    st.sidebar.markdown("---")
    st.sidebar.subheader("Filters")
    gender_f = st.sidebar.multiselect("Gender", sorted(df["Gender"].unique()))
    age_f = st.sidebar.multiselect("Age group", list(df["Age"].cat.categories))
    marital_f = st.sidebar.multiselect("Marital status", [0, 1],
                                        format_func=lambda x: "Married" if x == 1 else "Unmarried")
    city_f = st.sidebar.multiselect("City category", sorted(df["City_Category"].unique()))
    occ_f = st.sidebar.multiselect("Occupation", sorted(df["Occupation"].unique()))

    filtered = df.copy()
    if gender_f: filtered = filtered[filtered["Gender"].isin(gender_f)]
    if age_f: filtered = filtered[filtered["Age"].isin(age_f)]
    if marital_f: filtered = filtered[filtered["Marital_Status"].isin(marital_f)]
    if city_f: filtered = filtered[filtered["City_Category"].isin(city_f)]
    if occ_f: filtered = filtered[filtered["Occupation"].isin(occ_f)]

    st.markdown(f"**{len(filtered):,}** of {len(df):,} transactions match the current filters")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total purchase", f"${filtered['Purchase'].sum():,.0f}")
    c2.metric("Avg. purchase", f"${filtered['Purchase'].mean():,.0f}" if len(filtered) else "--")
    c3.metric("Transactions", f"{len(filtered):,}")
    c4.metric("Unique customers", f"{filtered['User_ID'].nunique():,}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        age_agg = filtered.groupby("Age", observed=True)["Purchase"].sum().reset_index()
        fig = px.bar(age_agg, x="Age", y="Purchase", title="Total purchase by age group",
                     color_discrete_sequence=["#B54834"])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        gender_agg = filtered.groupby("Gender")["Purchase"].sum().reset_index()
        fig = px.pie(gender_agg, names="Gender", values="Purchase", title="Purchase share by gender",
                      color_discrete_sequence=["#B54834", "#2F5D8A"])
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        cat_agg = filtered.groupby("Product_Category_1")["Purchase"].sum().reset_index().sort_values("Purchase", ascending=False)
        fig = px.bar(cat_agg, x="Product_Category_1", y="Purchase", title="Total purchase by product category",
                     color_discrete_sequence=["#6B5B4F"])
        fig.update_xaxes(type="category")
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        city_agg = filtered.groupby("City_Category")["Purchase"].sum().reset_index()
        fig = px.bar(city_agg, x="City_Category", y="Purchase", title="Total purchase by city category",
                     color_discrete_sequence=["#8A9A5B"])
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    occ_agg = filtered.groupby("Occupation")["Purchase"].sum().reset_index().sort_values("Purchase", ascending=False)
    fig = px.bar(occ_agg, x="Occupation", y="Purchase", title="Total purchase by occupation",
                 color_discrete_sequence=["#2F5D8A"])
    fig.update_xaxes(type="category")
    st.plotly_chart(fig, use_container_width=True)

# =================================================================
# PAGE 2: CUSTOMER SEGMENTATION (FM)
# =================================================================
elif page == "Customer segmentation (FM)":
    st.subheader("Frequency-Monetary customer segmentation")
    st.info(
        "This dataset has no date/timestamp field \u2014 it's a single Black Friday "
        "snapshot, not a transaction log over time \u2014 so true Recency can't be "
        "computed. Segmentation below uses Frequency (transaction count) and "
        "Monetary (total spend) only.",
        icon="\u2139\uFE0F",
    )

    seg_summary = segments.groupby("Segment").agg(
        Customers=("User_ID", "count"),
        Avg_Frequency=("Frequency", "mean"),
        Avg_Monetary=("Monetary", "mean"),
        Total_Monetary=("Monetary", "sum"),
    ).reset_index()
    seg_summary["Pct_of_Revenue"] = (seg_summary["Total_Monetary"] / segments["Monetary"].sum() * 100).round(1)
    seg_summary["Pct_of_Customers"] = (seg_summary["Customers"] / len(segments) * 100).round(1)
    seg_summary = seg_summary.sort_values("Total_Monetary", ascending=False)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(seg_summary, x="Pct_of_Customers", y="Segment", orientation="h",
                     title="Share of customers by segment", color_discrete_sequence=["#B54834"])
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.bar(seg_summary, x="Pct_of_Revenue", y="Segment", orientation="h",
                     title="Share of revenue by segment", color_discrete_sequence=["#2F5D8A"])
        st.plotly_chart(fig, use_container_width=True)

    champions_pct = seg_summary.loc[seg_summary["Segment"] == "Champions", "Pct_of_Customers"].values
    champions_rev = seg_summary.loc[seg_summary["Segment"] == "Champions", "Pct_of_Revenue"].values
    if len(champions_pct):
        st.success(
            f"Champions make up only {champions_pct[0]}% of customers but drive "
            f"{champions_rev[0]}% of total revenue \u2014 the clearest target for "
            f"loyalty and retention investment."
        )

    st.markdown("---")
    st.subheader("Frequency vs monetary value")
    sample = segments.sample(min(2000, len(segments)), random_state=42)
    fig = px.scatter(sample, x="Frequency", y="Monetary", color="Segment", opacity=0.6,
                      title="Each point is one customer")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Segment summary table")
    st.dataframe(seg_summary.round(1), use_container_width=True)

    st.download_button("Download full customer segment table (CSV)",
                        segments.to_csv(index=False), "customer_fm_segments.csv")

# =================================================================
# PAGE 3: MARKET BASKET ANALYSIS
# =================================================================
elif page == "Market basket analysis":
    st.subheader("Market basket analysis (Apriori)")
    st.info(
        "The dataset has no separate order/basket ID, so each customer's full set "
        "of purchased product categories during the event is treated as one "
        "basket. Rules below are filtered to lift > 1.0 (co-occurring more often "
        "than chance).",
        icon="\u2139\uFE0F",
    )

    min_lift = st.slider("Minimum lift", 1.0, float(rules["lift"].max()), 1.5, 0.05)
    min_conf = st.slider("Minimum confidence", 0.0, 1.0, 0.4, 0.05)

    filtered_rules = rules[(rules["lift"] >= min_lift) & (rules["confidence"] >= min_conf)]
    filtered_rules = filtered_rules.sort_values("lift", ascending=False)

    st.markdown(f"**{len(filtered_rules)}** rules match these thresholds")

    top20 = filtered_rules.head(20).copy()
    top20["rule"] = top20["antecedents"] + " \u2192 " + top20["consequents"]
    if len(top20):
        fig = px.bar(top20.sort_values("lift"), x="lift", y="rule", orientation="h",
                     title="Top rules by lift", color_discrete_sequence=["#B54834"])
        fig.update_layout(height=max(400, 24 * len(top20)))
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(filtered_rules[["antecedents", "consequents", "support", "confidence", "lift"]],
                 use_container_width=True)

# =================================================================
# PAGE 4: PURCHASE PREDICTION
# =================================================================
elif page == "Purchase prediction":
    st.subheader("Predict purchase amount")
    st.caption("Random Forest regression, R\u00b2 = 0.65 on held-out test data. "
               "Product category dominates the prediction \u2014 demographics explain very little of the amount spent per item.")

    model = load_model()

    col1, col2, col3 = st.columns(3)
    with col1:
        gender = st.selectbox("Gender", ["M", "F"])
        age = st.selectbox("Age group", ["0-17", "18-25", "26-35", "36-45", "46-50", "51-55", "55+"])
        occupation = st.selectbox("Occupation code", list(range(21)))
    with col2:
        city = st.selectbox("City category", ["A", "B", "C"])
        stay = st.selectbox("Years in current city", ["0", "1", "2", "3", "4+"])
        marital = st.selectbox("Marital status", [0, 1], format_func=lambda x: "Married" if x == 1 else "Unmarried")
    with col3:
        cat1 = st.selectbox("Product category 1", sorted(df["Product_Category_1"].unique()))
        cat2 = st.selectbox("Product category 2 (0 = none)", [0] + sorted(df["Product_Category_2"].dropna().unique().astype(int).tolist()))
        cat3 = st.selectbox("Product category 3 (0 = none)", [0] + sorted(df["Product_Category_3"].dropna().unique().astype(int).tolist()))

    if st.button("Predict purchase amount", type="primary"):
        input_row = pd.DataFrame([{
            "Gender": gender, "Age": age, "Occupation": occupation,
            "City_Category": city, "Stay_In_Current_City_Years": stay,
            "Marital_Status": marital, "Product_Category_1": cat1,
            "Product_Category_2": cat2, "Product_Category_3": cat3,
        }])
        pred = model.predict(input_row)[0]
        st.metric("Predicted purchase amount", f"${pred:,.0f}")

# =================================================================
# PAGE 5: BUSINESS RECOMMENDATIONS
# =================================================================
else:
    st.subheader("Business recommendations")
    st.markdown("""
| Area | Recommendation |
|---|---|
| **Customer segmentation** | Target customers aged 26-35 with personalized offers and loyalty programs \u2014 they generate the largest share of purchases. Champions (23% of customers) drive 60% of revenue and are the clearest loyalty-program target. |
| **Product strategy** | Increase inventory for Product Category 1 and Category 5, which together drive 56% of total revenue, to reduce stock-outs during peak demand. |
| **Gender marketing** | Create gender-specific campaigns where purchase behavior differs, while validating with average spending metrics rather than transaction counts alone. |
| **Geographic expansion** | Invest more in City Category B regions, which show the strongest purchasing activity by both transaction volume and total revenue. |
| **Occupation-based promotions** | Offer occupation-focused discounts for high-value customer segments (occupation codes 4, 0, and 7 lead in total spend). |
| **Recommendation engine** | Use the market basket rules above (e.g. category 10/13/15 co-purchases) to cross-sell related products and increase average order value. |
""")
    st.subheader("Key insights")
    st.markdown("""
- Customer base is predominantly male: 75.4% of transactions, 76.8% of revenue.
- Customers aged 26-35 generate the largest share of purchases (~40% of revenue).
- Product Category 1 drives the most revenue (37.5%), even though Category 5 has more transactions.
- City Category B leads in transaction volume (42.1%) and revenue ($2.08B).
- Occupation influences both spend level and product-category preference.
- Purchase amounts are moderately right-skewed (skew = 0.62); the top 10% of transactions contribute 20.6% of revenue.
- Champions (23% of customers, high frequency + high spend) generate 60% of total revenue \u2014 the highest-leverage segment for retention investment.
""")
