# Black Friday sales analysis

End-to-end analysis of 537,577 Black Friday transactions from 5,891 customers —
covering exploratory analysis, an interactive dashboard, customer
segmentation, market basket analysis, purchase-amount prediction, and a
Streamlit web app that ties it all together.

**Dataset:** [Black Friday Sales Data](https://github.com/GeeksforgeeksDS/Data-Analysis-with-Python-GFG/blob/main/20.%20Black%20Friday%20-%20Walkthrough/BlackFriday.csv) — 537,577 rows, 12 columns (User ID, Product ID, Gender, Age, Occupation, City Category, Stay in Current City Years, Marital Status, Product Categories, Purchase amount).

---

## Contents

```
black-friday-sales-analysis/
├── data/                          Cleaned datasets and computed outputs
│   ├── BlackFriday.csv               Original source data
│   ├── black_friday_powerbi.csv      Power BI-ready export
│   ├── customer_fm_segments.csv      Per-customer FM segmentation
│   ├── fm_segment_summary.csv        Segment-level rollup
│   ├── market_basket_rules.csv       Apriori association rules
│   ├── frequent_itemsets.csv         Frequent category itemsets
│   ├── model_comparison.csv          Regression model metrics
│   └── feature_importance.csv        Random Forest feature importances
├── notebooks/                     Analysis scripts (run in order)
│   ├── 01_eda_analysis.py            Full walkthrough + all charts
│   ├── 02_fm_segmentation.py         Frequency/Monetary segmentation
│   ├── 03_market_basket_analysis.py  Apriori market basket analysis
│   └── 04_predictive_modeling.py     Purchase amount regression models
├── dashboards/
│   ├── black_friday_dashboard.html   Standalone interactive dashboard (open in any browser)
│   └── powerbi_dashboard_guide.md    Power BI build guide (DAX measures, layout)
├── reports/
│   ├── Black_Friday_Business_Report.docx  Business recommendations + key insights
│   └── analysis_output.txt                Full console output from the EDA run
├── streamlit_app/                 Interactive web app (see its own README)
│   ├── app.py
│   ├── requirements.txt
│   └── data/                         Data files the app needs to run
├── images/                         All chart PNGs referenced in the reports
└── README.md
```

---

## Key insights

- Customer base is predominantly male: 75.4% of transactions, 76.8% of revenue.
- Customers aged 26-35 generate the largest share of purchases (~40% of revenue).
- Product Category 1 drives the most revenue (37.5%), even though Category 5 has more transactions — Category 1 purchases run at a higher average value.
- City Category B leads in transaction volume (42.1%) and revenue ($2.08B).
- Occupation influences both spend level and product-category preference — occupation codes 4, 0, and 7 lead in total spend.
- Purchase amounts are moderately right-skewed (skew = 0.62); the top 10% of transactions contribute 20.6% of revenue.
- **Champions** (23% of customers by FM segmentation) generate **60% of total revenue** — the highest-leverage segment for retention investment.
- Product category, not customer demographics, is what actually drives purchase amount: a Random Forest model gets R² = 0.65, and Product_Category_1 alone accounts for 95% of its predictive power.

Full detail and supporting charts are in `reports/Black_Friday_Business_Report.docx`.

## Business recommendations

| Area | Recommendation |
|---|---|
| Customer segmentation | Target 26-35s and "Champions" (high frequency + high spend) with personalized offers and loyalty programs. |
| Product strategy | Increase inventory for Category 1 and Category 5 (56% of revenue combined) to reduce stock-outs. |
| Gender marketing | Build gender-specific campaigns, but validate with average spend, not just transaction volume — the gap there is much smaller. |
| Geographic expansion | Invest more in City Category B, the strongest region by both volume and revenue. |
| Occupation-based promotions | Focus discounts on occupation codes 4, 0, and 7. |
| Recommendation engine | Use the market basket rules (e.g. categories 10/13/15 co-occur with lift up to 1.91) to cross-sell and lift average order value. |

## Methodology notes and honest limitations

- **RFM → FM segmentation**: the dataset has no date/timestamp field — it's a
  single Black Friday event snapshot, not a transaction log over time — so
  Recency cannot be computed. Segmentation uses Frequency and Monetary value
  only, and is labeled as FM (not RFM) throughout.
- **Market basket "baskets"**: there's no separate order/basket ID in the
  source data, so each customer's full set of purchased product categories is
  treated as one basket. This is a stated proxy, not a true multi-item-order
  signal.
- **Purchase prediction**: trained on `Gender`, `Age`, `Occupation`,
  `City_Category`, `Stay_In_Current_City_Years`, `Marital_Status`, and the
  three `Product_Category` columns. Random Forest (R² = 0.65) substantially
  outperforms Linear Regression (R² = 0.14); see `data/model_comparison.csv`.

## How to run each piece

### 1. Exploratory analysis
```bash
pip install pandas matplotlib seaborn
python notebooks/01_eda_analysis.py
```

### 2. Customer segmentation
```bash
python notebooks/02_fm_segmentation.py
```

### 3. Market basket analysis
```bash
pip install mlxtend
python notebooks/03_market_basket_analysis.py
```

### 4. Predictive modeling
```bash
pip install scikit-learn joblib
python notebooks/04_predictive_modeling.py
```

### 5. Interactive dashboard (no install needed)
Open `dashboards/black_friday_dashboard.html` directly in any browser.

### 6. Power BI dashboard
Follow `dashboards/powerbi_dashboard_guide.md` — import `data/black_friday_powerbi.csv` into Power BI Desktop and paste in the provided DAX measures.

### 7. Streamlit web app
```bash
cd streamlit_app
pip install -r requirements.txt
streamlit run app.py
```
See `streamlit_app/README.md` for full details on each page.

## Tech stack

Python, pandas, matplotlib, seaborn, scikit-learn, mlxtend (Apriori), joblib,
Streamlit, Plotly, Power BI (guide only), HTML/CSS/JS + Chart.js for the
standalone dashboard.

## Source

Walkthrough structure adapted from [GeeksforGeeks Data Analysis with Python](https://github.com/GeeksforgeeksDS/Data-Analysis-with-Python-GFG); all analysis, segmentation, modeling, dashboards, and the Streamlit app in this repo are original work built on top of that dataset.
