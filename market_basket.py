import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

df = pd.read_csv("Black Friday Datset.csv")

# ---------------------------------------------------------------
# Basket definition
# The dataset has no separate order/basket ID -- each row is one
# product purchased by one customer during the Black Friday event.
# We treat each customer's full set of purchased Product_Category_1
# values as one "basket" (a reasonable proxy for a single shopping
# event, since it's all Black-Friday-day activity). This is a
# necessary assumption given the data available and is called out
# explicitly here and in the final report.
# ---------------------------------------------------------------
baskets = df.groupby("User_ID")["Product_Category_1"].apply(lambda x: list(set(x))).tolist()
print("Number of baskets (customers):", len(baskets))
print("Sample basket:", baskets[0])

te = TransactionEncoder()
te_ary = te.fit(baskets).transform(baskets)
basket_df = pd.DataFrame(te_ary, columns=[f"Cat_{c}" for c in te.columns_])

print("\nBasket matrix shape:", basket_df.shape)

# Frequent itemsets
# min_support raised to 0.15 and itemset length capped at 3: three
# categories (1, 5, 8) appear in 90%+ of baskets, so a low support
# threshold explodes into tens of thousands of near-universal
# combinations that don't tell us anything about co-purchase behavior.
frequent_itemsets = apriori(basket_df, min_support=0.15, use_colnames=True, max_len=3, low_memory=True)
frequent_itemsets = frequent_itemsets.sort_values("support", ascending=False)
print("\nNumber of frequent itemsets (min_support=0.15, max_len=3):", len(frequent_itemsets))
print(frequent_itemsets.head(15))

# Association rules
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.1)
rules = rules.sort_values("lift", ascending=False)

# Keep readable columns
rules_display = rules[["antecedents", "consequents", "support", "confidence", "lift"]].copy()
rules_display["antecedents"] = rules_display["antecedents"].apply(lambda x: ", ".join(sorted(x)))
rules_display["consequents"] = rules_display["consequents"].apply(lambda x: ", ".join(sorted(x)))
rules_display = rules_display.round(3)

print("\nTop 15 association rules by lift:")
print(rules_display.head(15).to_string(index=False))

rules_display.to_csv("market_basket_rules.csv", index=False)
frequent_itemsets_display = frequent_itemsets.copy()
frequent_itemsets_display["itemsets"] = frequent_itemsets_display["itemsets"].apply(lambda x: ", ".join(sorted(x)))
frequent_itemsets_display.round(3).to_csv("frequent_itemsets.csv", index=False)

print("\nSaved: market_basket_rules.csv, frequent_itemsets.csv")
print("\nTotal rules found:", len(rules_display))