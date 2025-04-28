import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

def load_data(path):
    """Load grocery data."""
    data = pd.read_csv(path)
    return data

def create_basket(data):
    """Create basket format for MBA."""
    data['Transaction'] = data['Member_number'].astype(str) + "_" + data['Date']

    basket = (data.groupby(['Transaction', 'itemDescription'])['itemDescription']
              .count().unstack().reset_index().fillna(0)
              .set_index('Transaction'))

    # Fix both warnings with this change
    basket = (basket > 0).astype(bool)  # Convert directly to boolean
    return basket

def generate_rules(basket, min_support=0.001, min_lift=0.5):
    """Generate association rules."""
    frequent_itemsets = apriori(basket, min_support=min_support, use_colnames=True)
    if frequent_itemsets.empty:
        return pd.DataFrame()
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=min_lift)
    return rules

def get_recommendations(rules, product_name):
    """Recommend products based on a selected product."""
    if rules.empty:
        return []

    recommendations = rules[rules['antecedents'].apply(lambda x: product_name in x)]
    
    if recommendations.empty:
        return []
    
    # Sort by confidence first, then by lift if confidence is the same
    recommendations = recommendations.sort_values(by=['confidence', 'lift'], ascending=False)

    # Remove duplicates - keep the highest confidence or lift for each item
    unique_recommendations = []
    seen_items = set()
    
    for idx, row in recommendations.iterrows():
        # Get the consequent product from the rule
        consequents = list(row['consequents'])
        
        # If the product has already been recommended, skip it
        for item in consequents:
            if item not in seen_items:
                unique_recommendations.append(f"{item} (Confidence: {row['confidence']:.2f}, Lift: {row['lift']:.2f})")
                seen_items.add(item)
    
    return unique_recommendations

def get_top_products(data, top_n=5):
    """Get top selling products."""
    top_products = data['itemDescription'].value_counts().head(top_n).index.tolist()
    return top_products