import streamlit as st
import matplotlib.pyplot as plt
from src.mba_model import load_data, create_basket, generate_rules, get_recommendations, get_top_products

# Configure page
st.set_page_config(page_title="🛒 Grocery MBA", layout="wide")

# Cache data loading
@st.cache_data
def load_all():
    data = load_data("data/groceries.csv")
    basket = create_basket(data)
    rules = generate_rules(basket)
    return data, basket, rules

data, basket, rules = load_all()

# --- Main Interface ---
st.title("🛒 Smart Grocery Recommender")
st.write("Discover products that are frequently bought together!")

# Visualization 1 - Top Products
st.subheader("🔥 Top Selling Products")
top10 = data['itemDescription'].value_counts().head(10)
fig1, ax1 = plt.subplots()
top10.sort_values().plot(kind='barh', color='teal', ax=ax1)
plt.xlabel("Total Purchases")
st.pyplot(fig1)

selected = st.selectbox(
    "Select a product you purchased:", 
    sorted(data['itemDescription'].unique())
)  # Added missing closing parenthesis here

# Generate Recommendations
if st.button("Show Recommendations 🚀"):
    recs = get_recommendations(rules, selected)
    
    if recs:
        st.subheader("🎯 Recommended Pairings")
        cols = st.columns(3)
        for i, rec in enumerate(recs[:6]):
            cols[i%3].success(rec)
    else:
        st.warning("No strong associations found. Try these popular items:")
        for prod in get_top_products(data):
            st.markdown(f"- {prod}")

# Visualization 2 - Rules Analysis
st.subheader("📈 Rule Quality Analysis")
fig2, ax2 = plt.subplots()
sc = ax2.scatter(rules['support'], rules['confidence'], 
                c=rules['lift'], cmap='coolwarm', alpha=0.7)
plt.colorbar(sc, label='Lift Score')
ax2.set(xlabel="Support", ylabel="Confidence")
st.pyplot(fig2)
