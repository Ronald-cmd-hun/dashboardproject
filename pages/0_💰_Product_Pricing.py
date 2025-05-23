import streamlit as st
import pandas as pd

st.set_page_config(page_title="Product Pricing", layout="wide")
st.title("💰 Product Pricing Catalog")

if 'products_dict' not in st.session_state:
    st.warning("Product data not available. Please ensure the main application has run.")
    st.stop()

products_dict = st.session_state.products_dict
if products_dict:
    pricing_data = []
    for name, price in products_dict.items():
        pricing_data.append({"Product Name": name, "Base Price ($)": price})
    
    pricing_df = pd.DataFrame(pricing_data)
    pricing_df = pricing_df.sort_values(by="Base Price ($)", ascending=False)

    st.markdown("Below is a list of our AI solutions and their base pricing:")
    st.dataframe(
        pricing_df, 
        use_container_width=True,
        hide_index=True,
        column_config={
            "Product Name": st.column_config.TextColumn("Product Name", width="large"),
            "Base Price ($)": st.column_config.NumberColumn("Base Price ($)", format="$ %d"),
        }
    )
else:
    st.info("No product pricing information is currently available.")

st.markdown("---")
st.caption("Note: Prices are base prices and may vary based on customization, license length, and support packages.")
