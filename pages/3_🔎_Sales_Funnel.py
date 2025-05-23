import streamlit as st
import pandas as pd
from datetime import datetime
import plotly.express as px

st.set_page_config(page_title="Sales Funnel", layout="wide")

st.title("🔎 Sales Funnel Analysis (Simulated)")

# --- Retrieve data and filter selections from session state ---
if 'full_df_original' not in st.session_state:
    st.warning("Data not loaded. Please go to the main Overview page first.")
    st.stop()

full_df_original = st.session_state.full_df_original

selected_start_datetime = datetime.combine(st.session_state.selected_start_date, datetime.min.time())
selected_end_datetime = datetime.combine(st.session_state.selected_end_date, datetime.max.time())
selected_countries = st.session_state.selected_countries
selected_industries = st.session_state.selected_industries
selected_products = st.session_state.selected_products
selected_channels = st.session_state.selected_channels
current_view = st.session_state.current_view
selected_rep = st.session_state.get('selected_rep', None)

all_countries = sorted(full_df_original['Country'].unique())
all_industries = sorted(full_df_original['Industry'].unique())
all_products = sorted(full_df_original['Product'].unique())
all_channels = sorted(full_df_original['SaleChannel'].unique())

# --- Filter Data for this page ---
filtered_df = full_df_original.copy()
filtered_df = filtered_df[
    (filtered_df['SaleDate'] >= selected_start_datetime) & 
    (filtered_df['SaleDate'] <= selected_end_datetime) &
    (filtered_df['Country'].isin(selected_countries if selected_countries else all_countries)) &
    (filtered_df['Industry'].isin(selected_industries if selected_industries else all_industries)) &
    (filtered_df['Product'].isin(selected_products if selected_products else all_products)) &
    (filtered_df['SaleChannel'].isin(selected_channels if selected_channels else all_channels))
]
if current_view == "Individual Rep" and selected_rep:
    filtered_df = filtered_df[filtered_df['SalesRep'] == selected_rep]


if filtered_df.empty:
    st.warning("No data matches the current filter criteria for this page.")
    st.stop()

unique_customers_filtered = filtered_df['CustomerID'].nunique() 

if unique_customers_filtered > 0:
    val_closed_deals = unique_customers_filtered 
    val_proposals = int(val_closed_deals * 1.5) 
    val_demo_requests = int(val_proposals * 1.8)   
    val_visits = int(val_demo_requests * 2.5)      
    funnel_stages = ['Website Visits', 'Demo Requests', 'Proposals Sent', 'Closed Deals']
    funnel_values = [val_visits, val_demo_requests, val_proposals, val_closed_deals]
    for i in range(len(funnel_values)):
        if funnel_values[i] == 0 and (i > 0 and funnel_values[i-1] > 0):
                funnel_values[i] = 1 
        if i > 0 and funnel_values[i] > funnel_values[i-1]: 
            funnel_values[i] = funnel_values[i-1]
    funnel_data = pd.DataFrame(dict(number=funnel_values, stage=funnel_stages))
    fig_funnel = px.funnel(funnel_data, x='number', y='stage', title=None) # Removed title for compactness
    fig_funnel.update_layout(yaxis_title=None, xaxis_title="Count", 
                             height=380, margin=dict(l=10, r=10, t=10, b=10)) # Reduced height and margins
    st.plotly_chart(fig_funnel, use_container_width=True)
    st.caption("Note: Funnel data is simulated based on the number of unique customers in the filtered data.")
else:
    st.info("Not enough unique customer data to display sales funnel with current filters.")
