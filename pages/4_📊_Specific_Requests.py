import streamlit as st
import pandas as pd
import random 
from datetime import datetime
import plotly.express as px

st.set_page_config(page_title="Specific Requests", layout="wide")

st.title("📊 Specific Request Analysis (Simulated)")
st.caption("This data is simulated and scales with the number of transactions in the filtered data.")

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

request_types = ['Schedule Demo', 'AI Assistant Query', 'Job Application', 'Promotion Info', 'Support Ticket']
base_scale_factor = 0.005 
request_counts = [random.randint(max(1, int(len(filtered_df) * base_scale_factor * 0.5)), 
                                    max(5, int(len(filtered_df) * base_scale_factor * 2.0))) 
                    for _ in request_types]

specific_requests_df = pd.DataFrame({'RequestType': request_types, 'Count': request_counts})

if not specific_requests_df.empty:
    fig_specific_requests = px.bar(specific_requests_df, x='RequestType', y='Count', title=None, color='RequestType', labels={'RequestType': 'Request Type', 'Count': 'Number of Requests'})
    fig_specific_requests.update_layout(showlegend=False, height=350, margin=dict(l=10, r=10, t=10, b=10)) # Reduced height and margins
    st.plotly_chart(fig_specific_requests, use_container_width=True)
else: 
    st.info("No data for 'Specific Request Analysis'.")
