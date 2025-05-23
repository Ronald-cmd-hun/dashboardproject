import streamlit as st
import pandas as pd
from datetime import datetime
import plotly.express as px

st.set_page_config(page_title="User Segmentation", layout="wide")

st.title("👥 Rule-Based User Segmentation")

# --- Retrieve data and filter selections from session state ---
if 'full_df_original' not in st.session_state or \
   'create_segments' not in st.session_state:
    st.warning("Data not loaded or helper functions missing. Please go to the main Overview page first.")
    st.stop()

full_df_original = st.session_state.full_df_original
create_segments = st.session_state.create_segments 

# Retrieve filter values from session state
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


if not filtered_df.empty:
    filtered_df['Segment'] = filtered_df.apply(create_segments, axis=1)
else:
    filtered_df['Segment'] = pd.Series(dtype='object') 

if filtered_df.empty or 'Segment' not in filtered_df.columns or filtered_df['Segment'].dropna().empty:
    st.warning("No data available for segmentation with the current filters.")
    st.stop()

# --- Visualizations ---
col_segment1, col_segment2 = st.columns(2)
with col_segment1:
    st.subheader("Customer Segment Sizes")
    segment_sizes_df = filtered_df.groupby('Segment')['CustomerID'].nunique().reset_index()
    segment_sizes_df.columns = ['Segment', 'NumberOfCustomers']
    if not segment_sizes_df.empty:
        segment_sizes_df = segment_sizes_df.sort_values('NumberOfCustomers', ascending=False)
        fig_segment_sizes = px.bar(segment_sizes_df, x='Segment', y='NumberOfCustomers', title=None, color='Segment', labels={'Segment': 'Segment', 'NumberOfCustomers': 'Unique Customers'})
        fig_segment_sizes.update_layout(showlegend=False, height=300, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_segment_sizes, use_container_width=True)
    else: st.info("No data for 'Customer Segment Sizes'.")

    st.subheader("Most Frequent Country per Segment")
    unique_customers_segments_df = filtered_df[['CustomerID', 'Segment', 'Country']].drop_duplicates(subset=['CustomerID'])
    if not unique_customers_segments_df.empty and not unique_customers_segments_df['Segment'].dropna().empty:
        country_per_segment_list = []
        for segment_name in unique_customers_segments_df['Segment'].unique():
            if pd.isna(segment_name): continue # Skip if segment is NaN
            customers_in_segment = unique_customers_segments_df[unique_customers_segments_df['Segment'] == segment_name]
            if not customers_in_segment.empty:
                most_freq_country = customers_in_segment['Country'].mode()
                country_per_segment_list.append({'Segment': segment_name, 'Most Frequent Country': most_freq_country[0] if not most_freq_country.empty else 'N/A'})
        if country_per_segment_list:
            country_per_segment_df = pd.DataFrame(country_per_segment_list).sort_values('Segment')
            st.dataframe(country_per_segment_df.set_index('Segment'), height=200, use_container_width=True) # Use st.dataframe for tables
        else: st.info("Not enough data for 'Most Frequent Country per Segment'.")
    else: st.info("No segment data for country analysis.")

with col_segment2:
    st.subheader("Average Transactions per Customer in Segment")
    transactions_per_customer_df = filtered_df.groupby(['Segment', 'CustomerID'])['TransactionID'].count().reset_index(name='NumTransactions')
    if not transactions_per_customer_df.empty and not transactions_per_customer_df['Segment'].dropna().empty:
        avg_transactions_segment_df = transactions_per_customer_df.groupby('Segment')['NumTransactions'].mean().reset_index()
        avg_transactions_segment_df = avg_transactions_segment_df.sort_values('NumTransactions', ascending=False)
        fig_avg_transactions = px.bar(avg_transactions_segment_df, x='Segment', y='NumTransactions', title=None, color='Segment', labels={'Segment': 'Segment', 'NumTransactions': 'Avg Transactions/Customer'})
        fig_avg_transactions.update_layout(showlegend=False, height=300, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_avg_transactions, use_container_width=True)
    else: st.info("No data for 'Average Transactions per Customer in Segment'.")

    st.subheader("Sales Distribution by Resource Category per Segment")
    if 'Product' in filtered_df.columns: 
        def map_product_to_resource(product_name):
            if "Virtual Assistant" in product_name or "DEX Solution" in product_name: return 'Core AI Services'
            if "Prototyping" in product_name: return 'Dev Tools'
            if "Analyzer" in product_name or "Analytics Suite" in product_name: return 'Analytics'
            return 'Support Packages'
        filtered_df['ResourceCategory'] = filtered_df['Product'].apply(map_product_to_resource)
        segment_resource_sales = filtered_df.groupby(['Segment', 'ResourceCategory'])['SaleAmount'].sum().reset_index()
        if not segment_resource_sales.empty and not segment_resource_sales['Segment'].dropna().empty:
            fig_stacked_bar_segment = px.bar(segment_resource_sales, x='Segment', y='SaleAmount', color='ResourceCategory', title=None, labels={'Segment': 'Segment', 'SaleAmount': 'Total Sales ($)', 'ResourceCategory': 'Resource Category'}, color_discrete_sequence=px.colors.qualitative.Set2)
            fig_stacked_bar_segment.update_layout(barmode='stack', height=300, legend_title_text=None, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_stacked_bar_segment, use_container_width=True)
        else: st.info("No data for 'Sales Distribution by Resource Category per Segment'.")
    else: st.info("Product data needed for 'Resource Category' analysis is missing.")
