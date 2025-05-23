import streamlit as st
import pandas as pd
from datetime import datetime

# Helper function (can be in a utils.py if used by many pages)
@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

st.set_page_config(page_title="Data Export", layout="wide")

st.title("💾 Data Export")
st.markdown("Download the **filtered** sales dataset (including Segments) as a CSV file.")

# --- Retrieve data and filter selections from session state ---
if 'full_df_original' not in st.session_state or \
   'create_segments' not in st.session_state: # Ensure segmentation function is available
    st.warning("Data not loaded or helper functions missing. Please go to the main Overview page first.")
    st.stop()

full_df_original = st.session_state.full_df_original
create_segments = st.session_state.create_segments # Get segmentation function

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

# --- Filter Data for export ---
filtered_df_export = full_df_original.copy()

# Apply general filters
filtered_df_export = filtered_df_export[
    (filtered_df_export['SaleDate'] >= selected_start_datetime) & 
    (filtered_df_export['SaleDate'] <= selected_end_datetime) &
    (filtered_df_export['Country'].isin(selected_countries if selected_countries else all_countries)) &
    (filtered_df_export['Industry'].isin(selected_industries if selected_industries else all_industries)) &
    (filtered_df_export['Product'].isin(selected_products if selected_products else all_products)) &
    (filtered_df_export['SaleChannel'].isin(selected_channels if selected_channels else all_channels))
]

# Apply Sales Rep filter if in individual view
if current_view == "Individual Rep" and selected_rep:
    filtered_df_export = filtered_df_export[filtered_df_export['SalesRep'] == selected_rep]


if not filtered_df_export.empty:
    filtered_df_export['Segment'] = filtered_df_export.apply(create_segments, axis=1)
else:
    # If filtered_df_export is already empty, ensure 'Segment' column exists for consistency
    filtered_df_export['Segment'] = pd.Series(dtype='object')


if not filtered_df_export.empty:
    st.markdown("---")
    st.subheader("Filtered Data Preview (First 5 Records)")
    # Display a sample of the DataFrame that will be downloaded
    # To make it fit better, you might select only a subset of columns for preview if the df is very wide
    # For example: st.dataframe(filtered_df_export[['SaleDate', 'Product', 'SaleAmount', 'Country', 'Segment']].head())
    st.dataframe(filtered_df_export.head(), use_container_width=True)
    
    st.markdown("---")
    csv_data = convert_df_to_csv(filtered_df_export) 
    st.download_button(
        label="Download Filtered Sales Data (CSV)", 
        data=csv_data, 
        file_name='filtered_ai_sales_data_with_segments.csv', 
        mime='text/csv',
        key='download_button_export_page' # Added a key for uniqueness
    )
else:
    st.info("No data to preview or export based on current filters.")

st.markdown("---")
st.info("The downloaded data reflects all active filters from the sidebar (View As, Date Range, Customer & Product Filters).")
