import streamlit as st
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import uuid
import plotly.graph_objects as go 
import plotly.express as px 

# --- Set Page Configuration (do this first) ---
st.set_page_config(
    page_title="Sales Team Dashboard", 
    page_icon="🚀",
    layout="wide", 
    initial_sidebar_state="collapsed" # Start with sidebar collapsed for login
)

# --- Custom CSS for font sizes and ultra-compactness ---
st.markdown("""
    <style>
    /* Reduce font size for st.title (if any are used on other pages) */
    h1 {
        font-size: 24px !important; 
        margin-bottom: 0.2rem !important;
    }
    /* Reduce font size for st.header */
    h2 {
        font-size: 18px !important; 
        margin-top: 0.5rem !important;
        margin-bottom: 0.2rem !important;
    }
    /* Font size for st.subheader (graph titles) */
    h3 {
        font-size: 17px !important; 
        margin-top: 0.5rem !important; /* Space above subheader */
        margin-bottom: 0.3rem !important; /* Increased space below subheader for visibility */
    }
    /* Reduce font size for st.caption, if needed */
    .caption { 
        font-size: 0.7rem !important; 
        margin-top: -0.5rem !important; 
    }
    /* Reduce font size for st.markdown used as subheaders, if any were styled like h4 */
    h4 { 
        font-size: 14px !important; 
        margin-bottom: 0.1rem !important;
    }

    /* Ultra-compact main container */
    .main .block-container {
        padding-top: 0.2rem !important;
        padding-bottom: 0.1rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
    
    /* Chart spacing - Reduced negative top margin */
    .stPlotlyChart {
        margin-top: -10px !important; 
        margin-bottom: -15px !important;
    }
    
    /* Tighter header spacing (for markdown headers) */
    .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 { 
        margin-top: 0.1rem !important;
        margin-bottom: 0.1rem !important;
        padding-bottom: 0 !important; 
    }
    
    /* Ultra-compact metrics */
    div[data-testid="stMetric"] {
        margin-bottom: -10px !important;
        padding: 0.2rem !important;
    }
    
    /* Smaller metric values */
    div[data-testid="stMetricValue"] {
        font-size: 1.1rem !important;
    }
    
    /* Smaller metric labels */
    div[data-testid="stMetricLabel"] {
        font-size: 0.85rem !important;
    }
    
    /* Remove space around dividers */
    hr { 
        margin-top: 0.1rem !important;
        margin-bottom: 0.1rem !important;
    }
    
    /* Compact info boxes */
    div[data-testid="stAlert"] { 
        padding-top: 0.3rem !important;
        padding-bottom: 0.3rem !important;
        padding-left: 0.5rem !important;
        padding-right: 0.5rem !important;
        margin-top: 0.2rem !important;
        margin-bottom: 0.2rem !important;
        font-size: 0.8rem !important; 
    }

    /* Reduce space specifically for Streamlit's default headers if not markdown */
     div[data-testid="stHeader"] {
        padding-bottom: 2px !important; 
        margin-bottom: 0.2rem !important;
     }
     /* Space for st.subheader */
     div[data-testid="stSubheader"] {
        padding-bottom: 2px !important; 
        margin-bottom: 0.3rem !important; 
     }
    </style>
    """, unsafe_allow_html=True)


# --- Random Seed for Reproducibility ---
np.random.seed(42)
random.seed(42)

# --- Initialize session state variables ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'sales_reps_list' not in st.session_state: # Initialize early if needed by login or other pre-dashboard logic
    st.session_state.sales_reps_list = [f"Rep {chr(65+i)}" for i in range(10)] 

# --- Login Function ---
def check_login(username, password):
    # Hardcoded credentials for demonstration
    # In a real app, use a secure method (e.g., hashing, database)
    correct_username = "admin"
    correct_password = "password123"
    if username == correct_username and password == correct_password:
        return True
    return False

# --- Helper function to convert DataFrame to CSV for download ---
@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

# --- Data Generation Functions ---
@st.cache_data 
def generate_data(sales_reps_for_assignment): 
    """Generates the full sales, customer, and sales rep dataset."""
    num_customers = 5000 
    num_transactions = 28000 
    
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 6, 30)
    
    products_dict = {
        "AI Virtual Assistant Basic": 299, "AI Virtual Assistant Pro": 599, "AI Virtual Assistant Enterprise": 999,
        "Rapid Prototyping Starter": 499, "Rapid Prototyping Advanced": 899,
        "Digital Experience Analyzer": 799, "AI-Powered Analytics Suite": 1299, "Complete DEX Solution": 2499
    }
    if 'products_dict' not in st.session_state:
        st.session_state.products_dict = products_dict 
    
    product_names_list = list(products_dict.keys())
    industries = ["Technology", "Healthcare", "Finance", "Education", "Manufacturing", "Retail", "Government", "Telecommunications"]
    countries = ["UK", "USA", "Germany", "France", "Canada", "Australia", "Japan", "Singapore", "Brazil", "South Africa"]
    channels = ["Website", "Direct Sales", "Partner", "Trade Show", "Referral"]
    
    customer_ids = [str(uuid.uuid4()) for _ in range(num_customers)]
    customer_data = []
    for i, cust_id in enumerate(customer_ids):
        customer_data.append({
            "CustomerID": cust_id, "Company": f"Company_{i+1:05d}", "Industry": random.choice(industries),
            "Country": random.choice(countries), "EmployeeCount": random.choice(["1-50", "51-200", "201-500", "501-1000", "1000+"]),
            "FirstContactDate": (start_date + timedelta(days=random.randint(0, (end_date - start_date).days))).strftime('%Y-%m-%d')
        })
    customers_df = pd.DataFrame(customer_data)

    transactions = []
    for _ in range(num_transactions):
        customer_id = random.choice(customer_ids)
        product_name = random.choice(product_names_list)
        base_price = products_dict[product_name]
        price_variation = random.uniform(0.9, 1.1) 
        final_price = round(base_price * price_variation, 2)
        transaction_date_dt = start_date + timedelta(days=random.randint(0, (end_date - start_date).days))
        transactions.append({
            "TransactionID": str(uuid.uuid4()), "CustomerID": customer_id, "Product": product_name,
            "SaleDate": transaction_date_dt, 
            "SaleAmount": final_price, "SaleChannel": random.choice(channels),
            "IsRenewal": random.choices([True, False], weights=[0.3, 0.7])[0],
            "LicenseLength": random.choice([1, 2, 3]), 
            "SupportPackage": random.choice(["Basic", "Standard", "Premium"]),
            "DealSizeCategory": random.choice(["SMB", "Mid-Market", "Enterprise"]),
            "SalesRep": random.choice(sales_reps_for_assignment) 
        })
    sales_df = pd.DataFrame(transactions)
    full_df = pd.merge(sales_df, customers_df, on="CustomerID")
    full_df['SaleMonth'] = full_df['SaleDate'].dt.to_period('M').astype(str) 
    full_df['SaleYearMonthNum'] = full_df['SaleDate'].dt.strftime('%Y-%m')
    return full_df, customers_df

# --- Rule-Based Segmentation Function ---
def create_segments(row):
    if row['SaleAmount'] > 1000 and row['LicenseLength'] > 1: 
        return "High Value / Long-term"
    elif row['SaleAmount'] > 1000:
        return "High Value / Short-term"
    elif row['LicenseLength'] > 1:
        return "Low Value / Long-term"
    return "Low Value / Short-term"


# --- Main App Logic ---
if not st.session_state.logged_in:
    st.title("🔒 Sales Dashboard Login")
    with st.form("login_form"):
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        submitted = st.form_submit_button("Login")

        if submitted:
            if check_login(username, password):
                st.session_state.logged_in = True
                st.rerun() # Rerun to hide login and show dashboard
            else:
                st.error("Incorrect username or password.")
else:
    # --- Load and Prepare Data (Only if logged in) ---
    if 'full_df_original' not in st.session_state: 
        with st.spinner("Generating initial dataset... Please wait."):
            full_df, cust_df = generate_data(st.session_state.sales_reps_list)
            st.session_state.full_df_original = full_df
            st.session_state.customers_df_original = cust_df
    full_df_original = st.session_state.full_df_original
    customers_df_original = st.session_state.customers_df_original


    # --- Define Sales Targets (Global) ---
    monthly_targets_data = {
        '2023-01': 3500000, '2023-02': 3700000, '2023-03': 4000000,
        '2023-04': 4200000, '2023-05': 4500000, '2023-06': 4800000,
    } 
    targets_df = pd.DataFrame(list(monthly_targets_data.items()), columns=['SaleYearMonthNum', 'TargetAmount'])
    targets_df['SaleMonth'] = pd.to_datetime(targets_df['SaleYearMonthNum']).dt.to_period('M').astype(str)
    st.session_state.targets_df = targets_df 

    # --- Sidebar ---
    st.sidebar.markdown("### Sales Team Dashboard") 
    st.sidebar.markdown("---") 

    st.sidebar.subheader("🎯 View As")
    if 'current_view' not in st.session_state:
        st.session_state.current_view = "Sales Team" 
    view_options = ["Sales Team", "Individual Rep"]
    st.session_state.current_view = st.sidebar.radio("Select View:", view_options, index=view_options.index(st.session_state.current_view), key="view_selector")

    if st.session_state.current_view == "Individual Rep":
        if not st.session_state.sales_reps_list: 
            st.sidebar.warning("No sales reps available.")
            st.session_state.selected_rep = None
            current_rep_index = 0 
        elif 'selected_rep' not in st.session_state or st.session_state.selected_rep is None or st.session_state.selected_rep not in st.session_state.sales_reps_list:
            st.session_state.selected_rep = st.session_state.sales_reps_list[0]
            current_rep_index = 0
        else:
            current_rep_index = st.session_state.sales_reps_list.index(st.session_state.selected_rep)
        if st.session_state.sales_reps_list: 
            st.session_state.selected_rep = st.sidebar.selectbox("Select Sales Rep:", st.session_state.sales_reps_list, index=current_rep_index, key="rep_selector")
    else:
        st.session_state.selected_rep = None 

    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ Dashboard Filters") 
    st.sidebar.subheader("🗓️ Sale Date Range") 
    min_date = st.session_state.full_df_original['SaleDate'].min().date()
    max_date = st.session_state.full_df_original['SaleDate'].max().date()
    if 'selected_start_date' not in st.session_state: st.session_state.selected_start_date = min_date
    if 'selected_end_date' not in st.session_state: st.session_state.selected_end_date = max_date
    st.session_state.selected_start_date = st.sidebar.date_input("Start Date", st.session_state.selected_start_date, min_value=min_date, max_value=max_date, key="start_date_widget")
    st.session_state.selected_end_date = st.sidebar.date_input("End Date", st.session_state.selected_end_date, min_value=st.session_state.selected_start_date, max_value=max_date, key="end_date_widget")
    selected_start_datetime = datetime.combine(st.session_state.selected_start_date, datetime.min.time())
    selected_end_datetime = datetime.combine(st.session_state.selected_end_date, datetime.max.time())

    st.sidebar.subheader("🌍 Customer & Product Filters") 
    all_countries = sorted(st.session_state.full_df_original['Country'].unique())
    if 'selected_countries' not in st.session_state: st.session_state.selected_countries = all_countries
    st.session_state.selected_countries = st.sidebar.multiselect("Select Countries", all_countries, default=st.session_state.selected_countries, key="countries_widget")

    all_industries = sorted(st.session_state.full_df_original['Industry'].unique())
    if 'selected_industries' not in st.session_state: st.session_state.selected_industries = all_industries
    st.session_state.selected_industries = st.sidebar.multiselect("Select Industries", all_industries, default=st.session_state.selected_industries, key="industries_widget")

    all_products = sorted(st.session_state.full_df_original['Product'].unique())
    if 'selected_products' not in st.session_state: st.session_state.selected_products = all_products
    st.session_state.selected_products = st.sidebar.multiselect("Select Products", all_products, default=st.session_state.selected_products, key="products_widget")

    all_channels = sorted(st.session_state.full_df_original['SaleChannel'].unique())
    if 'selected_channels' not in st.session_state: st.session_state.selected_channels = all_channels
    st.session_state.selected_channels = st.sidebar.multiselect("Select Sale Channels", all_channels, default=st.session_state.selected_channels, key="channels_widget")

    st.session_state.create_segments = create_segments

    # --- Logout Button ---
    if st.sidebar.button("Logout", key="logout_button"):
        st.session_state.logged_in = False
        # Clear other session state variables related to filters if desired
        for key in ['current_view', 'selected_rep', 'selected_start_date', 'selected_end_date', 
                    'selected_countries', 'selected_industries', 'selected_products', 'selected_channels']:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

    # --- Main Page Content (Overview / KPIs & Sales Performance) ---
    # Main page title is removed, content starts with KPIs
    
    # Filter Data for the Main Page 
    df_main_page = st.session_state.full_df_original.copy() 
    df_main_page = df_main_page[(df_main_page['SaleDate'] >= selected_start_datetime) & (df_main_page['SaleDate'] <= selected_end_datetime)]
    if st.session_state.selected_countries: df_main_page = df_main_page[df_main_page['Country'].isin(st.session_state.selected_countries)]
    if st.session_state.selected_industries: df_main_page = df_main_page[df_main_page['Industry'].isin(st.session_state.selected_industries)]
    if st.session_state.selected_products: df_main_page = df_main_page[df_main_page['Product'].isin(st.session_state.selected_products)]
    if st.session_state.selected_channels: df_main_page = df_main_page[df_main_page['SaleChannel'].isin(st.session_state.selected_channels)]
    if st.session_state.current_view == "Individual Rep" and st.session_state.selected_rep:
        df_main_page = df_main_page[df_main_page['SalesRep'] == st.session_state.selected_rep]

    if df_main_page.empty: 
        st.warning("No data matches the current filter criteria. Please adjust your selections in the sidebar.")
        st.stop()

    # --- KPIs Section ---
    st.header("Product Sales Analysis") 
    total_sales_transactions = len(df_main_page)
    unique_customers_filtered = df_main_page['CustomerID'].nunique() 
    total_sales_amount_actual = df_main_page['SaleAmount'].sum()

    current_month_loop = st.session_state.selected_start_date.replace(day=1)
    end_month_loop = st.session_state.selected_end_date.replace(day=1)
    selected_period_months_num = []
    num_months_in_period = 0
    while current_month_loop <= end_month_loop:
        selected_period_months_num.append(current_month_loop.strftime('%Y-%m'))
        current_month_loop += relativedelta(months=1)
        num_months_in_period +=1
    if num_months_in_period == 0 and st.session_state.selected_start_date.strftime('%Y-%m') == st.session_state.selected_end_date.strftime('%Y-%m'):
        num_months_in_period = 1
        if st.session_state.selected_start_date.strftime('%Y-%m') not in selected_period_months_num:
                selected_period_months_num.append(st.session_state.selected_start_date.strftime('%Y-%m'))

    individual_target_fraction = 1.0
    if st.session_state.current_view == "Individual Rep" and st.session_state.sales_reps_list:
        individual_target_fraction = 1 / len(st.session_state.sales_reps_list) 
    elif st.session_state.current_view == "Individual Rep" and not st.session_state.sales_reps_list:
        individual_target_fraction = 0 

    team_total_sales_target_period = targets_df[targets_df['SaleYearMonthNum'].isin(selected_period_months_num)]['TargetAmount'].sum()
    current_view_total_target_period = team_total_sales_target_period * individual_target_fraction
    average_monthly_target_for_period = (current_view_total_target_period / num_months_in_period) if num_months_in_period > 0 else 0
    achievement_percentage = (total_sales_amount_actual / current_view_total_target_period * 100) if current_view_total_target_period > 0 else 0
    delta_vs_target = total_sales_amount_actual - current_view_total_target_period

    kpi_cols = st.columns(4)
    with kpi_cols[0]: st.metric("Transactions", f"{total_sales_transactions:,}", help="Total sales transactions")
    with kpi_cols[1]: st.metric("Customers", f"{unique_customers_filtered:,}", help="Unique customers with deals")
    with kpi_cols[2]: st.metric("Revenue", f"${total_sales_amount_actual:,.0f}", help="Total sales revenue")
    delta_text = f"{delta_vs_target:,.0f}" if abs(delta_vs_target) >= 1000 else f"{delta_vs_target:,.2f}"
    delta_color = "normal" if delta_vs_target >= 0 else "inverse"
    with kpi_cols[3]: st.metric("vs Target", f"{achievement_percentage:.0f}%", 
                                delta=f"${delta_text} {'▲' if delta_vs_target >=0 else '▼'}", 
                                delta_color=delta_color,
                                help="Revenue compared to target")
    st.caption(f"Period target: ${current_view_total_target_period:,.0f} | Avg monthly: ${average_monthly_target_for_period:,.0f}")

    # --- Sales and Product Performance Section (Integrated into Main Page) ---
    col_sales1, col_sales2 = st.columns(2)

    compact_chart_height = 180  
    compact_chart_margin = dict(l=5, r=5, t=5, b=5) 
    compact_legend = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)

    with col_sales1:
        st.subheader("Sales Trend by Month (vs. Target)") 
        monthly_sales_df = df_main_page.groupby(['SaleYearMonthNum', 'SaleMonth'])['SaleAmount'].sum().reset_index()
        if not monthly_sales_df.empty:
            monthly_sales_df = monthly_sales_df.sort_values('SaleYearMonthNum')
            monthly_targets_for_view_df = targets_df[['SaleYearMonthNum', 'TargetAmount']].copy()
            if st.session_state.current_view == "Individual Rep":
                monthly_targets_for_view_df['TargetAmount'] *= individual_target_fraction
            monthly_sales_vs_target_df = pd.merge(monthly_sales_df, monthly_targets_for_view_df, on='SaleYearMonthNum', how='left')
            monthly_sales_vs_target_df['TargetAmount'] = monthly_sales_vs_target_df['TargetAmount'].fillna(0)
            fig_sales_trend = go.Figure()
            fig_sales_trend.add_trace(go.Scatter(x=monthly_sales_vs_target_df['SaleMonth'], y=monthly_sales_vs_target_df['SaleAmount'], mode='lines+markers', name='Actual Sales', line=dict(color='royalblue', width=2), marker=dict(size=5))) 
            fig_sales_trend.add_trace(go.Scatter(x=monthly_sales_vs_target_df['SaleMonth'], y=monthly_sales_vs_target_df['TargetAmount'], mode='lines+markers', name='Sales Target', line=dict(color='firebrick', width=2, dash='dash'), marker=dict(symbol='star', size=6))) 
            fig_sales_trend.update_layout(title_text=None, xaxis_title=None, yaxis_title="Amount ($)", legend_title_text=None, 
                                          legend=compact_legend, hovermode="x unified", 
                                          height=compact_chart_height, margin=compact_chart_margin)
            st.plotly_chart(fig_sales_trend, use_container_width=True)
        else: st.info("No data for 'Sales Trend by Month' with current filters.")

        st.subheader("Sales by Channel") 
        channel_sales_df = df_main_page.groupby('SaleChannel')['SaleAmount'].sum().reset_index()
        if not channel_sales_df.empty:
            fig_channel_sales = px.pie(channel_sales_df, values='SaleAmount', names='SaleChannel', title=None, color_discrete_sequence=px.colors.qualitative.Pastel)
            fig_channel_sales.update_traces(textposition='inside', textinfo='percent+label')
            fig_channel_sales.update_layout(height=compact_chart_height, legend_title_text=None, margin=compact_chart_margin, showlegend=False) 
            st.plotly_chart(fig_channel_sales, use_container_width=True)
        else: st.info("No data for 'Sales by Channel' with current filters.")

    with col_sales2:
        st.subheader("Top Product Sales") 
        product_sales_agg = df_main_page.groupby('Product')['SaleAmount'].sum().nlargest(7).reset_index()
        if not product_sales_agg.empty:
            fig_product_sales = px.bar(product_sales_agg, x='Product', y='SaleAmount', title=None, color='Product', 
                                       labels={'Product': 'Product', 'SaleAmount': 'Total Sales ($)'}) 
            fig_product_sales.update_layout(xaxis_title=None, yaxis_title="Sales ($)", showlegend=False, 
                                            height=compact_chart_height, margin=compact_chart_margin)
            if average_monthly_target_for_period > 0:
                fig_product_sales.add_hline(y=average_monthly_target_for_period, line_dash="dot", annotation_text=f"Avg M. Target", annotation_position="bottom right", line_color="grey")
            st.plotly_chart(fig_product_sales, use_container_width=True)
        else: st.info("No data for 'Top Product Sales' with current filters.")

        st.subheader("Sales by Region") 
        country_sales_agg = df_main_page.groupby('Country')['SaleAmount'].sum().nlargest(7).reset_index()
        if not country_sales_agg.empty:
            fig_region_sales = px.bar(country_sales_agg, x='Country', y='SaleAmount', title=None, color='Country', 
                                      labels={'Country': 'Country', 'SaleAmount': 'Total Sales ($)'})
            fig_region_sales.update_layout(xaxis_title=None, yaxis_title="Sales ($)", showlegend=False, 
                                           height=compact_chart_height, margin=compact_chart_margin)
            if average_monthly_target_for_period > 0:
                fig_region_sales.add_hline(y=average_monthly_target_for_period, line_dash="dot", annotation_text=f"Avg M. Target", annotation_position="bottom right", line_color="grey")
            st.plotly_chart(fig_region_sales, use_container_width=True)
        else: st.info("No data for 'Sales by Region' with current filters.")
