import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import plotly.express as px
import plotly.figure_factory as ff # For confusion matrix

st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

st.title("🔮 Customer Churn Prediction (Simulated)")
st.markdown("""
This page demonstrates a simple model to predict customer churn.
- **Churn Definition:** A customer is considered 'churned' if they made purchases in Jan-Mar 2023 but NOT in Apr-Jun 2023.
- **Model:** Logistic Regression.
- **Features:** Based on customer activity in Jan-Mar 2023 and their static attributes.
""")

# --- Helper: Define Churn Period ---
period1_end_date = datetime(2023, 3, 31)
period2_start_date = datetime(2023, 4, 1)

@st.cache_data # Cache the feature engineering and model training
def prepare_and_train_churn_model(_df_full, _customers_df):
    """
    Prepares data, engineers features, and trains a churn model.
    _df_full should be the original, unfiltered transaction dataframe.
    _customers_df should be the original, unfiltered customer dataframe.
    """
    
    df_period1 = _df_full[_df_full['SaleDate'] <= period1_end_date].copy()
    customers_in_period1 = df_period1['CustomerID'].unique()

    if len(customers_in_period1) == 0:
        return None, "No customer activity in the first period (Jan-Mar 2023) to build a churn model."

    features_list = []
    for cust_id in customers_in_period1:
        cust_data_p1 = df_period1[df_period1['CustomerID'] == cust_id]
        recency_p1 = (period1_end_date - cust_data_p1['SaleDate'].max()).days
        frequency_p1 = cust_data_p1['TransactionID'].nunique()
        monetary_p1 = cust_data_p1['SaleAmount'].sum()
        avg_transaction_value_p1 = cust_data_p1['SaleAmount'].mean()
        unique_products_p1 = cust_data_p1['Product'].nunique()
        avg_license_length_p1 = cust_data_p1['LicenseLength'].mean() if 'LicenseLength' in cust_data_p1.columns and not cust_data_p1['LicenseLength'].empty else 0
        
        features_list.append({
            'CustomerID': cust_id, 'Recency_P1': recency_p1, 'Frequency_P1': frequency_p1,
            'Monetary_P1': monetary_p1, 'AvgTransactionValue_P1': avg_transaction_value_p1,
            'UniqueProducts_P1': unique_products_p1, 'AvgLicenseLength_P1': avg_license_length_p1,
        })
    
    if not features_list:
        return None, "Could not engineer features for customers in Period 1."
    df_features = pd.DataFrame(features_list)

    df_period2 = _df_full[_df_full['SaleDate'] >= period2_start_date].copy()
    customers_in_period2 = df_period2['CustomerID'].unique()
    df_features['Churned'] = df_features['CustomerID'].apply(lambda x: 0 if x in customers_in_period2 else 1)

    df_features = pd.merge(df_features, _customers_df[['CustomerID', 'Industry', 'Country', 'EmployeeCount']], on='CustomerID', how='left')
    df_features = df_features.fillna({
        'AvgLicenseLength_P1': 0, 'Industry': 'Unknown', 
        'Country': 'Unknown', 'EmployeeCount': 'Unknown'
    })

    X = df_features.drop(['CustomerID', 'Churned'], axis=1)
    y = df_features['Churned']

    if len(X) < 10 or y.nunique() < 2 :
        return None, f"Not enough diverse data for model training. Found {len(X)} samples and {y.nunique()} unique target classes."

    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ], remainder='passthrough')

    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced'))
    ])

    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    except ValueError:
         return None, "Not enough samples in one of the churn classes to perform a stratified split for model training."

    model_pipeline.fit(X_train, y_train)
    y_pred = model_pipeline.predict(X_test)
    y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]

    feature_names_out = [] 
    try:
        raw_feature_names = model_pipeline.named_steps['preprocessor'].get_feature_names_out()
        if isinstance(raw_feature_names, np.ndarray):
            feature_names_out = raw_feature_names.tolist() 
        elif isinstance(raw_feature_names, list): 
            feature_names_out = raw_feature_names
    except Exception: 
        try:
            ohe_categories = model_pipeline.named_steps['preprocessor'].transformers_[1][1].categories_
            new_categorical_features = []
            for i, cat_col in enumerate(categorical_features):
                new_categorical_features.extend([f"{cat_col}_{val}" for val in ohe_categories[i]])
            feature_names_out = numerical_features + new_categorical_features 
        except Exception: 
            if hasattr(model_pipeline.named_steps['classifier'], 'coef_') and \
               model_pipeline.named_steps['classifier'].coef_ is not None and \
               len(model_pipeline.named_steps['classifier'].coef_) > 0:
                 feature_names_out = ["Feature_" + str(i) for i in range(len(model_pipeline.named_steps['classifier'].coef_[0]))]
            else: 
                feature_names_out = [] 
    
    current_coefficients = model_pipeline.named_steps['classifier'].coef_[0] if hasattr(model_pipeline.named_steps['classifier'], 'coef_') else np.array([])


    results = {
        'model': model_pipeline,
        'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test,
        'y_pred': y_pred, 'y_pred_proba': y_pred_proba,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0.5,
        'feature_names': feature_names_out, 
        'coefficients': current_coefficients 
    }
    return results, None


# --- Retrieve data from session state (set by the main app) ---
if 'full_df_original' not in st.session_state or \
   'customers_df_original' not in st.session_state:
    st.warning("Original dataset not found in session state. Please run the main app (Overview page) first.")
    st.stop()

full_df_for_model = st.session_state.full_df_original
customers_df_for_model = st.session_state.customers_df_original

# --- Train Model and Display Results ---
model_results, error_message = prepare_and_train_churn_model(full_df_for_model, customers_df_for_model)

if error_message:
    st.error(f"Model Training Error: {error_message}")
    st.stop()

if model_results:
    st.header("Model Performance Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{model_results['accuracy']:.2f}")
    col1.metric("ROC AUC", f"{model_results['roc_auc']:.2f}")
    col2.metric("Precision (Churn)", f"{model_results['precision']:.2f}")
    col2.metric("Recall (Churn)", f"{model_results['recall']:.2f}")
    col3.metric("F1-Score (Churn)", f"{model_results['f1']:.2f}")
    
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(model_results['y_test'], model_results['y_pred'])
    z_text = [[str(y) for y in x] for x in cm]
    fig_cm = ff.create_annotated_heatmap(
        z=cm, x=['Predicted Not Churn', 'Predicted Churn'], 
        y=['Actual Not Churn', 'Actual Churn'], annotation_text=z_text, colorscale='Blues')
    fig_cm.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig_cm, use_container_width=True)

    st.header("Feature Importance")
    st.markdown("Importance is based on the absolute magnitude of the logistic regression coefficients.")
    
    feature_names_list = model_results.get('feature_names')
    coefficients_array = model_results.get('coefficients')

    if isinstance(feature_names_list, list) and \
       len(feature_names_list) > 0 and \
       isinstance(coefficients_array, np.ndarray) and \
       coefficients_array.size > 0 and \
       len(feature_names_list) == coefficients_array.size:
        
        importance_df = pd.DataFrame({
            'Feature': feature_names_list,
            'Importance': np.abs(coefficients_array)
        }).sort_values(by='Importance', ascending=False).head(15)

        fig_importance = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                                title="Top 15 Most Important Features", height=400) 
        fig_importance.update_layout(yaxis={'categoryorder':'total ascending'}, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_importance, use_container_width=True)
    else:
        st.warning("Could not display feature importance. Data might be insufficient or names/coefficients mismatched.")

    st.header("Sample Churn Predictions (from Test Set)")
    if isinstance(model_results['X_test'], pd.DataFrame):
        X_test_df = model_results['X_test'] # This is a DataFrame with original indices
        y_test_series = model_results['y_test'] # This is a Series with original indices
        y_pred_array = model_results['y_pred'] # Numpy array, 0-indexed based on X_test order
        y_pred_proba_array = model_results['y_pred_proba'] # Numpy array, 0-indexed

        num_samples_to_show = 10
        if len(X_test_df) >= num_samples_to_show:
            # Get the first N original indices from X_test
            sample_original_indices = X_test_df.head(num_samples_to_show).index
            
            # Create the display DataFrame using these sample original indices from X_test
            # Select only a few key features for brevity in the display table
            features_for_display = ['Recency_P1', 'Frequency_P1', 'Monetary_P1']
            # Ensure these features actually exist in X_test_df to avoid KeyErrors
            existing_features_for_display = [f for f in features_for_display if f in X_test_df.columns]
            display_df = X_test_df.loc[sample_original_indices, existing_features_for_display].copy()
            
            # Add Actual_Churn using the same original indices from y_test_series
            display_df['Actual_Churn'] = y_test_series.loc[sample_original_indices]
            
            # To get the corresponding predictions, we need the integer positions (iloc)
            # of these sample_original_indices within the full X_test DataFrame.
            # This gives us the correct indices for the 0-indexed y_pred_array and y_pred_proba_array.
            integer_positions_for_sample = [X_test_df.index.get_loc(idx) for idx in sample_original_indices]

            display_df['Predicted_Churn_Probability'] = y_pred_proba_array[integer_positions_for_sample]
            display_df['Predicted_Churn_Label'] = y_pred_array[integer_positions_for_sample]
            
            # Define final columns to show, ensuring they exist
            cols_to_show_final = existing_features_for_display + ['Actual_Churn', 'Predicted_Churn_Probability', 'Predicted_Churn_Label']
            cols_to_show_final = [col for col in cols_to_show_final if col in display_df.columns]


            st.dataframe(
                display_df[cols_to_show_final]
                .sort_values(by='Predicted_Churn_Probability', ascending=False),
                column_config={
                    "Predicted_Churn_Probability": st.column_config.ProgressColumn(
                        "Churn Probability", format="%.2f", min_value=0, max_value=1,
                    ),
                },
                hide_index=True # Hide the original complex index for cleaner display
            )
            st.caption("Showing a sample from the test set, sorted by highest predicted churn probability.")
        else:
            st.info("Test set is smaller than the number of samples requested for display.")
    else:
        st.info("Test set data for sample predictions is not in the expected DataFrame format.")

else:
    st.error("Churn model could not be trained or results are unavailable.")

st.markdown("---")
st.info("This is a simplified churn model for demonstration. Real-world churn prediction involves more complex feature engineering, model selection, and validation.")

