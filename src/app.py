import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from datetime import datetime
from io import BytesIO
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utility import *
import json
import os
from pathlib import Path
import joblib

# File history management functions
def get_history_file_path():
    """Get the path for the history JSON file"""
    history_dir = Path("history")
    history_dir.mkdir(exist_ok=True)
    return history_dir / "upload_history.json"

def load_upload_history() -> List[Dict]:
    """Load upload history from JSON file"""
    history_file = get_history_file_path()
    if history_file.exists():
        try:
            with open(history_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            st.warning(f"Error loading history: {e}")
            return []
    return []

def save_upload_history(history: List[Dict]):
    """Save upload history to JSON file"""
    history_file = get_history_file_path()
    try:
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        st.error(f"Error saving history: {e}")

def add_to_upload_history(filename: str, file_size: int = None):
    """Add a new file to upload history, maintaining max 10 entries"""
    history = load_upload_history()
    
    # Create new entry
    new_entry = {
        "filename": filename,
        "upload_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "file_size": file_size,
        "status": "Processed"
    }
    
    # Add to beginning of list
    history.insert(0, new_entry)
    
    # Keep only last 10 entries
    history = history[:10]
    
    # Save updated history
    save_upload_history(history)
    
    return history

def display_upload_history():
    """Display the upload history in the sidebar"""
    history = load_upload_history()
    
    if history:
        st.sidebar.markdown("---")
        st.sidebar.subheader("📁 Recent Upload History")
        st.sidebar.markdown("*Last 10 uploaded files*")
        
        for idx, entry in enumerate(history, 1):
            with st.sidebar.expander(f"{idx}. {entry['filename']}", expanded=False):
                st.write(f"**Uploaded:** {entry['upload_time']}")
                if entry.get('file_size'):
                    size_mb = entry['file_size'] / (1024 * 1024)
                    st.write(f"**Size:** {size_mb:.2f} MB")
                st.write(f"**Status:** {entry['status']}")
    else:
        st.sidebar.markdown("---")
        st.sidebar.info("No upload history available")

def check_master_dataset_exists():
    """Check if the master dataset (kunmings.pkl) exists"""
    master_file_path = Path("src/kunmings.pkl")
    return master_file_path.exists()

def load_master_dataset():
    """Load the master dataset if it exists"""
    try:
        if check_master_dataset_exists():
            df=load_data('kunmings.pkl')
            if 'Product Id' in df.columns:
                df['Product Id'] = df['Product Id'].astype(str)
            return df
        else:
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading master dataset: {str(e)}")
        return pd.DataFrame()

def load_data(file):
    # Handle different input types
    if isinstance(file, str):
        # String file path (for database files)
        file_type = file.split('.')[-1]
        if file_type == 'csv':
            df=pd.read_csv(file)
            if 'Product Id' in df.columns:
                df['Product Id'] = df['Product Id'].astype(str)
            return df
        elif file_type == 'pkl':
            df = pd.read_pickle(f"src/{file}")
            if 'Product Id' in df.columns:
                df['Product Id'] = df['Product Id'].astype(str)
            return df
        elif file_type in ['xlsx', 'xls']:
            df = pd.read_excel(file, sheet_name=None)
            df_dict = {}
            for sheet_name, df_ in df.items():
                if 'Product Id' in df_.columns:
                    df_['Product Id'] = df_['Product Id'].astype(str)
                df_dict[sheet_name] = df_
            # st.info(df.keys())
            return df_dict
            
            
            
    else:
        # File object from Streamlit uploader
        if hasattr(file, 'name'):
            file_type = file.name.split('.')[-1]
        else:
            file_type = 'xlsx'  # Default assumption for uploaded files
        
        if file_type in ['xlsx', 'xls']:
            df = pd.read_excel(file, sheet_name=None)
            df_dict = {}
            for sheet_name, df_ in df.items():
                df_dict[sheet_name] = df_
            # st.info(df.keys())
            return df_dict
        elif file_type == 'pkl':
            df = pd.read_pickle(f"src/{file}")
            return df
        elif file_type == 'csv':
            return pd.read_csv(file)


def save_data(df):
    # Create src directory if it doesn't exist
    Path("src").mkdir(exist_ok=True)
    df.to_pickle('src/kunmings.pkl')
    
def create_color_key(df,color_map):
    df['Color Key'] = df.Color.map(lambda x: color_map[x] if x in color_map else '')
    return df
def create_bucket(df,stock_bucket=stock_bucket):
    """
    df : Monthly Stock Data Sheet
    stock_bucket : Dictionary containing bucket ranges
    """
    for key , values in stock_bucket.items():
        lower_bound , upper_bound = values
        index = df[(df['Weight']>=lower_bound) & (df['Weight']<upper_bound)].index.tolist()
        df.loc[index,'Buckets'] = key
    return df

def calculate_avg(df):
    """
    df : Monthly Stock Data Sheet
    """
    df['Avg Cost Total'] = df['Weight'] * df['Average\nCost\n(USD)']
    return df

def create_date_join(df):
    """
    df : Monthly Stock Data Sheet
    """
    df['Month'] = pd.to_datetime('today').month_name()
    df['Year'] = pd.to_datetime('today').year
    df['Join'] = df['Month'].astype(str) + '-' + df['Year'].map(lambda x: x-2000).astype(str)
    return df
def concatenate_first_two_rows(df):
    result = {}
    for col in df.columns:
        value1 = str(df.iloc[0][col])
        value2 = str(df.iloc[1][col])
        result[col] = f"{value1}_{value2}"
    return result
def populate_max_qty(df,MONTHLY_STOCK_DATA):
    """
    df : Max Qty Sheet
    MONTHLY_STOCK_DATA : Monthly Stock Data Sheet
    """
    columns=list(concatenate_first_two_rows(df.iloc[0:2,2:]).values())
    columns = ['Months','Buckets'] + columns
    df.columns = columns
    df=df.iloc[2:,:]
    df.reset_index(drop=True,inplace=True)
    _MAX_QTY_ = []
    MONTHLY_STOCK_DATA['Max Qty'] = None
    for indx, row in MONTHLY_STOCK_DATA.iterrows():
        join = row['Join']
        Shape = row['Shape key']
        Color = row['Color Key']
        Bucket = row['Buckets']
        if pd.isna(Color):
            value = None
        else:
            col_name = f"{Shape}_{Color}"
            if col_name in df.columns.tolist():
                value = df[(df['Months'] == join) & (df['Buckets'] == Bucket)][col_name].values.tolist()
            else:
                value = 0
        _MAX_QTY_.append(value)
    MONTHLY_STOCK_DATA['Max Qty'] = _MAX_QTY_
    MONTHLY_STOCK_DATA['Max Qty']=MONTHLY_STOCK_DATA['Max Qty'].map(lambda x:x[0] if isinstance(x, list) and len(x) > 0 else 0)
    return MONTHLY_STOCK_DATA

def populate_min_qty(df,MONTHLY_STOCK_DATA):
    """
    df : Buying Min Qty Sheet
    MONTHLY_STOCK_DATA : Monthly Stock Data Sheet
    """
    columns=list(concatenate_first_two_rows(df.iloc[0:2,2:]).values())
    columns = ['Months','Buckets'] + columns
    df.columns = columns
    df=df.iloc[2:,:]
    df.reset_index(drop=True,inplace=True)
    _MIN_QTY_ = []
    MONTHLY_STOCK_DATA['Min Qty'] = None
    for _, row in MONTHLY_STOCK_DATA.iterrows():
        join = row['Join']
        Shape = row['Shape key']
        Color = row['Color Key']
        Bucket = row['Buckets']
        if pd.isna(Color):
            value = None
        else:
            col_name = f"{Shape}_{Color}"
            if col_name in df.columns.tolist():
                value = df[(df['Months'] == join) & (df['Buckets'] == Bucket)][col_name].values.tolist()
            else:
                value = 0
        _MIN_QTY_.append(value)
    MONTHLY_STOCK_DATA['Min Qty'] = _MIN_QTY_
    MONTHLY_STOCK_DATA['Min Qty']=MONTHLY_STOCK_DATA['Min Qty'].map(lambda x:x[0] if isinstance(x, list) and len(x) > 0 else 0)
    return MONTHLY_STOCK_DATA
def populate_selling_prices(df,MONTHLY_STOCK_DATA):
    """
    df : Buying Max Prices Sheet 
    MONTHLY_STOCK_DATA : Monthly Stock Data Sheet
    """
    columns=list(concatenate_first_two_rows(df.iloc[0:2,1:]).values())
    columns = ['Buckets'] + columns
    df.columns = columns
    df=df.iloc[2:,:]
    df.reset_index(drop=True,inplace=True)
    _SELLING_PRICE_ = []
    MONTHLY_STOCK_DATA['Min Selling Price'] = None
    for indx, row in MONTHLY_STOCK_DATA.iterrows():
        join = row['Join']
        Shape = row['Shape key']
        Color = row['Color Key']
        Bucket = row['Buckets']
        if pd.isna(Color):
            value = None
        else:
            col_name = f"{Shape}_{Color}"
            if col_name in df.columns.tolist():
                value = df[(df['Buckets'] == Bucket)][col_name].values.tolist()
            else:
                value = 0
        _SELLING_PRICE_.append(value)
    MONTHLY_STOCK_DATA['Min Selling Price'] = _SELLING_PRICE_
    MONTHLY_STOCK_DATA['Min Selling Price']=MONTHLY_STOCK_DATA['Min Selling Price'].map(lambda x: x[0] if (isinstance(x,list) and len(x)>0) else 0)
    MONTHLY_STOCK_DATA['Min Selling Price'] = MONTHLY_STOCK_DATA['Max Buying Price'] * MONTHLY_STOCK_DATA['Min Selling Price'] 
    return MONTHLY_STOCK_DATA
def populate_buying_prices(df,MONTHLY_STOCK_DATA):
    """
    df : Buying Max Prices Sheet 
    MONTHLY_STOCK_DATA : Monthly Stock Data Sheet
    """
    columns=list(concatenate_first_two_rows(df.iloc[0:2,2:]).values())
    columns = ['Months','Buckets'] + columns
    df.columns = columns
    df=df.iloc[2:,:]
    df.reset_index(drop=True,inplace=True)
    _BUYING_PRICE_ = []
    MONTHLY_STOCK_DATA['Max Buying Price'] = None
    for indx, row in MONTHLY_STOCK_DATA.iterrows():
        join = row['Join']
        Shape = row['Shape key']
        Color = row['Color Key']
        Bucket = row['Buckets']
        if pd.isna(Color):
            value = None
        else:
            col_name = f"{Shape}_{Color}"
            if col_name in df.columns.tolist():
                value = df[(df['Months'] == join) & (df['Buckets'] == Bucket)][col_name].values.tolist()
            else:
                value = 0
        _BUYING_PRICE_.append(value)
    MONTHLY_STOCK_DATA['Max Buying Price'] = _BUYING_PRICE_
    MONTHLY_STOCK_DATA['Max Buying Price']=MONTHLY_STOCK_DATA['Max Buying Price'].map(lambda x:x[0] if isinstance(x, list) and len(x) > 0 else 0)
    return MONTHLY_STOCK_DATA
def calculate_buying_price_avg(df):
    df['Buying Price Avg'] = df['Max Buying Price'] * df['Weight']
    return df

def get_quarter(month):
    Quarter_Month_Map = {
    'Q1': ['January', 'February', 'March'],
    'Q2': ['April', 'May', 'June'],
    'Q3': ['July', 'August', 'September'],
    'Q4': ['October', 'November', 'December']
    }
    year = pd.to_datetime('today').year
    yr = year - 2000

    if month in Quarter_Month_Map['Q1']:
        return f'Q1-{yr}'
    elif month in Quarter_Month_Map['Q2']:
        return f'Q2-{yr}'
    elif month in Quarter_Month_Map['Q3']:
        return f'Q3-{yr}'
    elif month in Quarter_Month_Map['Q4']:
        return f'Q4-{yr}'
    else:
        return None
def populate_quarter(df):
    """
    df : Monthly Stock Data Sheet
    """
    df['Quarter'] = df['Month'].apply(get_quarter)
    return df
def create_shape_key(x):
    if x.__contains__(r'HEART'):
        return 'Other'
    elif x.__contains__(r'CUSHION'):
        return 'Cushion'
    elif x.__contains__(r'OVAL'):
        return 'Oval'
    elif x.__contains__(r'PEAR'):
        return 'Pear'
    elif x.__contains__(r'CUT-CORNERED'):
        return 'Radiant'
    elif x.__contains__(r'MODIFIED RECTANGULAR'):
        return 'Cushion'
    elif x.__contains__(r'MODIFIED SQUARE'):
        return 'Cushion'
    
    elif x.__contains__(r'MARQUISE MODIFIED'):
        return 'Other'
    elif x.__contains__(r'ROUND_CORNERED'):
        return 'Cushion'
    elif x.__contains__(r'EMERALD'):
        return 'Other'
    else:
        return 'Other'
def update_max_qty(df_max_qty,json_data_name = 'max_qty.pkl'): 
    """
    Update the max quantity for a specific month, bucket, shape, and color.
    """
    try:
        json_data_name = 'src/'+json_data_name
        json_data = joblib.load(json_data_name)
    except:
        json_data = {}
    columns=list(concatenate_first_two_rows(df_max_qty.iloc[0:2,2:]).values())
    columns = ['Months','Buckets'] + columns
    df_max_qty.columns = columns
    df_max_qty=df_max_qty.iloc[2:,:]
    json_data = {}
    for col in df_max_qty.columns[2:]:
        json_data[col] = {}
        for month in df_max_qty['Months'].unique():
            json_data[col][month] = {}
            for bucket in df_max_qty['Buckets'].unique():
                json_data[col][month][bucket] = df_max_qty[(df_max_qty['Months'] == month) & (df_max_qty['Buckets']==bucket)][col].iloc[0]
    joblib.dump(json_data, json_data_name)
def poplutate_monthly_stock_sheet(file):
    """
    df_stock : Monthly Stock Data Sheet
    df_buying : Buying Max Prices Sheet
    df_min_qty : Buying Min Qty Sheet
    df_max_qty : Max Qty Sheet
    """
    df = load_data(file)
    
    df_stock = df['Monthly Stock Data']
    df_stock.rename(columns={'avg': 'Avg Cost Total'}, inplace=True)
    df_buying = df['Buying Max Prices']
    df_min_qty = df['MIN Data']
    update_max_qty(df_min_qty, json_data_name='min_qty.pkl')
    df_max_qty = df['MAX Data']
    update_max_qty(df_max_qty, json_data_name='max_qty.pkl')
    df_min_sp = df['Min Selling Price']
    if df_stock.empty or df_buying.empty or df_min_qty.empty or df_max_qty.empty:
        raise ValueError("One or more dataframes are empty. Please check the input files.")
    df_stock = create_date_join(df_stock)
    df_stock = populate_quarter(df_stock)
    df_stock = calculate_avg(df_stock)
    df_stock = create_bucket(df_stock)
    df_stock = create_color_key(df_stock, color_map)
    df_stock['Shape key'] = df_stock['Shape'].apply(create_shape_key)
    df_stock = populate_max_qty(df_max_qty, df_stock)
    df_stock = populate_min_qty(df_min_qty, df_stock)
    df_stock = populate_buying_prices(df_buying, df_stock)
    df_stock = calculate_buying_price_avg(df_stock)
    df_stock = populate_selling_prices(df_min_sp,df_stock)
    df_stock.fillna(0,inplace=True)
    cols = df_stock.columns.tolist()
    df_stock = df_stock.groupby(['Product Id','Year', 'Month']).first().reset_index().loc[:,cols]
    return df_stock
def calculate_qoq_variance_percentage(current_quarter_price, previous_quarter_price):
    """
    Calculate quarter-on-quarter variance percentage of price.
    
    Args:
        current_quarter_price (float): Price for the current quarter
        previous_quarter_price (float): Price for the previous quarter
    
    Returns:
        float: Variance percentage (positive for increase, negative for decrease)
        
    Raises:
        ValueError: If previous quarter price is zero or negative
        TypeError: If inputs are not numeric
    """
    # Input validation
    if not isinstance(current_quarter_price, (int, float)) or not isinstance(previous_quarter_price, (int, float)):
        raise TypeError("Both prices must be numeric values")
    
    if previous_quarter_price <= 0:
        variance_percentage = 0.00001
        # raise ValueError("Previous quarter price must be positive (cannot be zero or negative)")
    
    # Calculate variance percentage with ZeroDivisionError handling
    try:
        if previous_quarter_price != 0:
            variance_percentage = ((current_quarter_price - previous_quarter_price) / previous_quarter_price) * 100
        else:
            total = previous_quarter_price + current_quarter_price
            if total != 0:
                variance_percentage = ((current_quarter_price - previous_quarter_price) / total) * 100
            else:
                variance_percentage = 0
    except ZeroDivisionError:
        variance_percentage = 0
        
    return round(variance_percentage, 2)


def calculate_qoq_variance_series(price_data):
    """
    Calculate quarter-on-quarter variance for a series of quarterly prices.
    
    Args:
        price_data (list): List of quarterly prices in chronological order
    
    Returns:
        list: List of QoQ variance percentages (starts from Q2 since Q1 has no previous quarter)
    """
    if len(price_data) < 2:
        raise ValueError("Need at least 2 quarters of data to calculate variance")
    
    variances = []
    for i in range(1, len(price_data)):
        variance = calculate_qoq_variance_percentage(price_data[i], price_data[i-1])
        variances.append(variance)
    
    return variances
def monthly_variance(df,col):
    analysis=df.groupby(['Month','Year'],as_index=False)[col].sum()
    analysis['Num_Month'] = analysis['Month'].map(month_map)
    analysis.sort_values(by=['Year','Num_Month'],inplace=True)
    analysis['Monthly_change']=analysis[col].pct_change().fillna(0).round(2)*100
    analysis['qaurter_change']=[0]+calculate_qoq_variance_series(analysis[col].tolist())
    return analysis
def gap_analysis(max_qty,min_qty,stock_in_hand):
    """
    max_qty : Maximum Quantity
    min_qty : Minimum Quantity
    stock_in_hand : Stock in Hand
    """
    if stock_in_hand > max_qty:
        excess_qty = stock_in_hand - max_qty
        return excess_qty
    elif stock_in_hand < min_qty:
        deficit_qty = stock_in_hand - min_qty
        return deficit_qty
    else:
        return 0

def get_filtered_data(FILTER_MONTH,FILTE_YEAR,FILTER_SHAPE,FILTER_COLOR,FILTER_BUCKET):
    """
    file : Monthly Stock Data Sheet
    FILTER_MONTH : Month to filter
    FILTE_YEAR : Year to filter
    FILTER_SHAPE : Shape Key to filter
    FILTER_COLOR : Color Key to filter
    FILTER_BUCKET : Buckets to filter
    FILTER_MONTHLY_VAR_COL : Column to calculate monthly variance
    PARENT_DF : Parent DataFrame to concatenate with the monthly stock data
    """
    master_df = load_data('kunmings.pkl')
    cols = master_df.columns.tolist()
    master_df = master_df.groupby(['Product Id','Year', 'Month']).first().reset_index().loc[:,cols]
    if (type(FILTE_YEAR)==str) & (str(FILTE_YEAR).isnumeric()):
        FILTE_YEAR = int(FILTE_YEAR)
    filter_data=master_df[(master_df['Month'] == FILTER_MONTH) & \
                                      (master_df['Year'] == FILTE_YEAR) & \
                                        (master_df['Shape key'] == FILTER_SHAPE) &\
                                        (master_df['Color Key'] == FILTER_COLOR) &\
                                        (master_df['Buckets'] == FILTER_BUCKET)]
    max_qty = filter_data['Max Qty'].max()
    min_qty = filter_data['Min Qty'].min()
    stock_in_hand = filter_data.shape[0]
    if stock_in_hand == 0:
        max_qty_dict = joblib.load('src/max_qty.pkl')
        min_qty_dict = joblib.load('src/min_qty.pkl')
        filter_shape_color = FILTER_SHAPE + '_' + FILTER_COLOR
        latest_month_max_qty_dict = list(max_qty_dict[filter_shape_color].keys())[-1]
        latest_month_min_qty_dict = list(min_qty_dict[filter_shape_color].keys())[-1]
        max_qty = max_qty_dict[filter_shape_color][latest_month_max_qty_dict][FILTER_BUCKET]
        min_qty = min_qty_dict[filter_shape_color][latest_month_max_qty_dict][FILTER_BUCKET]
        
    gap_analysis_op = gap_analysis(max_qty, min_qty, stock_in_hand)
    _filter_ = master_df[(master_df['Shape key'] == FILTER_SHAPE) &\
                                        (master_df['Color Key'] == FILTER_COLOR) &\
                                        (master_df['Buckets'] == FILTER_BUCKET)]
    try:
        max_buying_price = filter_data['Max Buying Price'].max()
        # Handle ZeroDivisionError
        weight_sum = filter_data['Weight'].sum()
        if weight_sum != 0:
            current_avg_cost = (sum(filter_data['Avg Cost Total'])/weight_sum) * 0.9
        else:
            current_avg_cost = 0
        min_selling_price = filter_data['Min Selling Price'].min()
        return [filter_data,int(max_buying_price),int(current_avg_cost), gap_analysis_op,min_selling_price]
    except:
        return [pd.DataFrame(columns=master_df.columns.tolist()),f"There is {filter_data.shape[0]} rows after filter",f"There is {filter_data.shape[0]} rows after filter",gap_analysis_op,0]

def get_summary_metrics(filter_data,Filter_Month,FILTER_SHAPE,FILTE_YEAR,FILTER_COLOR,FILTER_BUCKET,FILTER_MONTHLY_VAR_COL):
    FILTE_YEAR = int(FILTE_YEAR)
    master_df = load_data('kunmings.pkl')
    cols = master_df.columns.tolist()
    master_df = master_df.groupby(['Product Id','Year', 'Month']).first().reset_index().loc[:,cols]
    _filter_ = master_df[(master_df['Shape key'] == FILTER_SHAPE) &\
                                        (master_df['Color Key'] == FILTER_COLOR) &\
                                        (master_df['Buckets'] == FILTER_BUCKET)]
    
    # Calculate previous month considering year transitions
    current_month_num = month_map[Filter_Month]
    Prev_Month_Name = None
    Prev_Year = FILTE_YEAR
    
    if current_month_num == 1:  # January
        # Previous month is December of previous year
        Prev_Month_Name = 'December'
        Prev_Year = FILTE_YEAR - 1
    else:
        # Find the previous month in the same year
        for Month_Name, Month_Num in month_map.items():
            if Month_Num == current_month_num - 1:
                Prev_Month_Name = Month_Name
                break
    
    Prev_filter_data=master_df[(master_df['Month'] == Prev_Month_Name) & \
                                      (master_df['Year'] == Prev_Year) & \
                                        (master_df['Shape key'] == FILTER_SHAPE) &\
                                        (master_df['Color Key'] == FILTER_COLOR) &\
                                        (master_df['Buckets'] == FILTER_BUCKET)]
    try:
        if FILTER_MONTHLY_VAR_COL == 'Current Average Cost':
            FILTER_MONTHLY_VAR_COL='Buying Price Avg'
            avg_value = Prev_filter_data[FILTER_MONTHLY_VAR_COL].mean()
            
            # Calculate current and previous average costs with ZeroDivisionError handling
            weight_sum_current = filter_data['Weight'].sum()
            weight_sum_prev = Prev_filter_data['Weight'].sum()
            
            if weight_sum_current != 0:
                current_avg_cost = (sum(filter_data['Avg Cost Total'])/weight_sum_current) * 0.9
            else:
                current_avg_cost = 0
                
            if weight_sum_prev != 0:
                prev_current_avg_cost = (sum(Prev_filter_data['Avg Cost Total'])/weight_sum_prev) * 0.9
            else:
                prev_current_avg_cost = 0
            
            # Calculate MOM Variance with ZeroDivisionError handling
            if prev_current_avg_cost != 0:
                MOM_Variance = ((current_avg_cost - prev_current_avg_cost) / prev_current_avg_cost) * 100
            else:
                MOM_Variance = 0
                
            var_analysis = monthly_variance(_filter_,FILTER_MONTHLY_VAR_COL)
            MOM_Percent_Change = var_analysis[(var_analysis['Month'] == Filter_Month) & (var_analysis['Year'] == FILTE_YEAR)]['Monthly_change'].values.tolist()[0]
            MOM_QoQ_Percent_Change = var_analysis[(var_analysis['Month'] == Filter_Month) & (var_analysis['Year'] == FILTE_YEAR)]['qaurter_change'].values.tolist()[0]
            if MOM_Percent_Change == np.inf or pd.isna(MOM_Percent_Change) :
                MOM_Percent_Change = 0
            if MOM_QoQ_Percent_Change == np.inf or pd.isna(MOM_QoQ_Percent_Change):
                MOM_QoQ_Percent_Change = 0
            if pd.isna(MOM_Variance) or MOM_Variance == np.inf :
                MOM_Variance = 0
            return [MOM_Variance, MOM_Percent_Change, MOM_QoQ_Percent_Change]
        elif FILTER_MONTHLY_VAR_COL == 'Max Buying Price':
            avg_value = _filter_[FILTER_MONTHLY_VAR_COL].mean()
            # Handle ZeroDivisionError
            if avg_value != 0 and filter_data.shape[0] != 0:
                MOM_Variance = (sum((filter_data[FILTER_MONTHLY_VAR_COL] - avg_value)/ avg_value )/filter_data.shape[0]) * 100
            else:
                MOM_Variance = 0
            var_analysis = monthly_variance(_filter_,FILTER_MONTHLY_VAR_COL)
            MOM_Percent_Change = var_analysis[(var_analysis['Month'] == Filter_Month) & (var_analysis['Year'] == FILTE_YEAR)]['Monthly_change'].values.tolist()[0]
            MOM_QoQ_Percent_Change = var_analysis[(var_analysis['Month'] == Filter_Month) & (var_analysis['Year'] == FILTE_YEAR)]['qaurter_change'].values.tolist()[0]
            if MOM_Percent_Change == np.inf or pd.isna(MOM_Percent_Change) :
                MOM_Percent_Change = 0
            if MOM_QoQ_Percent_Change == np.inf or pd.isna(MOM_QoQ_Percent_Change):
                MOM_QoQ_Percent_Change = 0
            if pd.isna(MOM_Variance) or MOM_Variance == np.inf :
                MOM_Variance = 0
            return [MOM_Variance, MOM_Percent_Change, MOM_QoQ_Percent_Change]
        elif FILTER_MONTHLY_VAR_COL == 'Min Selling Price':
            avg_value = _filter_[FILTER_MONTHLY_VAR_COL].mean()
            # Handle ZeroDivisionError
            if avg_value != 0 and filter_data.shape[0] != 0:
                MOM_Variance = (sum((filter_data[FILTER_MONTHLY_VAR_COL] - avg_value)/ avg_value )/filter_data.shape[0]) * 100
            else:
                MOM_Variance = 0
            var_analysis = monthly_variance(_filter_,FILTER_MONTHLY_VAR_COL)
            MOM_Percent_Change = var_analysis[(var_analysis['Month'] == Filter_Month) & (var_analysis['Year'] == FILTE_YEAR)]['Monthly_change'].values.tolist()[0]
            MOM_QoQ_Percent_Change = var_analysis[(var_analysis['Month'] == Filter_Month) & (var_analysis['Year'] == FILTE_YEAR)]['qaurter_change'].values.tolist()[0]
            if MOM_Percent_Change == np.inf or pd.isna(MOM_Percent_Change) :
                MOM_Percent_Change = 0
            if MOM_QoQ_Percent_Change == np.inf or pd.isna(MOM_QoQ_Percent_Change):
                MOM_QoQ_Percent_Change = 0
            if pd.isna(MOM_Variance) or MOM_Variance == np.inf :
                MOM_Variance = 0
            return [MOM_Variance, MOM_Percent_Change, MOM_QoQ_Percent_Change]
        else:
            return [0,0,0]
            
    except:
        return [0,0,0]
        
    
def get_gap_summary_table(master_df, selected_month, selected_year, selected_shape, selected_color, selected_bucket):
    """
    Generate GAP summary table for all combinations of filter values
    """
    gap_summary = []
    
    # Get unique values for each filter
    months = [selected_month] if selected_month != "None" else list(master_df['Month'].unique())
    years = [selected_year] if selected_year != "None" else list(master_df['Year'].unique())
    shapes = [selected_shape] if selected_shape != "None" else list(master_df['Shape key'].unique())
    colors = [selected_color] if selected_color != "None" else list(master_df['Color Key'].unique())
    buckets = [selected_bucket] if selected_bucket != "None" else list(master_df['Buckets'].unique())
    
    # Generate all combinations
    for month in months:
        for year in years:
            for shape in shapes:
                for color in colors:
                    for bucket in buckets:
                        # Filter data for current combination
                        filtered_data = master_df[
                            (master_df['Month'] == month) & 
                            (master_df['Year'] == year) & 
                            (master_df['Shape key'] == shape) & 
                            (master_df['Color Key'] == color) & 
                            (master_df['Buckets'] == bucket)
                        ]
                        
                        if not filtered_data.empty:
                            max_qty = int(filtered_data['Max Qty'].max())
                            min_qty = int(filtered_data['Min Qty'].min())
                            stock_in_hand = filtered_data.shape[0]
                            gap_value = gap_analysis(max_qty, min_qty, stock_in_hand)
                            
                            gap_summary.append({
                                'Month': month,
                                'Year': year,
                                'Shape': shape,
                                'Color': color,
                                'Bucket': bucket,
                                'Max Qty': max_qty,
                                'Min Qty': min_qty,
                                'Stock in Hand': stock_in_hand,
                                'GAP Value': int(gap_value),
                                'Status': 'Excess' if gap_value > 0 else 'Need' if gap_value < 0 else 'Adequate'
                                })
                        else:
                            max_qty_dict = joblib.load('src/max_qty.pkl')
                            min_qty_dict = joblib.load('src/min_qty.pkl')
                            filter_shape_color = shape + '_' + color
                            latest_month_max_qty_dict = list(max_qty_dict[filter_shape_color].keys())[-1]
                            latest_month_min_qty_dict = list(min_qty_dict[filter_shape_color].keys())[-1]
                            max_qty = max_qty_dict[filter_shape_color][latest_month_max_qty_dict][bucket]
                            min_qty = min_qty_dict[filter_shape_color][latest_month_max_qty_dict][bucket]
                            gap_value = gap_analysis(max_qty, min_qty, 0)
                            gap_summary.append({
                                'Month': month,
                                'Year': year,
                                'Shape': shape,
                                'Color': color,
                                'Bucket': bucket,
                                'Max Qty': max_qty,
                                'Min Qty': min_qty,
                                'Stock in Hand': stock_in_hand,
                                'GAP Value': int(gap_value),
                                'Status': 'Excess' if gap_value > 0 else 'Need' if gap_value < 0 else 'Adequate'
                                })
                            
    
    return pd.DataFrame(gap_summary).sort_values(by=['Shape','Color','Bucket'])

def get_final_data(file,PARENT_DF = 'kunmings.pkl'):
    df = poplutate_monthly_stock_sheet(file)
    parent_df = load_data(PARENT_DF)
    master_df = pd.concat([df, parent_df], ignore_index=True,axis=0)
    # master_df = master_df.drop_duplicates(subset='Product Id')
    cols = master_df.columns.tolist()
    master_df = master_df.groupby(['Product Id','Year', 'Month']).first().reset_index().loc[:,cols]
    save_data(master_df)
    return master_df
def sort_months(months):
    """
    Sort months supporting both full names and abbreviations.
    
    Args:
        months: List of month names (full names or abbreviations)
    
    Returns:
        List of months sorted in chronological order
    """
    import calendar
    
    # Create mapping for both full names and abbreviations
    month_mapping = {}
    
    for i in range(1, 13):
        full_name = calendar.month_name[i]
        abbr_name = calendar.month_abbr[i]
        month_mapping[full_name] = i
        month_mapping[abbr_name] = i
        month_mapping[full_name.lower()] = i
        month_mapping[abbr_name.lower()] = i
    
    # Sort based on month order
    sorted_months = sorted(months, key=lambda month: month_mapping.get(month, 13))
    
    return sorted_months

def create_trend_visualization(master_df, selected_shape=None, selected_color=None, selected_bucket=None, 
                             selected_variance_column=None, selected_month=None, selected_year=None):
    """
    Create trend line visualizations for MOM Variance and MOM QoQ Percent Change
    
    Args:
        master_df: Master dataframe containing all data
        selected_shape: Selected shape filter
        selected_color: Selected color filter  
        selected_bucket: Selected bucket filter
        selected_variance_column: Column to calculate variance for
        selected_month: Selected month filter (for highlighting)
        selected_year: Selected year filter (for highlighting)
    
    Returns:
        plotly figure object
    """
    
    # Filter data based on selections - if all are None, use entire dataset
    if selected_shape is None and selected_color is None and selected_bucket is None:
        filtered_df = master_df
        title_suffix = "All Data"
    else:
        filtered_df = master_df.copy()
        title_parts = []
        
        if selected_shape is not None:
            filtered_df = filtered_df[filtered_df['Shape key'] == selected_shape]
            title_parts.append(selected_shape)
        if selected_color is not None:
            filtered_df = filtered_df[filtered_df['Color Key'] == selected_color]
            title_parts.append(selected_color)
        if selected_bucket is not None:
            filtered_df = filtered_df[filtered_df['Buckets'] == selected_bucket]
            title_parts.append(selected_bucket)
        
        title_suffix = " | ".join(title_parts) if title_parts else "Filtered Data"
    
    if filtered_df.empty:
        # Return empty figure if no data
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for selected filters",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16)
        )
        return fig
    
    # Prepare variance column
    variance_col = selected_variance_column
    if variance_col == 'Current Average Cost':
        variance_col = 'Buying Price Avg'
    elif variance_col == 'None' or variance_col == None:
        variance_col = 'Max Buying Price' # Default column
    
    # Calculate monthly variance data
    try:
        var_analysis = monthly_variance(filtered_df, variance_col)
        
        # Create date column for proper sorting
        var_analysis['Date']='01'+'-'+var_analysis['Num_Month'].astype(str)+'-'+var_analysis['Year'].astype(str)
        var_analysis['Date'] = pd.to_datetime(var_analysis['Date'], format='%d-%m-%Y')
        var_analysis = var_analysis.sort_values('Date')
        
        # Filter data up to selected month/year if specified
        if selected_month is not None and selected_month != "None" and selected_year is not None and selected_year != "None":
            selected_year_int = int(selected_year)
            selected_month_num = month_map.get(selected_month, 0)
            
            # Create a cutoff date
            cutoff_date = pd.to_datetime(f"{selected_year_int}-{selected_month_num:02d}-01")
            var_analysis_filtered = var_analysis[var_analysis['Date'] <= cutoff_date].copy()
            
            # Add highlight column for selected month
            var_analysis_filtered['is_selected'] = ((var_analysis_filtered['Month'] == selected_month) & 
                                                   (var_analysis_filtered['Year'] == selected_year_int))
        else:
            var_analysis_filtered = var_analysis.copy()
            var_analysis_filtered['is_selected'] = False
        
        # Create subplot with secondary y-axis
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Monthly Variance Trend', 'Quarter-over-Quarter Change'),
            vertical_spacing=0.1,
            specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
        )
        
        # Separate selected and non-selected points for Monthly Variance
        non_selected = var_analysis_filtered[~var_analysis_filtered['is_selected']]
        selected = var_analysis_filtered[var_analysis_filtered['is_selected']]
        
        # Add Monthly Variance line
        if not non_selected.empty:
            fig.add_trace(
                go.Scatter(
                    x=non_selected['Date'],
                    y=non_selected['Monthly_change'],
                    mode='lines+markers',
                    name='Monthly Change %',
                    line=dict(color='#1f77b4', width=3),
                    marker=dict(size=8, color='#1f77b4'),
                    hovertemplate='<b>%{x|%b %Y}</b><br>' +
                                 'Monthly Change: %{y:.2f}%<br>' +
                                 '<extra></extra>',
                    showlegend=True
                ),
                row=1, col=1
            )
        
        # Add highlighted point for selected month (Monthly Variance)
        if not selected.empty:
            fig.add_trace(
                go.Scatter(
                    x=selected['Date'],
                    y=selected['Monthly_change'],
                    mode='markers',
                    name='Selected Month',
                    marker=dict(size=15, color='#ff0000', symbol='star'),
                    hovertemplate='<b>%{x|%b %Y} (Selected)</b><br>' +
                                 'Monthly Change: %{y:.2f}%<br>' +
                                 '<extra></extra>',
                    showlegend=True
                ),
                row=1, col=1
            )
        
        # Add QoQ Change line
        if not non_selected.empty:
            fig.add_trace(
                go.Scatter(
                    x=non_selected['Date'],
                    y=non_selected['qaurter_change'],
                    mode='lines+markers',
                    name='QoQ Change %',
                    line=dict(color='#ff7f0e', width=3),
                    marker=dict(size=8, color='#ff7f0e'),
                    hovertemplate='<b>%{x|%b %Y}</b><br>' +
                                 'QoQ Change: %{y:.2f}%<br>' +
                                 '<extra></extra>',
                    showlegend=True
                ),
                row=2, col=1
            )
        
        # Add highlighted point for selected month (QoQ)
        if not selected.empty:
            fig.add_trace(
                go.Scatter(
                    x=selected['Date'],
                    y=selected['qaurter_change'],
                    mode='markers',
                    name='Selected Month (QoQ)',
                    marker=dict(size=15, color='#ff0000', symbol='star'),
                    hovertemplate='<b>%{x|%b %Y} (Selected)</b><br>' +
                                 'QoQ Change: %{y:.2f}%<br>' +
                                 '<extra></extra>',
                    showlegend=False  # Don't show duplicate in legend
                ),
                row=2, col=1
            )
        
        # Add zero reference lines
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.7, row=1, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.7, row=2, col=1)
        
        # Update layout
        title_text = f"Trend Analysis - {title_suffix}"
        if selected_month is not None and selected_month != "None":
            title_text += f" (Data up to {selected_month} {selected_year})"
            
        fig.update_layout(
            title=title_text,
            height=600,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        # Update subplot title colors to orange
        fig.update_annotations(font=dict(color='black', size=16))
        
        # Update x-axis
        fig.update_xaxes(
            title_text="Date",
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            row=2, col=1
        )
        
        # Update y-axes
        fig.update_yaxes(
            title_text="Monthly Change (%)",
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            row=1, col=1
        )
        
        fig.update_yaxes(
            title_text="QoQ Change (%)",
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            row=2, col=1
        )
        
        return fig
        
    except Exception as e:
        # Return error figure
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating visualization: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16)
        )
        return fig

def create_summary_charts(master_df, selected_shape, selected_color, selected_bucket, 
                         selected_month=None, selected_year=None):
    """
    Create summary charts showing overall trends across all months/years
    
    Args:
        master_df: Master dataframe
        selected_shape: Selected shape filter
        selected_color: Selected color filter
        selected_bucket: Selected bucket filter
        selected_month: Selected month filter (for highlighting)
        selected_year: Selected year filter (for highlighting)
    
    Returns:
        plotly figure object
    """
    cols = master_df.columns.tolist()
    master_df = master_df.groupby(['Product Id','Year', 'Month']).first().reset_index().loc[:,cols]
    # Filter data - if all are None, use entire dataset
    if selected_shape is None and selected_color is None and selected_bucket is None:
        filtered_df = master_df
        title_suffix = "All Data"
    else:
        filtered_df = master_df.copy()
        title_parts = []
        
        if selected_shape is not None:
            filtered_df = filtered_df[filtered_df['Shape key'] == selected_shape]
            title_parts.append(selected_shape)
        if selected_color is not None:
            filtered_df = filtered_df[filtered_df['Color Key'] == selected_color]
            title_parts.append(selected_color)
        if selected_bucket is not None:
            filtered_df = filtered_df[filtered_df['Buckets'] == selected_bucket]
            title_parts.append(selected_bucket)
        
        title_suffix = " | ".join(title_parts) if title_parts else "Filtered Data"
    
    if filtered_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for selected filters",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16)
        )
        return fig
    
    # Group by month and year to get summary statistics
    summary_data = filtered_df.groupby(['Month', 'Year']).agg({
        'Avg Cost Total': 'mean',
        'Max Buying Price': 'mean',
        'Weight': 'sum',
        'Product Id': 'count'
    }).reset_index()
    
    
    # Create date column for proper sorting
    summary_data['Num_Month'] = summary_data['Month'].map(month_map)
    summary_data['Date']='01'+'-'+summary_data['Num_Month'].astype(str)+'-'+summary_data['Year'].astype(str)
    summary_data['Date'] = pd.to_datetime(summary_data['Date'], format='%d-%m-%Y')
    summary_data = summary_data.sort_values('Date')
    
    # Filter data up to selected month/year if specified
    if selected_month is not None and selected_month != "None" and selected_year is not None and selected_year != "None":
        selected_year_int = int(selected_year)
        selected_month_num = month_map.get(selected_month, 0)
        
        # Create a cutoff date
        cutoff_date = pd.to_datetime(f"{selected_year_int}-{selected_month_num:02d}-01")
        summary_data_filtered = summary_data[summary_data['Date'] <= cutoff_date].copy()
        
        # Add highlight column for selected month
        summary_data_filtered['is_selected'] = ((summary_data_filtered['Month'] == selected_month) & 
                                               (summary_data_filtered['Year'] == selected_year_int))
    else:
        summary_data_filtered = summary_data.copy()
        summary_data_filtered['is_selected'] = False
    
    # Separate selected and non-selected points
    non_selected = summary_data_filtered[~summary_data_filtered['is_selected']]
    selected = summary_data_filtered[summary_data_filtered['is_selected']]
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Average Cost Trend', 'Max Buying Price Trend', 
                       'Total Weight', 'Product Count'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Average Cost Trend
    if not non_selected.empty:
        fig.add_trace(
            go.Scatter(
                x=non_selected['Date'],
                y=non_selected['Avg Cost Total'],
                mode='lines+markers',
                name='Avg Cost',
                line=dict(color='#2E86AB', width=2),
                marker=dict(size=6),
                showlegend=False
            ),
            row=1, col=1
        )
    if not selected.empty:
        fig.add_trace(
            go.Scatter(
                x=selected['Date'],
                y=selected['Avg Cost Total'],
                mode='markers',
                name='Selected Month',
                marker=dict(size=12, color='#ff0000', symbol='star'),
                showlegend=False
            ),
            row=1, col=1
        )
    
    # Max Buying Price Trend
    if not non_selected.empty:
        fig.add_trace(
            go.Scatter(
                x=non_selected['Date'],
                y=non_selected['Max Buying Price'],
                mode='lines+markers',
                name='Max Buying Price',
                line=dict(color='#A23B72', width=2),
                marker=dict(size=6),
                showlegend=False
            ),
            row=1, col=2
        )
    if not selected.empty:
        fig.add_trace(
            go.Scatter(
                x=selected['Date'],
                y=selected['Max Buying Price'],
                mode='markers',
                name='Selected Month',
                marker=dict(size=12, color='#ff0000', symbol='star'),
                showlegend=False
            ),
            row=1, col=2
        )
    
    # Total Weight
    if not non_selected.empty:
        fig.add_trace(
            go.Scatter(
                x=non_selected['Date'],
                y=non_selected['Weight'],
                mode='lines+markers',
                name='Total Weight',
                line=dict(color='#F18F01', width=2),
                marker=dict(size=6),
                showlegend=False
            ),
            row=2, col=1
        )
    if not selected.empty:
        fig.add_trace(
            go.Scatter(
                x=selected['Date'],
                y=selected['Weight'],
                mode='markers',
                name='Selected Month',
                marker=dict(size=12, color='#ff0000', symbol='star'),
                showlegend=False
            ),
            row=2, col=1
        )
    
    # Product Count
    if not non_selected.empty:
        fig.add_trace(
            go.Scatter(
                x=non_selected['Date'],
                y=non_selected['Product Id'],
                mode='lines+markers',
                name='Product Count',
                line=dict(color='#C73E1D', width=2),
                marker=dict(size=6),
                showlegend=False
            ),
            row=2, col=2
        )
    if not selected.empty:
        fig.add_trace(
            go.Scatter(
                x=selected['Date'],
                y=selected['Product Id'],
                mode='markers',
                name='Selected Month',
                marker=dict(size=12, color='#ff0000', symbol='star'),
                showlegend=False
            ),
            row=2, col=2
        )
    
    # Update layout
    title_text = f"Summary Analytics - {title_suffix}"
    if selected_month is not None and selected_month != "None":
        title_text += f" (Data up to {selected_month} {selected_year})"
        
    fig.update_layout(
        title=title_text,
        height=500,
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    # Update subplot title colors to orange
    fig.update_annotations(font=dict(color='black', size=16))
    
    # Update all x-axes
    for i in range(1, 3):
        for j in range(1, 3):
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', row=i, col=j)
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', row=i, col=j)
    
    return fig
def main():
    st.set_page_config(page_title="Yellow Diamond Dashboard", layout="wide")
    st.title("Yellow Diamond Dashboard")
    st.markdown("Upload Excel files to process multiple sheets and filter data.")
    
    # Initialize session state
    if 'data_processed' not in st.session_state:
        st.session_state.data_processed = False
    if 'master_df' not in st.session_state:
        st.session_state.master_df = pd.DataFrame()
    if 'upload_history' not in st.session_state:
        st.session_state.upload_history = load_upload_history()
    
    # Check and load master dataset if exists and not already loaded
    if st.session_state.master_df.empty and check_master_dataset_exists():
        with st.spinner("Loading master database..."):
            try:
                st.session_state.master_df = load_master_dataset()
                if not st.session_state.master_df.empty:
                    st.success("✅ Master database loaded successfully!")
            except Exception as e:
                st.error(f"Error loading master database: {str(e)}")
        
    # Sidebar for controls
    st.sidebar.header("Controls")
    
    # Display master database status
    if not st.session_state.master_df.empty:
        st.sidebar.success(f"Master DB: {len(st.session_state.master_df)} records")
    else:
        st.sidebar.warning("No master database found")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload Excel File",
        type=['xlsx', 'xls'],
        help="Upload an Excel file with multiple sheets to add to master database"
    )
    
    # Display upload history
    display_upload_history()
    
    # Main content area
    if uploaded_file is not None and not st.session_state.data_processed:
        with st.spinner("Processing Excel file..."):
            try:
                # Get file size
                file_size = uploaded_file.size if hasattr(uploaded_file, 'size') else None
                
                # Process the file
                st.subheader("🗄️ Updating Master Database")
                st.session_state.master_df = get_final_data(uploaded_file)
                cols = st.session_state.master_df.columns.tolist()
                st.session_state.master_df = st.session_state.master_df.groupby(['Product Id','Year', 'Month']).first().reset_index().loc[:,cols]
                st.session_state.data_processed = True
                
                # Add to upload history after successful processing
                st.session_state.upload_history = add_to_upload_history(
                    filename=uploaded_file.name,
                    file_size=file_size
                )
                
                # Show success message
                st.success(f"✅ Successfully processed: {uploaded_file.name}")
                st.info(f"Master database now contains {len(st.session_state.master_df)} records")
                
                # Force sidebar refresh to show updated history
                st.rerun()
                
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                # Still add to history but mark as failed
                history = load_upload_history()
                new_entry = {
                    "filename": uploaded_file.name,
                    "upload_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "file_size": uploaded_file.size if hasattr(uploaded_file, 'size') else None,
                    "status": "Failed"
                }
                history.insert(0, new_entry)
                history = history[:10]
                save_upload_history(history)
                st.session_state.upload_history = history
                
    # Show dashboard if master_df has data (regardless of upload)
    if not st.session_state.master_df.empty:
        # Create filter columns
        Month,Year,Shape,Color,Bucket,Variance_Column = st.columns(6)
        with Month:
            categories = ["None"]+sort_months(list(st.session_state.master_df['Month'].unique()))
            selected_month = st.selectbox("Filter by Month", categories)
        with Year:
            years = ["None"]+sorted(list(st.session_state.master_df['Year'].unique()))
            selected_year = st.selectbox("Filter by Year", years)
        with Shape:
            shapes = ["None"]+list(st.session_state.master_df['Shape key'].unique())
            selected_shape = st.selectbox("Filter by Shape", shapes)
        with Color:
            colors = ["None"]+['WXYZ','FLY','FY','FIY','FVY']
            selected_color = st.selectbox("Filter by Color", colors)
        with Bucket:
            buckets = ["None"]+list(stock_bucket.keys())
            selected_bucket = st.selectbox("Filter by Bucket", buckets)
        with Variance_Column:
            variance_columns = ["None"]+['Current Average Cost','Max Buying Price','Min Selling Price']
            selected_variance_column = st.selectbox("Select Variance Column", variance_columns)
        
        # Apply filters
        cols = st.session_state.master_df.columns.tolist()
        st.session_state.master_df = st.session_state.master_df.groupby(['Product Id','Year', 'Month']).first().reset_index().loc[:,cols]
        filtered_df = st.session_state.master_df.copy()
        
        # Check if all filters are selected (not None)
        all_filters_selected = ((selected_month != "None") & (selected_year != "None") & 
                               (selected_shape != "None") & (selected_color != "None") & 
                               (selected_bucket != "None"))
        
        # Apply partial filters to the dataframe for visualizations
        display_df = st.session_state.master_df.copy()
        if selected_month != "None":
            display_df = display_df[display_df['Month'] == selected_month]
        if selected_year != "None":
            display_df = display_df[display_df['Year'] == int(selected_year)]
        if selected_shape != "None":
            display_df = display_df[display_df['Shape key'] == selected_shape]
        if selected_color != "None":
            display_df = display_df[display_df['Color Key'] == selected_color]
        if selected_bucket != "None":
            display_df = display_df[display_df['Buckets'] == selected_bucket]
        
        # Display summary metrics
        st.subheader("📊 Summary Metrics")
        
        if all_filters_selected:
            # Get detailed metrics when all filters are selected
            filter_data,max_buying_price,current_avg_cost,gap_output,min_selling_price = get_filtered_data(
                selected_month, selected_year, selected_shape, selected_color, selected_bucket)
            MOM_Variance,MOM_Percent_Change,MOM_QoQ_Percent_Change = get_summary_metrics(
                filter_data,selected_month,selected_shape,selected_year,
                selected_color,selected_bucket,selected_variance_column)
            
            mbp,cac,mom_var,mom_perc,qoq_perc,GAP,msp = st.columns(7)
            if type(max_buying_price)!= str:
                with GAP:
                    st.metric("Gap Analysis",value=gap_output,help=f"{'Excess' if gap_output>0 else 'Need' if gap_output < 0 else 'Enough'}")
                with mbp:
                    st.metric("Max Buying Price", f"${max_buying_price:,.2f}")
                with msp:
                    st.metric("Min Selling Price",f"${min_selling_price:,.2f}")
                with cac:
                    st.metric("Current Avg Cost", f"${current_avg_cost:,.2f}",help="90% of Sum of (Average Cost Total) / Weight ")
                with mom_var:
                    st.metric("MOM Variance ", f"{MOM_Variance:,.2f}%")
                with mom_perc:
                    st.metric("MOM Percent Change", f"{MOM_Percent_Change:.2f}%")
                with qoq_perc:
                    st.metric("MOM QoQ Percent Change", f"{MOM_QoQ_Percent_Change:.2f}%")
            else:
                with GAP:
                    st.metric("Gap Analysis",value=gap_output,help=f"{'Excess' if gap_output>0 else 'Need' if gap_output < 0 else 'Enough'}")
                with mbp:
                    st.metric("Max Buying Price", f"0")
                with msp:
                    st.metric("Min Selling Price", f"0")
                with cac:
                    st.metric("Current Avg Cost", f"0")
                with mom_var:
                    st.metric("MOM Variance ", f"0")
                with mom_perc:
                    st.metric("MOM Percent Change", f"0")
                with qoq_perc:
                    st.metric("MOM QoQ Percent Change", f"0")
                if filter_data.empty:
                    st.info("No data present for this specific filter combination")
        else:
            # Show aggregated metrics for partial or no filters
            if not display_df.empty:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    avg_max_buying = display_df['Max Buying Price'].mean()
                    st.metric("Avg Max Buying Price", f"${avg_max_buying:,.2f}")
                with col2:
                    avg_min_selling = display_df['Min Selling Price'].mean()
                    st.metric("Avg Min Selling Price", f"${avg_min_selling:,.2f}")
                with col3:
                    total_weight = display_df['Weight'].sum()
                    st.metric("Total Weight", f"{total_weight:,.2f}")
                with col4:
                    total_products = len(display_df)
                    st.metric("Total Products", f"{total_products:,}")
                
                st.info("💡 Select all filters (Month, Year, Shape, Color, and Bucket) to view detailed metrics including Gap Analysis and MOM Variance calculations.")
            else:
                st.warning("No data available for selected filters")
        
        # Add visualization section - show for all cases
        st.subheader("📈 Trend Analysis")
            
        # Create tabs for different visualizations
        st.markdown("""
        <style>
            /* Style all tab labels */
            .stTabs [data-baseweb="tab-list"] {
                gap: 24px;
            }
            
            .stTabs [data-baseweb="tab-list"] button {
                height: 50px;
                padding-left: 20px;
                padding-right: 20px;
            }
            
            /* Inactive tabs - VIOLET */
            .stTabs [data-baseweb="tab-list"] button p {
                color: #8B00FF;  /* Violet for inactive tabs */
                font-size: 18px;
            }
            
            /* Active tab - RED */
            .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] p {
                color: #FF0000;  /* Red for active tab */
                font-weight: bold;
            }
            
            /* Hover effect */
            .stTabs [data-baseweb="tab-list"] button:hover p {
                color: #FF0000;
                transition: color 0.3s;
            }
            
            /* Tab underline/highlight - RED */
            .stTabs [data-baseweb="tab-highlight"] {
                background-color: #FF0000;
                height: 3px;
            }
            
            /* Tab panels background (optional) */
            .stTabs [data-baseweb="tab-panel"] {
                padding-top: 20px;
            }
        </style>
        """, unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["📊 Variance Trends", "📈 Summary Analytics"])
        
        with tab1:
            if selected_variance_column != "None":
                # Use display_df for visualizations with month/year highlighting
                trend_fig = create_trend_visualization(
                    st.session_state.master_df,  # Use full dataset for trends
                    selected_shape if selected_shape != "None" else None, 
                    selected_color if selected_color != "None" else None, 
                    selected_bucket if selected_bucket != "None" else None, 
                    selected_variance_column,
                    selected_month if selected_month != "None" else None,
                    selected_year if selected_year != "None" else None
                )
                st.plotly_chart(trend_fig, use_container_width=True)
            else:
                st.info("Please select a variance column to view trend analysis.")
        
        with tab2:
            # Use full master_df for visualizations with month/year highlighting
            summary_fig = create_summary_charts(
                st.session_state.master_df,  # Use full dataset for trends
                selected_shape if selected_shape != "None" else None, 
                selected_color if selected_color != "None" else None, 
                selected_bucket if selected_bucket != "None" else None,
                selected_month if selected_month != "None" else None,
                selected_year if selected_year != "None" else None
            )
            st.plotly_chart(summary_fig, use_container_width=True)
        
        # Data Table and Downloads
        st.subheader("📊 Data Table")
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )
        
        # Download buttons
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("💾 Download Filtered Data")
            if 'Product Id' in display_df.columns:
                download_columns = ['Product Id','Shape key','Color Key','Avg Cost Total',
                                  'Min Qty','Max Qty','Buying Price Avg','Max Buying Price']
                available_columns = [col for col in download_columns if col in display_df.columns]
                csv = display_df.loc[:,available_columns].to_csv(index=False)
                st.download_button(
                    label="Download Filtered Data as CSV",
                    data=csv,
                    file_name=f"filtered_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            st.subheader("💾 Download Master Data")
            csv = st.session_state.master_df.to_csv(index=False)
            st.download_button(
                label="Download Master Data as CSV",
                data=csv,
                file_name=f"master_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        # GAP Summary Table - Show for all combinations
        st.subheader("📋 GAP Summary")
        gap_summary_df = get_gap_summary_table(
            st.session_state.master_df, 
            selected_month, 
            selected_year, 
            selected_shape, 
            selected_color, 
            selected_bucket
        )
        
        if not gap_summary_df.empty:
            # Apply styling to highlight negative GAP values in red
            def highlight_negative_gap(row):
                if row['GAP Value'] < 0:
                    return ['background-color: #ffebee; color: #c62828'] * len(row)
                else:
                    return [''] * len(row)
            def highlight_shape_gap(row):
                if row['GAP Value'] < 0:
                    return ['background-color: #ffebee; color: #c62828'] * len(row)
                else:
                    if row['Shape']=='Cushion':
                        return ['background-color: #baffc9; color: #c62828'] * len(row)
                    elif row['Shape']=='Oval':
                        return ['background-color: #bae1ff; color: #c62828'] * len(row)
                    elif row['Shape']=='Pear':
                        return ['background-color: #ffb3ba; color: #c62828'] * len(row)
                    elif row['Shape']=='Radiant':
                        return ['background-color: #ffdfba; color: #c62828'] * len(row)
                    elif row['Shape']=='Other':
                        return ['background-color: #ffffba; color: #c62828'] * len(row)
                    else:
                        return [''] * len(row)
            styled_df = gap_summary_df.style.apply(highlight_shape_gap, axis=1)
            # styled_df = gap_summary_df.style.apply(highlight_negative_gap, axis=1)
            
            st.dataframe(
                styled_df,
                use_container_width=True,
                hide_index=True
            )
            
            # Download GAP Summary
            st.subheader("💾 Download GAP Summary")
            gap_summary_df_cols = ['Month','Year','Shape','Color','Bucket','GAP Value']
            gap_csv = gap_summary_df.loc[:,gap_summary_df_cols].to_csv(index=False)
            gap_csv_excess = gap_summary_df[gap_summary_df['Status']=='Excess'].loc[:,gap_summary_df_cols].to_csv(index=False)
            gap_csv_need = gap_summary_df[gap_summary_df['Status']=='Need'].loc[:,gap_summary_df_cols].to_csv(index=False)
            st.download_button(
                label="Download GAP Summary as CSV",
                data=gap_csv,
                file_name=f"gap_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            st.download_button(
                label="Download GAP Excess Summary as CSV",
                data=gap_csv_excess,
                file_name=f"gap_summary_excess_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            st.download_button(
                label="Download GAP Need Summary as CSV",
                data=gap_csv_need,
                file_name=f"gap_summary_need_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.info("No data available for GAP analysis with current filters.")
        
    else:
        st.info("No data in master database. Upload an Excel file to get started!")
        
    # Reset button
    if st.sidebar.button("Reset Data Processing"):
        st.session_state.data_processed = False
        # Don't clear master_df if it was loaded from file
        if uploaded_file is not None:
            st.session_state.master_df = load_master_dataset() if check_master_dataset_exists() else pd.DataFrame()
        st.rerun()
    
    # Clear history button
    if st.sidebar.button("Clear Upload History"):
        save_upload_history([])
        st.session_state.upload_history = []
        st.success("Upload history cleared!")
        st.rerun()
    
if __name__ == "__main__":
    main()
