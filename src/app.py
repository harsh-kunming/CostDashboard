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
import logging

# Configure logging for production
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
            logger.error(f"Error loading upload history: {e}")
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
        logger.error(f"Error saving upload history: {e}")

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
            df = load_data('kunmings.pkl')
            if df is not None and not df.empty:
                if 'Product Id' in df.columns:
                    df['Product Id'] = df['Product Id'].astype(str)
                return df
            else:
                return pd.DataFrame()
        else:
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading master dataset: {str(e)}")
        logger.error(f"Error loading master dataset: {e}")
        return pd.DataFrame()

def load_data(file):
    """Enhanced load_data function with better error handling"""
    try:
        # Handle different input types
        if isinstance(file, str):
            # String file path (for database files)
            file_type = file.split('.')[-1]
            if file_type == 'csv':
                df = pd.read_csv(file)
                if 'Product Id' in df.columns:
                    df['Product Id'] = df['Product Id'].astype(str)
                return df
            elif file_type == 'pkl':
                df = pd.read_pickle(f"src/{file}")
                if df is not None and not df.empty:
                    if 'Product Id' in df.columns:
                        df['Product Id'] = df['Product Id'].astype(str)
                return df
            elif file_type in ['xlsx', 'xls']:
                df = pd.read_excel(file, sheet_name=None)
                df_dict = {}
                for sheet_name, df_ in df.items():
                    if df_ is not None and not df_.empty:
                        if 'Product Id' in df_.columns:
                            df_['Product Id'] = df_['Product Id'].astype(str)
                        df_dict[sheet_name] = df_
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
                    if df_ is not None and not df_.empty:
                        if 'Product Id' in df_.columns:
                            df_['Product Id'] = df_['Product Id'].astype(str)
                        df_dict[sheet_name] = df_
                return df_dict
            elif file_type == 'pkl':
                df = pd.read_pickle(f"src/{file}")
                if df is not None and not df.empty:
                    if 'Product Id' in df.columns:
                        df['Product Id'] = df['Product Id'].astype(str)
                return df
            elif file_type == 'csv':
                df = pd.read_csv(file)
                if df is not None and not df.empty:
                    if 'Product Id' in df.columns:
                        df['Product Id'] = df['Product Id'].astype(str)
                return df
                
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise e

def save_data(df):
    """Save data with error handling"""
    try:
        # Create src directory if it doesn't exist
        Path("src").mkdir(exist_ok=True)
        df.to_pickle('src/kunmings.pkl')
        logger.info("Data saved successfully")
    except Exception as e:
        logger.error(f"Error saving data: {e}")
        raise e
    
def create_color_key(df, color_map):
    """Create color key with error handling"""
    try:
        df['Color Key'] = df.Color.map(lambda x: color_map.get(x, '') if pd.notna(x) else '')
        return df
    except Exception as e:
        logger.error(f"Error creating color key: {e}")
        return df

def create_bucket(df, stock_bucket=stock_bucket):
    """Create bucket with enhanced error handling"""
    try:
        df['Buckets'] = None
        for key, values in stock_bucket.items():
            lower_bound, upper_bound = values
            mask = (df['Weight'] >= lower_bound) & (df['Weight'] < upper_bound)
            df.loc[mask, 'Buckets'] = key
        return df
    except Exception as e:
        logger.error(f"Error creating buckets: {e}")
        return df

def calculate_avg(df):
    """Calculate average with error handling"""
    try:
        df['Avg Cost Total'] = df['Weight'] * df['Average\nCost\n(USD)']
        return df
    except Exception as e:
        logger.error(f"Error calculating average: {e}")
        return df

def create_date_join(df):
    """Create date join with error handling"""
    try:
        df['Month'] = pd.to_datetime('today').month_name()
        df['Year'] = pd.to_datetime('today').year
        df['Join'] = df['Month'].astype(str) + '-' + df['Year'].map(lambda x: x-2000).astype(str)
        return df
    except Exception as e:
        logger.error(f"Error creating date join: {e}")
        return df

def concatenate_first_two_rows(df):
    """Concatenate first two rows with error handling"""
    try:
        result = {}
        for col in df.columns:
            value1 = str(df.iloc[0][col]) if not pd.isna(df.iloc[0][col]) else ''
            value2 = str(df.iloc[1][col]) if not pd.isna(df.iloc[1][col]) else ''
            result[col] = f"{value1}_{value2}"
        return result
    except Exception as e:
        logger.error(f"Error concatenating rows: {e}")
        return {}

def safe_divide(numerator, denominator, default=0):
    """Safely divide numbers, handling zero division"""
    try:
        if denominator == 0 or pd.isna(denominator) or pd.isna(numerator):
            return default
        return numerator / denominator
    except:
        return default

def populate_max_qty(df, MONTHLY_STOCK_DATA):
    """Populate max qty with enhanced error handling"""
    try:
        columns = list(concatenate_first_two_rows(df.iloc[0:2, 2:]).values())
        columns = ['Months', 'Buckets'] + columns
        df.columns = columns
        df = df.iloc[2:, :]
        df.reset_index(drop=True, inplace=True)
        
        _MAX_QTY_ = []
        MONTHLY_STOCK_DATA['Max Qty'] = None
        
        for indx, row in MONTHLY_STOCK_DATA.iterrows():
            try:
                join = row['Join']
                Shape = row['Shape key']
                Color = row['Color Key']
                Bucket = row['Buckets']
                
                if pd.isna(Color) or pd.isna(Shape) or pd.isna(Bucket):
                    value = 0
                else:
                    col_name = f"{Shape}_{Color}"
                    if col_name in df.columns.tolist():
                        filtered_df = df[(df['Months'] == join) & (df['Buckets'] == Bucket)]
                        if not filtered_df.empty and col_name in filtered_df.columns:
                            value = filtered_df[col_name].iloc[0]
                        else:
                            value = 0
                    else:
                        value = 0
                _MAX_QTY_.append(value)
            except Exception as e:
                logger.error(f"Error processing max qty for row {indx}: {e}")
                _MAX_QTY_.append(0)
        
        MONTHLY_STOCK_DATA['Max Qty'] = _MAX_QTY_
        MONTHLY_STOCK_DATA['Max Qty'] = MONTHLY_STOCK_DATA['Max Qty'].apply(
            lambda x: x[0] if isinstance(x, list) and len(x) > 0 else (x if pd.notna(x) else 0)
        )
        return MONTHLY_STOCK_DATA
    except Exception as e:
        logger.error(f"Error populating max qty: {e}")
        MONTHLY_STOCK_DATA['Max Qty'] = 0
        return MONTHLY_STOCK_DATA

def populate_min_qty(df, MONTHLY_STOCK_DATA):
    """Populate min qty with enhanced error handling"""
    try:
        columns = list(concatenate_first_two_rows(df.iloc[0:2, 2:]).values())
        columns = ['Months', 'Buckets'] + columns
        df.columns = columns
        df = df.iloc[2:, :]
        df.reset_index(drop=True, inplace=True)
        
        _MIN_QTY_ = []
        MONTHLY_STOCK_DATA['Min Qty'] = None
        
        for _, row in MONTHLY_STOCK_DATA.iterrows():
            try:
                join = row['Join']
                Shape = row['Shape key']
                Color = row['Color Key']
                Bucket = row['Buckets']
                
                if pd.isna(Color) or pd.isna(Shape) or pd.isna(Bucket):
                    value = 0
                else:
                    col_name = f"{Shape}_{Color}"
                    if col_name in df.columns.tolist():
                        filtered_df = df[(df['Months'] == join) & (df['Buckets'] == Bucket)]
                        if not filtered_df.empty and col_name in filtered_df.columns:
                            value = filtered_df[col_name].iloc[0]
                        else:
                            value = 0
                    else:
                        value = 0
                _MIN_QTY_.append(value)
            except Exception as e:
                logger.error(f"Error processing min qty: {e}")
                _MIN_QTY_.append(0)
        
        MONTHLY_STOCK_DATA['Min Qty'] = _MIN_QTY_
        MONTHLY_STOCK_DATA['Min Qty'] = MONTHLY_STOCK_DATA['Min Qty'].apply(
            lambda x: x[0] if isinstance(x, list) and len(x) > 0 else (x if pd.notna(x) else 0)
        )
        return MONTHLY_STOCK_DATA
    except Exception as e:
        logger.error(f"Error populating min qty: {e}")
        MONTHLY_STOCK_DATA['Min Qty'] = 0
        return MONTHLY_STOCK_DATA

def populate_selling_prices(df, MONTHLY_STOCK_DATA):
    """Populate selling prices with enhanced error handling"""
    try:
        columns = list(concatenate_first_two_rows(df.iloc[0:2, 1:]).values())
        columns = ['Buckets'] + columns
        df.columns = columns
        df = df.iloc[2:, :]
        df.reset_index(drop=True, inplace=True)
        
        _SELLING_PRICE_ = []
        MONTHLY_STOCK_DATA['Min Selling Price'] = None
        
        for indx, row in MONTHLY_STOCK_DATA.iterrows():
            try:
                Shape = row['Shape key']
                Color = row['Color Key']
                Bucket = row['Buckets']
                
                if pd.isna(Color) or pd.isna(Shape) or pd.isna(Bucket):
                    value = 0
                else:
                    col_name = f"{Shape}_{Color}"
                    if col_name in df.columns.tolist():
                        filtered_df = df[df['Buckets'] == Bucket]
                        if not filtered_df.empty and col_name in filtered_df.columns:
                            value = filtered_df[col_name].iloc[0]
                        else:
                            value = 0
                    else:
                        value = 0
                _SELLING_PRICE_.append(value)
            except Exception as e:
                logger.error(f"Error processing selling price: {e}")
                _SELLING_PRICE_.append(0)
        
        MONTHLY_STOCK_DATA['Min Selling Price'] = _SELLING_PRICE_
        MONTHLY_STOCK_DATA['Min Selling Price'] = MONTHLY_STOCK_DATA['Min Selling Price'].apply(
            lambda x: x[0] if isinstance(x, list) and len(x) > 0 else (x if pd.notna(x) else 0)
        )
        
        # Safe multiplication
        MONTHLY_STOCK_DATA['Min Selling Price'] = (
            MONTHLY_STOCK_DATA['Max Buying Price'].fillna(0) * 
            MONTHLY_STOCK_DATA['Min Selling Price'].fillna(0)
        )
        return MONTHLY_STOCK_DATA
    except Exception as e:
        logger.error(f"Error populating selling prices: {e}")
        MONTHLY_STOCK_DATA['Min Selling Price'] = 0
        return MONTHLY_STOCK_DATA

def populate_buying_prices(df, MONTHLY_STOCK_DATA):
    """Populate buying prices with enhanced error handling"""
    try:
        columns = list(concatenate_first_two_rows(df.iloc[0:2, 2:]).values())
        columns = ['Months', 'Buckets'] + columns
        df.columns = columns
        df = df.iloc[2:, :]
        df.reset_index(drop=True, inplace=True)
        
        _BUYING_PRICE_ = []
        MONTHLY_STOCK_DATA['Max Buying Price'] = None
        
        for indx, row in MONTHLY_STOCK_DATA.iterrows():
            try:
                join = row['Join']
                Shape = row['Shape key']
                Color = row['Color Key']
                Bucket = row['Buckets']
                
                if pd.isna(Color) or pd.isna(Shape) or pd.isna(Bucket):
                    value = 0
                else:
                    col_name = f"{Shape}_{Color}"
                    if col_name in df.columns.tolist():
                        filtered_df = df[(df['Months'] == join) & (df['Buckets'] == Bucket)]
                        if not filtered_df.empty and col_name in filtered_df.columns:
                            value = filtered_df[col_name].iloc[0]
                        else:
                            value = 0
                    else:
                        value = 0
                _BUYING_PRICE_.append(value)
            except Exception as e:
                logger.error(f"Error processing buying price: {e}")
                _BUYING_PRICE_.append(0)
        
        MONTHLY_STOCK_DATA['Max Buying Price'] = _BUYING_PRICE_
        MONTHLY_STOCK_DATA['Max Buying Price'] = MONTHLY_STOCK_DATA['Max Buying Price'].apply(
            lambda x: x[0] if isinstance(x, list) and len(x) > 0 else (x if pd.notna(x) else 0)
        )
        return MONTHLY_STOCK_DATA
    except Exception as e:
        logger.error(f"Error populating buying prices: {e}")
        MONTHLY_STOCK_DATA['Max Buying Price'] = 0
        return MONTHLY_STOCK_DATA

def calculate_buying_price_avg(df):
    """Calculate buying price average with error handling"""
    try:
        df['Buying Price Avg'] = df['Max Buying Price'].fillna(0) * df['Weight'].fillna(0)
        return df
    except Exception as e:
        logger.error(f"Error calculating buying price avg: {e}")
        df['Buying Price Avg'] = 0
        return df

def get_quarter(month):
    """Get quarter with error handling"""
    try:
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
    except Exception as e:
        logger.error(f"Error getting quarter for month {month}: {e}")
        return None

def populate_quarter(df):
    """Populate quarter with error handling"""
    try:
        df['Quarter'] = df['Month'].apply(get_quarter)
        return df
    except Exception as e:
        logger.error(f"Error populating quarter: {e}")
        df['Quarter'] = None
        return df

def create_shape_key(x):
    """Create shape key with error handling"""
    try:
        if pd.isna(x):
            return 'Other'
        
        x_upper = str(x).upper()
        if 'HEART' in x_upper:
            return 'Other'
        elif 'CUSHION' in x_upper:
            return 'Cushion'
        elif 'OVAL' in x_upper:
            return 'Oval'
        elif 'PEAR' in x_upper:
            return 'Pear'
        elif 'CUT-CORNERED' in x_upper:
            return 'Radiant'
        elif 'MODIFIED RECTANGULAR' in x_upper:
            return 'Cushion'
        elif 'MODIFIED SQUARE' in x_upper:
            return 'Cushion'
        elif 'MARQUISE MODIFIED' in x_upper:
            return 'Other'
        elif 'ROUND_CORNERED' in x_upper:
            return 'Cushion'
        elif 'EMERALD' in x_upper:
            return 'Other'
        else:
            return 'Other'
    except Exception as e:
        logger.error(f"Error creating shape key for {x}: {e}")
        return 'Other'

def update_max_qty(df_max_qty, json_data_name='max_qty.pkl'): 
    """Update max qty with enhanced error handling"""
    try:
        json_data_name = 'src/' + json_data_name
        try:
            json_data = joblib.load(json_data_name)
        except:
            json_data = {}
            
        columns = list(concatenate_first_two_rows(df_max_qty.iloc[0:2, 2:]).values())
        columns = ['Months', 'Buckets'] + columns
        df_max_qty.columns = columns
        df_max_qty = df_max_qty.iloc[2:, :]
        json_data = {}
        
        for col in df_max_qty.columns[2:]:
            json_data[col] = {}
            for month in df_max_qty['Months'].unique():
                json_data[col][month] = {}
                for bucket in df_max_qty['Buckets'].unique():
                    filtered_data = df_max_qty[
                        (df_max_qty['Months'] == month) & 
                        (df_max_qty['Buckets'] == bucket)
                    ]
                    if not filtered_data.empty:
                        json_data[col][month][bucket] = filtered_data[col].iloc[0]
                    else:
                        json_data[col][month][bucket] = 0
        
        # Ensure src directory exists
        Path("src").mkdir(exist_ok=True)
        joblib.dump(json_data, json_data_name)
        
    except Exception as e:
        logger.error(f"Error updating max qty: {e}")

def poplutate_monthly_stock_sheet(file):
    """Populate monthly stock sheet with comprehensive error handling"""
    try:
        df = load_data(file)
        
        if not df or not isinstance(df, dict):
            raise ValueError("Unable to load data or data is not in expected format")
        
        required_sheets = ['Monthly Stock Data', 'Buying Max Prices', 'MIN Data', 'MAX Data', 'Min Selling Price']
        for sheet in required_sheets:
            if sheet not in df:
                raise ValueError(f"Required sheet '{sheet}' not found in uploaded file")
        
        df_stock = df['Monthly Stock Data']
        if 'avg' in df_stock.columns:
            df_stock.rename(columns={'avg': 'Avg Cost Total'}, inplace=True)
        
        df_buying = df['Buying Max Prices']
        df_min_qty = df['MIN Data']
        update_max_qty(df_min_qty, json_data_name='min_qty.pkl')
        df_max_qty = df['MAX Data']
        update_max_qty(df_max_qty, json_data_name='max_qty.pkl')
        df_min_sp = df['Min Selling Price']
        
        if any(sheet_df.empty for sheet_df in [df_stock, df_buying, df_min_qty, df_max_qty]):
            raise ValueError("One or more dataframes are empty. Please check the input files.")
        
        # Process data step by step with error handling
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
        df_stock = populate_selling_prices(df_min_sp, df_stock)
        df_stock = df_stock[~(df_stock['Color']=='U-v')]
        df_stock = df_stock[~(df_stock['Weight']<.5)]
        df_stock.reset_index(drop=True,inplace=True)
        df_stock.fillna(0, inplace=True)
        cols = df_stock.columns.tolist()
        df_stock = df_stock.groupby(['Product Id', 'Year', 'Month']).first().reset_index().loc[:, cols]
        
        return df_stock
        
    except Exception as e:
        logger.error(f"Error populating monthly stock sheet: {e}")
        raise e

def calculate_qoq_variance_percentage(current_quarter_price, previous_quarter_price):
    """Calculate QoQ variance with enhanced error handling"""
    try:
        if not isinstance(current_quarter_price, (int, float)) or not isinstance(previous_quarter_price, (int, float)):
            return 0.0
        
        if pd.isna(current_quarter_price) or pd.isna(previous_quarter_price):
            return 0.0
        
        if previous_quarter_price == 0:
            if current_quarter_price == 0:
                return 0.0
            else:
                return 100.0  # 100% increase from 0
        
        variance_percentage = ((current_quarter_price - previous_quarter_price) / previous_quarter_price) * 100
        
        if pd.isna(variance_percentage) or np.isinf(variance_percentage):
            return 0.0
            
        return round(variance_percentage, 2)
        
    except Exception as e:
        logger.error(f"Error calculating QoQ variance: {e}")
        return 0.0

def calculate_qoq_variance_series(price_data):
    """Calculate QoQ variance series with error handling"""
    try:
        if not price_data or len(price_data) < 2:
            return []
        
        variances = []
        for i in range(1, len(price_data)):
            variance = calculate_qoq_variance_percentage(price_data[i], price_data[i-1])
            variances.append(variance)
        
        return variances
        
    except Exception as e:
        logger.error(f"Error calculating QoQ variance series: {e}")
        return []

def monthly_variance(df, col):
    """Calculate monthly variance with enhanced error handling"""
    try:
        if df.empty or col not in df.columns:
            return pd.DataFrame()
        
        analysis = df.groupby(['Month', 'Year'], as_index=False)[col].sum()
        
        if analysis.empty:
            return pd.DataFrame()
        
        analysis['Num_Month'] = analysis['Month'].map(month_map)
        analysis.sort_values(by=['Year', 'Num_Month'], inplace=True)
        
        # Safe percentage change calculation
        analysis['Monthly_change'] = analysis[col].pct_change().fillna(0) * 100
        analysis['Monthly_change'] = analysis['Monthly_change'].replace([np.inf, -np.inf], 0)
        
        # QoQ calculation with error handling
        try:
            qoq_changes = calculate_qoq_variance_series(analysis[col].tolist())
            analysis['qaurter_change'] = [0] + qoq_changes
        except:
            analysis['qaurter_change'] = 0
        
        # Round values
        analysis['Monthly_change'] = analysis['Monthly_change'].round(2)
        analysis['qaurter_change'] = pd.Series(analysis['qaurter_change']).round(2)
        
        return analysis
        
    except Exception as e:
        logger.error(f"Error calculating monthly variance: {e}")
        return pd.DataFrame()

def gap_analysis(max_qty, min_qty, stock_in_hand):
    """Gap analysis with error handling"""
    try:
        max_qty = float(max_qty) if pd.notna(max_qty) else 0
        min_qty = float(min_qty) if pd.notna(min_qty) else 0
        stock_in_hand = float(stock_in_hand) if pd.notna(stock_in_hand) else 0
        
        if stock_in_hand > max_qty:
            return stock_in_hand - max_qty
        elif stock_in_hand < min_qty:
            return stock_in_hand - min_qty
        else:
            return 0
            
    except Exception as e:
        logger.error(f"Error in gap analysis: {e}")
        return 0

def get_filtered_data(FILTER_MONTH, FILTER_YEAR, FILTER_SHAPE, FILTER_COLOR, FILTER_BUCKET):
    """Get filtered data with enhanced error handling"""
    try:
        master_df = load_data('kunmings.pkl')
        master_df = master_df[~(master_df['Color']=='U-v')]
        master_df = master_df[~(master_df['Weight']<.5)]
        master_df.reset_index(drop=True,inplace=True)
        if master_df is None or master_df.empty:
            return [pd.DataFrame(), "No master data available", "No master data available", 0, 0]
        
        cols = master_df.columns.tolist()
        master_df = master_df.groupby(['Product Id', 'Year', 'Month']).first().reset_index().loc[:, cols]
        
        if isinstance(FILTER_YEAR, str) and FILTER_YEAR.isnumeric():
            FILTER_YEAR = int(FILTER_YEAR)
        
        filter_data = master_df[
            (master_df['Month'] == FILTER_MONTH) & 
            (master_df['Year'] == FILTER_YEAR) & 
            (master_df['Shape key'] == FILTER_SHAPE) &
            (master_df['Color Key'] == FILTER_COLOR) &
            (master_df['Buckets'] == FILTER_BUCKET)
        ]
        
        max_qty = filter_data['Max Qty'].max() if not filter_data.empty else 0
        min_qty = filter_data['Min Qty'].min() if not filter_data.empty else 0
        stock_in_hand = filter_data.shape[0]
        
        if stock_in_hand == 0:
            try:
                max_qty_dict = joblib.load('src/max_qty.pkl')
                min_qty_dict = joblib.load('src/min_qty.pkl')
                filter_shape_color = f"{FILTER_SHAPE}_{FILTER_COLOR}"
                
                if filter_shape_color in max_qty_dict:
                    latest_month_max = list(max_qty_dict[filter_shape_color].keys())[-1]
                    max_qty = max_qty_dict[filter_shape_color][latest_month_max].get(FILTER_BUCKET, 0)
                
                if filter_shape_color in min_qty_dict:
                    latest_month_min = list(min_qty_dict[filter_shape_color].keys())[-1]
                    min_qty = min_qty_dict[filter_shape_color][latest_month_min].get(FILTER_BUCKET, 0)
            except Exception as e:
                logger.error(f"Error loading qty dictionaries: {e}")
                max_qty, min_qty = 0, 0
        
        gap_analysis_op = gap_analysis(max_qty, min_qty, stock_in_hand)
        
        if not filter_data.empty:
            max_buying_price = filter_data['Max Buying Price'].max()
            weight_sum = filter_data['Weight'].sum()
            current_avg_cost = safe_divide(filter_data['Avg Cost Total'].sum(), weight_sum, 0) * 0.9
            min_selling_price = filter_data['Min Selling Price'].min()
            
            return [filter_data, int(max_buying_price), int(current_avg_cost), gap_analysis_op, min_selling_price]
        else:
            empty_df = pd.DataFrame(columns=master_df.columns.tolist())
            return [empty_df, f"No data for filters", f"No data for filters", gap_analysis_op, 0]
            
    except Exception as e:
        logger.error(f"Error getting filtered data: {e}")
        empty_df = pd.DataFrame()
        return [empty_df, "Error processing data", "Error processing data", 0, 0]

def get_summary_metrics(filter_data, Filter_Month, FILTER_SHAPE, FILTER_YEAR, FILTER_COLOR, FILTER_BUCKET, FILTER_MONTHLY_VAR_COL):
    """Get summary metrics with comprehensive error handling"""
    try:
        FILTER_YEAR = int(FILTER_YEAR)
        master_df = load_data('kunmings.pkl')
        master_df = master_df[~(master_df['Color']=='U-v')]
        master_df = master_df[~(master_df['Weight']<.5)]
        master_df.reset_index(drop=True,inplace=True)

        if master_df is None or master_df.empty:
            return [0, 0, 0]
        
        cols = master_df.columns.tolist()
        master_df = master_df.groupby(['Product Id', 'Year', 'Month']).first().reset_index().loc[:, cols]
        
        _filter_ = master_df[
            (master_df['Shape key'] == FILTER_SHAPE) &
            (master_df['Color Key'] == FILTER_COLOR) &
            (master_df['Buckets'] == FILTER_BUCKET)
        ]
        
        if _filter_.empty:
            return [0, 0, 0]
        
        # Calculate previous month
        current_month_num = month_map.get(Filter_Month, 1)
        Prev_Month_Name = None
        Prev_Year = FILTER_YEAR
        
        if current_month_num == 1:
            Prev_Month_Name = 'December'
            Prev_Year = FILTER_YEAR - 1
        else:
            for Month_Name, Month_Num in month_map.items():
                if Month_Num == current_month_num - 1:
                    Prev_Month_Name = Month_Name
                    break
        
        Prev_filter_data = master_df[
            (master_df['Month'] == Prev_Month_Name) & 
            (master_df['Year'] == Prev_Year) & 
            (master_df['Shape key'] == FILTER_SHAPE) &
            (master_df['Color Key'] == FILTER_COLOR) &
            (master_df['Buckets'] == FILTER_BUCKET)
        ]
        
        if FILTER_MONTHLY_VAR_COL == 'Current Average Cost':
            variance_col = 'Buying Price Avg'
        elif FILTER_MONTHLY_VAR_COL in ['Max Buying Price', 'Min Selling Price']:
            variance_col = FILTER_MONTHLY_VAR_COL
        else:
            variance_col = 'Max Buying Price'  # Default
        
        # Calculate variance metrics
        var_analysis = monthly_variance(_filter_, variance_col)
        
        if var_analysis.empty:
            return [0, 0, 0]
        
        # Get MOM metrics
        current_analysis = var_analysis[
            (var_analysis['Month'] == Filter_Month) & 
            (var_analysis['Year'] == FILTER_YEAR)
        ]
        
        if not current_analysis.empty:
            MOM_Percent_Change = current_analysis['Monthly_change'].iloc[0]
            MOM_QoQ_Percent_Change = current_analysis['qaurter_change'].iloc[0]
        else:
            MOM_Percent_Change = 0
            MOM_QoQ_Percent_Change = 0
        
        # Calculate MOM Variance
        if FILTER_MONTHLY_VAR_COL == 'Current Average Cost':
            if not filter_data.empty and not Prev_filter_data.empty:
                weight_sum_current = filter_data['Weight'].sum()
                weight_sum_prev = Prev_filter_data['Weight'].sum()
                
                current_avg_cost = safe_divide(filter_data['Avg Cost Total'].sum(), weight_sum_current, 0) * 0.9
                prev_current_avg_cost = safe_divide(Prev_filter_data['Avg Cost Total'].sum(), weight_sum_prev, 0) * 0.9
                
                MOM_Variance = safe_divide((current_avg_cost - prev_current_avg_cost), prev_current_avg_cost, 0) * 100
            else:
                MOM_Variance = 0
        else:
            if not filter_data.empty and variance_col in filter_data.columns:
                avg_value = _filter_[variance_col].mean()
                if avg_value != 0:
                    deviations = (filter_data[variance_col] - avg_value) / avg_value
                    MOM_Variance = (deviations.sum() / len(filter_data)) * 100 if len(filter_data) > 0 else 0
                else:
                    MOM_Variance = 0
            else:
                MOM_Variance = 0
        
        # Clean up infinite or NaN values
        MOM_Variance = 0 if pd.isna(MOM_Variance) or np.isinf(MOM_Variance) else MOM_Variance
        MOM_Percent_Change = 0 if pd.isna(MOM_Percent_Change) or np.isinf(MOM_Percent_Change) else MOM_Percent_Change
        MOM_QoQ_Percent_Change = 0 if pd.isna(MOM_QoQ_Percent_Change) or np.isinf(MOM_QoQ_Percent_Change) else MOM_QoQ_Percent_Change
        
        return [round(MOM_Variance, 2), round(MOM_Percent_Change, 2), round(MOM_QoQ_Percent_Change, 2)]
        
    except Exception as e:
        logger.error(f"Error getting summary metrics: {e}")
        return [0, 0, 0]

def get_gap_summary_table(master_df, selected_month, selected_year, selected_shape, selected_color, selected_bucket):
    """Generate GAP summary table with enhanced error handling"""
    try:
        if master_df.empty:
            return pd.DataFrame()
        
        gap_summary = []
        master_df = master_df[~(master_df['Color']=='U-v')]
        master_df = master_df[~(master_df['Weight']<.5)]
        master_df.reset_index(drop=True,inplace=True)

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
                            try:
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
                                    min_selling_price = int(filtered_data['Min Selling Price'].max())
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
                                        'Status': 'Excess' if gap_value > 0 else 'Need' if gap_value < 0 else 'Adequate',
                                        'Min Selling Price' : min_selling_price
                                    })
                                else:
                                    # Try to get from saved dictionaries
                                    try:
                                        max_qty_dict = joblib.load('src/max_qty.pkl')
                                        min_qty_dict = joblib.load('src/min_qty.pkl')
                                        filter_shape_color = f"{shape}_{color}"
                                        
                                        if filter_shape_color in max_qty_dict:
                                            latest_month = list(max_qty_dict[filter_shape_color].keys())[-1]
                                            max_qty = max_qty_dict[filter_shape_color][latest_month].get(bucket, 0)
                                        else:
                                            max_qty = 0
                                        
                                        if filter_shape_color in min_qty_dict:
                                            latest_month = list(min_qty_dict[filter_shape_color].keys())[-1]
                                            min_qty = min_qty_dict[filter_shape_color][latest_month].get(bucket, 0)
                                        else:
                                            min_qty = 0
                                        
                                        gap_value = gap_analysis(max_qty, min_qty, 0)
                                        
                                        gap_summary.append({
                                            'Month': month,
                                            'Year': year,
                                            'Shape': shape,
                                            'Color': color,
                                            'Bucket': bucket,
                                            'Max Qty': max_qty,
                                            'Min Qty': min_qty,
                                            'Stock in Hand': 0,
                                            'GAP Value': int(gap_value),
                                            'Status': 'Excess' if gap_value > 0 else 'Need' if gap_value < 0 else 'Adequate'
                                        })
                                    except Exception as e2:
                                        logger.error(f"Error loading qty dictionaries for {shape}_{color}: {e2}")
                                        continue
                                        
                            except Exception as e:
                                logger.error(f"Error processing gap summary for {month}-{year}-{shape}-{color}-{bucket}: {e}")
                                continue
        
        if gap_summary:
            return pd.DataFrame(gap_summary).sort_values(by=['Shape', 'Color', 'Bucket'])
        else:
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"Error generating gap summary table: {e}")
        return pd.DataFrame()

def get_final_data(file, PARENT_DF='kunmings.pkl'):
    """Get final data with error handling"""
    try:
        df = poplutate_monthly_stock_sheet(file)
        parent_df = load_data(PARENT_DF)
        
        if parent_df is None or parent_df.empty:
            master_df = df
        else:
            master_df = pd.concat([df, parent_df], ignore_index=True, axis=0)
        
        cols = master_df.columns.tolist()
        master_df = master_df.groupby(['Product Id', 'Year', 'Month']).first().reset_index().loc[:, cols]
        master_df = master_df[~(master_df['Color']=='U-v')]
        master_df = master_df[~(master_df['Weight']<.5)]
        master_df.reset_index(drop=True,inplace=True)

        save_data(master_df)
        
        return master_df
        
    except Exception as e:
        logger.error(f"Error getting final data: {e}")
        raise e

def sort_months(months):
    """Sort months with error handling"""
    try:
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
        
    except Exception as e:
        logger.error(f"Error sorting months: {e}")
        return months  # Return original list if sorting fails

def create_trend_visualization(master_df, selected_shape=None, selected_color=None, selected_bucket=None, 
                             selected_variance_column=None, selected_month=None, selected_year=None):
    """Create trend line visualizations with comprehensive error handling"""
    try:
        # Input validation
        if master_df is None or master_df.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font=dict(size=16)
            )
            return fig
        
        # Filter data based on selections
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
        
        title_suffix = " | ".join(title_parts) if title_parts else "All Data"
        
        if filtered_df.empty:
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
        elif variance_col == 'None' or variance_col is None:
            variance_col = 'Max Buying Price'  # Default column
        
        # Check if variance column exists
        if variance_col not in filtered_df.columns:
            fig = go.Figure()
            fig.add_annotation(
                text=f"Column '{variance_col}' not found in data",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font=dict(size=16)
            )
            return fig
        
        # Calculate monthly variance data
        var_analysis = monthly_variance(filtered_df, variance_col)
        
        if var_analysis.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No variance data available for analysis",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font=dict(size=16)
            )
            return fig
        
        # Create date column for proper sorting
        var_analysis['Date'] = pd.to_datetime(
            '01-' + var_analysis['Num_Month'].astype(str) + '-' + var_analysis['Year'].astype(str), 
            format='%d-%m-%Y',
            errors='coerce'
        )
        var_analysis = var_analysis.dropna(subset=['Date']).sort_values('Date')
        
        if var_analysis.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No valid date data for visualization",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font=dict(size=16)
            )
            return fig
        
        # Filter data up to selected month/year if specified
        if (selected_month is not None and selected_month != "None" and 
            selected_year is not None and selected_year != "None"):
            try:
                selected_year_int = int(selected_year)
                selected_month_num = month_map.get(selected_month, 0)
                
                cutoff_date = pd.to_datetime(f"{selected_year_int}-{selected_month_num:02d}-01")
                var_analysis_filtered = var_analysis[var_analysis['Date'] <= cutoff_date].copy()
                
                var_analysis_filtered['is_selected'] = (
                    (var_analysis_filtered['Month'] == selected_month) & 
                    (var_analysis_filtered['Year'] == selected_year_int)
                )
            except Exception as e:
                logger.error(f"Error filtering by date: {e}")
                var_analysis_filtered = var_analysis.copy()
                var_analysis_filtered['is_selected'] = False
        else:
            var_analysis_filtered = var_analysis.copy()
            var_analysis_filtered['is_selected'] = False
        
        if var_analysis_filtered.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No data available after date filtering",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font=dict(size=16)
            )
            return fig
        
        # Separate selected and non-selected points
        non_selected = var_analysis_filtered[~var_analysis_filtered['is_selected']]
        selected = var_analysis_filtered[var_analysis_filtered['is_selected']]
        
        # Create subplot with secondary y-axis
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Monthly Variance Trend', 'Quarter-over-Quarter Change'),
            vertical_spacing=0.1,
            specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
        )
        
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
                    showlegend=False
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
        
        # Update subplot title colors
        fig.update_annotations(font=dict(color='black', size=16))
        
        # Update axes
        fig.update_xaxes(
            title_text="Date",
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            row=2, col=1
        )
        
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
        logger.error(f"Error creating trend visualization: {e}")
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
    """Create summary charts with comprehensive error handling - FIXED VERSION"""
    try:
        # Input validation
        if master_df is None or master_df.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font=dict(size=16)
            )
            return fig
        
        # Group data first to avoid duplicates
        cols = master_df.columns.tolist()
        master_df_grouped = master_df.groupby(['Product Id', 'Year', 'Month']).first().reset_index().loc[:, cols]
        
        # Filter data based on selections
        filtered_df = master_df_grouped.copy()
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
        
        title_suffix = " | ".join(title_parts) if title_parts else "All Data"
        
        # Check if filtered data is empty
        if filtered_df.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No data available for selected filters",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font=dict(size=16, color='red')
            )
            fig.update_layout(
                title=f"Summary Analytics - {title_suffix}",
                height=500,
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            return fig
        
        # Check if required columns exist
        required_cols = ['Avg Cost Total', 'Max Buying Price', 'Weight', 'Product Id', 'Month', 'Year']
        missing_cols = [col for col in required_cols if col not in filtered_df.columns]
        
        if missing_cols:
            fig = go.Figure()
            fig.add_annotation(
                text=f"Missing required columns: {', '.join(missing_cols)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font=dict(size=16)
            )
            return fig
        
        # Group by month and year to get summary statistics
        try:
            summary_data = filtered_df.groupby(['Month', 'Year']).agg({
                'Avg Cost Total': 'mean',
                'Max Buying Price': 'mean',
                'Weight': 'sum',
                'Product Id': 'count'
            }).reset_index()
        except Exception as e:
            logger.error(f"Error grouping data: {e}")
            fig = go.Figure()
            fig.add_annotation(
                text="Error processing summary data",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font=dict(size=16)
            )
            return fig
        
        # Check if summary data is empty after grouping
        if summary_data.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No summary data available after grouping",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font=dict(size=16, color='red')
            )
            fig.update_layout(
                title=f"Summary Analytics - {title_suffix}",
                height=500,
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            return fig
        
        # Create date column for proper sorting
        try:
            summary_data['Num_Month'] = summary_data['Month'].map(month_map)
            # Remove any rows where month mapping failed
            summary_data = summary_data.dropna(subset=['Num_Month'])
            
            if summary_data.empty:
                fig = go.Figure()
                fig.add_annotation(
                    text="No valid month data found",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, xanchor='center', yanchor='middle',
                    showarrow=False, font=dict(size=16)
                )
                return fig
            
            # Create date for sorting
            summary_data['Date'] = pd.to_datetime(
                '01-' + summary_data['Num_Month'].astype(int).astype(str) + '-' + summary_data['Year'].astype(str), 
                format='%d-%m-%Y',
                errors='coerce'
            )
            
            # Remove rows with invalid dates
            summary_data = summary_data.dropna(subset=['Date'])
            
            if summary_data.empty:
                fig = go.Figure()
                fig.add_annotation(
                    text="No valid date data for visualization",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, xanchor='center', yanchor='middle',
                    showarrow=False, font=dict(size=16)
                )
                return fig
            
            summary_data = summary_data.sort_values('Date')
            
        except Exception as e:
            logger.error(f"Error processing dates: {e}")
            fig = go.Figure()
            fig.add_annotation(
                text="Error processing date data",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font=dict(size=16)
            )
            return fig
        
        # Filter data up to selected month/year if specified
        if (selected_month is not None and selected_month != "None" and 
            selected_year is not None and selected_year != "None"):
            try:
                selected_year_int = int(selected_year)
                selected_month_num = month_map.get(selected_month, 0)
                
                if selected_month_num > 0:
                    cutoff_date = pd.to_datetime(f"{selected_year_int}-{selected_month_num:02d}-01")
                    summary_data_filtered = summary_data[summary_data['Date'] <= cutoff_date].copy()
                    
                    # Add highlight column for selected month
                    summary_data_filtered['is_selected'] = (
                        (summary_data_filtered['Month'] == selected_month) & 
                        (summary_data_filtered['Year'] == selected_year_int)
                    )
                else:
                    summary_data_filtered = summary_data.copy()
                    summary_data_filtered['is_selected'] = False
            except Exception as e:
                logger.error(f"Error filtering by selected date: {e}")
                summary_data_filtered = summary_data.copy()
                summary_data_filtered['is_selected'] = False
        else:
            summary_data_filtered = summary_data.copy()
            summary_data_filtered['is_selected'] = False
        
        # Final check for empty data
        if summary_data_filtered.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No data available after date filtering",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font=dict(size=16, color='red')
            )
            fig.update_layout(
                title=f"Summary Analytics - {title_suffix}",
                height=500,
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            return fig
        
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
        
        # Color scheme
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        
        # Average Cost Trend
        if not non_selected.empty:
            fig.add_trace(
                go.Scatter(
                    x=non_selected['Date'],
                    y=non_selected['Avg Cost Total'],
                    mode='lines+markers',
                    name='Avg Cost',
                    line=dict(color=colors[0], width=2),
                    marker=dict(size=6),
                    showlegend=False,
                    hovertemplate='<b>%{x|%b %Y}</b><br>Avg Cost: $%{y:,.2f}<extra></extra>'
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
                    showlegend=False,
                    hovertemplate='<b>%{x|%b %Y} (Selected)</b><br>Avg Cost: $%{y:,.2f}<extra></extra>'
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
                    line=dict(color=colors[1], width=2),
                    marker=dict(size=6),
                    showlegend=False,
                    hovertemplate='<b>%{x|%b %Y}</b><br>Max Price: $%{y:,.2f}<extra></extra>'
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
                    showlegend=False,
                    hovertemplate='<b>%{x|%b %Y} (Selected)</b><br>Max Price: $%{y:,.2f}<extra></extra>'
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
                    line=dict(color=colors[2], width=2),
                    marker=dict(size=6),
                    showlegend=False,
                    hovertemplate='<b>%{x|%b %Y}</b><br>Weight: %{y:,.2f}<extra></extra>'
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
                    showlegend=False,
                    hovertemplate='<b>%{x|%b %Y} (Selected)</b><br>Weight: %{y:,.2f}<extra></extra>'
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
                    line=dict(color=colors[3], width=2),
                    marker=dict(size=6),
                    showlegend=False,
                    hovertemplate='<b>%{x|%b %Y}</b><br>Count: %{y}<extra></extra>'
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
                    showlegend=False,
                    hovertemplate='<b>%{x|%b %Y} (Selected)</b><br>Count: %{y}<extra></extra>'
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
        
        # Update subplot title colors
        fig.update_annotations(font=dict(color='black', size=16))
        
        # Update all axes
        for i in range(1, 3):
            for j in range(1, 3):
                fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', row=i, col=j)
                fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', row=i, col=j)
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating summary charts: {e}")
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating summary visualization: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16)
        )
        fig.update_layout(
            title="Summary Analytics - Error",
            height=500,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        return fig

def main():
    """Main application function with comprehensive error handling"""
    try:
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
                        logger.info("Master database loaded successfully")
                except Exception as e:
                    st.error(f"Error loading master database: {str(e)}")
                    logger.error(f"Error loading master database: {e}")
            
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
                    st.session_state.master_df = st.session_state.master_df.groupby(['Product Id', 'Year', 'Month']).first().reset_index().loc[:, cols]
                    st.session_state.master_df = st.session_state.master_df[~(st.session_state.master_df['Color']=='U-v')]
                    st.session_state.master_df = st.session_state.master_df[~(st.session_state.master_df['Weight']<.5)]
                    st.session_state.master_df.reset_index(drop=True,inplace=True)

                    st.session_state.data_processed = True
                    
                    # Add to upload history after successful processing
                    st.session_state.upload_history = add_to_upload_history(
                        filename=uploaded_file.name,
                        file_size=file_size
                    )
                    
                    # Show success message
                    st.success(f"✅ Successfully processed: {uploaded_file.name}")
                    st.info(f"Master database now contains {len(st.session_state.master_df)} records")
                    logger.info(f"Successfully processed file: {uploaded_file.name}")
                    
                    # Force sidebar refresh to show updated history
                    st.rerun()
                    
                except Exception as e:
                    error_msg = f"Error processing file: {str(e)}"
                    st.error(error_msg)
                    logger.error(f"Error processing file {uploaded_file.name}: {e}")
                    
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
            try:
                # Create filter columns
                Month, Year, Shape, Color, Bucket, Variance_Column = st.columns(6)
                
                with Month:
                    categories = ["None"] + sort_months(list(st.session_state.master_df['Month'].unique()))
                    selected_month = st.selectbox("Filter by Month", categories)
                with Year:
                    years = ["None"] + sorted(list(st.session_state.master_df['Year'].unique()))
                    selected_year = st.selectbox("Filter by Year", years)
                with Shape:
                    shapes = ["None"] + list(st.session_state.master_df['Shape key'].unique())
                    selected_shape = st.selectbox("Filter by Shape", shapes)
                with Color:
                    colors = ["None"] + ['WXYZ', 'FLY', 'FY', 'FIY', 'FVY']
                    selected_color = st.selectbox("Filter by Color", colors)
                with Bucket:
                    buckets = ["None"] + list(stock_bucket.keys())
                    selected_bucket = st.selectbox("Filter by Bucket", buckets)
                with Variance_Column:
                    variance_columns = ["None"] + ['Current Average Cost', 'Max Buying Price', 'Min Selling Price']
                    selected_variance_column = st.selectbox("Select Variance Column", variance_columns)
                
                # Apply filters
                cols = st.session_state.master_df.columns.tolist()
                st.session_state.master_df = st.session_state.master_df.groupby(['Product Id', 'Year', 'Month']).first().reset_index().loc[:, cols]
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
                    try:
                        filter_data, max_buying_price, current_avg_cost, gap_output, min_selling_price = get_filtered_data(
                            selected_month, selected_year, selected_shape, selected_color, selected_bucket)
                        MOM_Variance, MOM_Percent_Change, MOM_QoQ_Percent_Change = get_summary_metrics(
                            filter_data, selected_month, selected_shape, selected_year,
                            selected_color, selected_bucket, selected_variance_column)
                        
                        mbp, cac, mom_var, mom_perc, qoq_perc, GAP, msp = st.columns(7)
                        
                        if isinstance(max_buying_price, (int, float)):
                            with GAP:
                                st.metric("Gap Analysis", value=gap_output, 
                                         help=f"{'Excess' if gap_output > 0 else 'Need' if gap_output < 0 else 'Enough'}")
                            with mbp:
                                st.metric("Max Buying Price", f"${max_buying_price:,.2f}")
                            with msp:
                                st.metric("Min Selling Price", f"${min_selling_price:,.2f}")
                            with cac:
                                st.metric("Current Avg Cost", f"${current_avg_cost:,.2f}", 
                                         help="90% of Sum of (Average Cost Total) / Weight")
                            with mom_var:
                                st.metric("MOM Variance", f"{MOM_Variance:,.2f}%")
                            with mom_perc:
                                st.metric("MOM Percent Change", f"{MOM_Percent_Change:.2f}%")
                            with qoq_perc:
                                st.metric("MOM QoQ Percent Change", f"{MOM_QoQ_Percent_Change:.2f}%")
                        else:
                            # Display zeros when no data
                            with GAP:
                                st.metric("Gap Analysis", value=gap_output, 
                                         help=f"{'Excess' if gap_output > 0 else 'Need' if gap_output < 0 else 'Enough'}")
                            with mbp:
                                st.metric("Max Buying Price", "0")
                            with msp:
                                st.metric("Min Selling Price", "0")
                            with cac:
                                st.metric("Current Avg Cost", "0")
                            with mom_var:
                                st.metric("MOM Variance", "0")
                            with mom_perc:
                                st.metric("MOM Percent Change", "0")
                            with qoq_perc:
                                st.metric("MOM QoQ Percent Change", "0")
                            
                            if filter_data.empty:
                                st.info("No data present for this specific filter combination")
                                
                    except Exception as e:
                        st.error(f"Error calculating detailed metrics: {str(e)}")
                        logger.error(f"Error calculating detailed metrics: {e}")
                else:
                    # Show aggregated metrics for partial or no filters
                    if not display_df.empty:
                        try:
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
                        except Exception as e:
                            st.error(f"Error calculating aggregated metrics: {str(e)}")
                            logger.error(f"Error calculating aggregated metrics: {e}")
                    else:
                        st.warning("No data available for selected filters")
                
                # Add visualization section
                st.subheader("📈 Trend Analysis")
                
                # Custom CSS for tabs
                st.markdown("""
                <style>
                    .stTabs [data-baseweb="tab-list"] {
                        gap: 24px;
                    }
                    
                    .stTabs [data-baseweb="tab-list"] button {
                        height: 50px;
                        padding-left: 20px;
                        padding-right: 20px;
                    }
                    
                    .stTabs [data-baseweb="tab-list"] button p {
                        color: #8B00FF;
                        font-size: 18px;
                    }
                    
                    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] p {
                        color: #FF0000;
                        font-weight: bold;
                    }
                    
                    .stTabs [data-baseweb="tab-list"] button:hover p {
                        color: #FF0000;
                        transition: color 0.3s;
                    }
                    
                    .stTabs [data-baseweb="tab-highlight"] {
                        background-color: #FF0000;
                        height: 3px;
                    }
                    
                    .stTabs [data-baseweb="tab-panel"] {
                        padding-top: 20px;
                    }
                </style>
                """, unsafe_allow_html=True)
                
                tab1, tab2 = st.tabs(["📊 Variance Trends", "📈 Summary Analytics"])
                
                with tab1:
                    try:
                        if selected_variance_column != "None":
                            trend_fig = create_trend_visualization(
                                st.session_state.master_df,
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
                    except Exception as e:
                        st.error(f"Error creating trend visualization: {str(e)}")
                        logger.error(f"Error in trend visualization tab: {e}")
                
                with tab2:
                    try:
                        summary_fig = create_summary_charts(
                            st.session_state.master_df,
                            selected_shape if selected_shape != "None" else None, 
                            selected_color if selected_color != "None" else None, 
                            selected_bucket if selected_bucket != "None" else None,
                            selected_month if selected_month != "None" else None,
                            selected_year if selected_year != "None" else None
                        )
                        st.plotly_chart(summary_fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error creating summary visualization: {str(e)}")
                        logger.error(f"Error in summary analytics tab: {e}")
                
                # Data Table and Downloads
                st.subheader("📊 Data Table")
                cols = display_df.columns.tolist()
                display_df = display_df.groupby(['Product Id','Year', 'Month']).first().reset_index().loc[:,cols]
                try:
                    st.dataframe(
                        display_df,
                        use_container_width=True,
                        hide_index=True
                    )
                except Exception as e:
                    st.error(f"Error displaying data table: {str(e)}")
                    logger.error(f"Error displaying data table: {e}")
                
                # Download buttons
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("💾 Download Filtered Data")
                    try:
                        if 'Product Id' in display_df.columns:
                            download_columns = ['Product Id', 'Shape key', 'Color Key', 'Avg Cost Total',
                                              'Min Qty', 'Max Qty', 'Buying Price Avg', 'Max Buying Price']
                            available_columns = [col for col in download_columns if col in display_df.columns]
                            csv = display_df.loc[:, available_columns].to_csv(index=False)
                            st.download_button(
                                label="Download Filtered Data as CSV",
                                data=csv,
                                file_name=f"filtered_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                    except Exception as e:
                        st.error(f"Error preparing filtered data download: {str(e)}")
                        logger.error(f"Error preparing filtered data download: {e}")
                
                with col2:
                    st.subheader("💾 Download Master Data")
                    try:
                        csv = st.session_state.master_df.to_csv(index=False)
                        st.download_button(
                            label="Download Master Data as CSV",
                            data=csv,
                            file_name=f"master_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    except Exception as e:
                        st.error(f"Error preparing master data download: {str(e)}")
                        logger.error(f"Error preparing master data download: {e}")
                
                # GAP Summary Table
                st.subheader("📋 GAP Summary")
                try:
                    gap_summary_df = get_gap_summary_table(
                        st.session_state.master_df, 
                        selected_month, 
                        selected_year, 
                        selected_shape, 
                        selected_color, 
                        selected_bucket
                    )
                    
                    if not gap_summary_df.empty:
                        # Styling function
                        def highlight_shape_gap(row):
                            if row['GAP Value'] < 0:
                                return ['background-color: #ffebee; color: #c62828'] * len(row)
                            else:
                                if row['Shape'] == 'Cushion':
                                    return ['background-color: #baffc9; color: #c62828'] * len(row)
                                elif row['Shape'] == 'Oval':
                                    return ['background-color: #bae1ff; color: #c62828'] * len(row)
                                elif row['Shape'] == 'Pear':
                                    return ['background-color: #ffb3ba; color: #c62828'] * len(row)
                                elif row['Shape'] == 'Radiant':
                                    return ['background-color: #ffdfba; color: #c62828'] * len(row)
                                elif row['Shape'] == 'Other':
                                    return ['background-color: #ffffba; color: #c62828'] * len(row)
                                else:
                                    return [''] * len(row)
                        
                        styled_df = gap_summary_df.style.apply(highlight_shape_gap, axis=1)
                        
                        st.dataframe(
                            styled_df,
                            use_container_width=True,
                            hide_index=True
                        )
                        
                        # Download GAP Summary
                        st.subheader("💾 Download GAP Summary")
                        try:
                            gap_summary_df_cols = ['Month', 'Year', 'Shape', 'Color', 'Bucket', 'GAP Value']
                            gap_csv = gap_summary_df.loc[:, gap_summary_df_cols].to_csv(index=False)
                            gap_csv_excess = gap_summary_df[gap_summary_df['Status'] == 'Excess'].loc[:, gap_summary_df_cols+['min_selling_price']].to_csv(index=False)
                            gap_csv_need = gap_summary_df[gap_summary_df['Status'] == 'Need'].loc[:, gap_summary_df_cols].to_csv(index=False)
                            
                            col_gap1, col_gap2, col_gap3 = st.columns(3)
                            
                            with col_gap1:
                                st.download_button(
                                    label="Download GAP Summary as CSV",
                                    data=gap_csv,
                                    file_name=f"gap_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv"
                                )
                            
                            with col_gap2:
                                st.download_button(
                                    label="Download GAP Excess Summary as CSV",
                                    data=gap_csv_excess,
                                    file_name=f"gap_summary_excess_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv"
                                )
                            
                            with col_gap3:
                                st.download_button(
                                    label="Download GAP Need Summary as CSV",
                                    data=gap_csv_need,
                                    file_name=f"gap_summary_need_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv"
                                )
                        except Exception as e:
                            st.error(f"Error preparing GAP summary downloads: {str(e)}")
                            logger.error(f"Error preparing GAP summary downloads: {e}")
                            
                    else:
                        st.info("No data available for GAP analysis with current filters.")
                        
                except Exception as e:
                    st.error(f"Error generating GAP summary: {str(e)}")
                    logger.error(f"Error generating GAP summary: {e}")
                    
            except Exception as e:
                st.error(f"Error in dashboard display: {str(e)}")
                logger.error(f"Error in dashboard display: {e}")
        else:
            st.info("No data in master database. Upload an Excel file to get started!")
            
        # Control buttons in sidebar
        if st.sidebar.button("Reset Data Processing"):
            st.session_state.data_processed = False
            # Don't clear master_df if it was loaded from file
            if uploaded_file is not None:
                st.session_state.master_df = load_master_dataset() if check_master_dataset_exists() else pd.DataFrame()
            st.rerun()
        
        if st.sidebar.button("Clear Upload History"):
            save_upload_history([])
            st.session_state.upload_history = []
            st.success("Upload history cleared!")
            st.rerun()
            
    except Exception as e:
        st.error(f"Critical application error: {str(e)}")
        logger.error(f"Critical application error: {e}")

if __name__ == "__main__":
    main()
