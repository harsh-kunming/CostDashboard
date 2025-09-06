import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple
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
from functools import lru_cache
import hashlib
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Configure logging for production
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Stable session state initialization
def initialize_session_state():
    """Initialize session state with stable defaults"""
    defaults = {
        'data_processed': False,
        'master_df': pd.DataFrame(),
        'upload_history': [],
        'last_filter_hash': None,
        'cached_gap_summary': pd.DataFrame(),
        'cached_gap_summary_hash': None,
        'stable_display_df': pd.DataFrame(),
        'gap_summary_display_html': None,
        'show_upload_history': False
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

# Cache configurations with stable keys
@st.cache_data(ttl=3600, show_spinner=False)
def load_cached_master_dataset():
    """Load master dataset with proper compression handling"""
    try:
        master_file_path = Path(r"C:\Users\himabirla\Downloads\CostDashboard-main (3)\CostDashboard-main\src\kunmings.pkl")
        if master_file_path.exists():
            # Try loading with compression first (for files saved with gzip)
            try:
                df = pd.read_pickle(master_file_path, compression='gzip')
                logger.info("Loaded compressed pickle file successfully")
            except:
                # Fallback to uncompressed loading
                try:
                    df = pd.read_pickle(master_file_path, compression=None)
                    logger.info("Loaded uncompressed pickle file successfully")
                except Exception as e:
                    logger.error(f"Failed to load pickle file: {e}")
                    # Try to detect file type and handle accordingly
                    with open(master_file_path, 'rb') as f:
                        header = f.read(2)
                        if header == b'\x1f\x8b':  # gzip magic number
                            df = pd.read_pickle(master_file_path, compression='gzip')
                        else:
                            df = pd.read_pickle(master_file_path, compression=None)
            
            if df is not None and not df.empty:
                if 'Product Id' in df.columns:
                    df['Product Id'] = df['Product Id'].astype(str)
                return df
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error loading master dataset: {e}")
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def apply_data_filters_stable(df_csv_string: str):
    """Stable cached version of data filtering"""
    try:
        # Reconstruct dataframe from string representation
        from io import StringIO
        df = pd.read_csv(StringIO(df_csv_string))
        
        original_count = len(df)
        
        # Vectorized filtering
        mask = pd.Series(True, index=df.index)
        
        if 'Color' in df.columns:
            color_mask = ~df['Color'].isin(['U-V', 'S-T', 'Fancy Deep Yellow'])
            mask &= color_mask
            
        if 'Weight' in df.columns:
            weight_mask = df['Weight'] >= 0.5
            mask &= weight_mask
            
        if 'Shape key' in df.columns:
            shape_mask = df['Shape key'] != 'Other'
            mask &= shape_mask
        
        df_filtered = df[mask].reset_index(drop=True)
        
        filtered_count = len(df_filtered)
        if original_count != filtered_count:
            logger.info(f"Data filtering: {original_count} -> {filtered_count} rows")
        
        return df_filtered
        
    except Exception as e:
        logger.error(f"Error applying data filters: {e}")
        return pd.DataFrame()

def apply_data_filters(df):
    """Optimized data filtering with stable caching"""
    if df is None or df.empty:
        return df
    
    try:
        # Create stable hash for caching
        df_csv_string = df.to_csv(index=False)
        return apply_data_filters_stable(df_csv_string)
    except:
        return df

@st.cache_data(show_spinner=False)
def load_qty_dictionaries():
    """Cache loading of quantity dictionaries"""
    try:
        max_qty_dict = joblib.load(r'C:\Users\himabirla\Downloads\CostDashboard-main (3)\CostDashboard-main\src\max_qty.pkl')
        min_qty_dict = joblib.load(r'C:\Users\himabirla\Downloads\CostDashboard-main (3)\CostDashboard-main\src\min_qty.pkl')
        max_buy_dict = joblib.load(r'C:\Users\himabirla\Downloads\CostDashboard-main (3)\CostDashboard-main\src\max_buy.pkl')
        return max_qty_dict, min_qty_dict, max_buy_dict
    except Exception as e:
        logger.error(f"Error loading qty dictionaries: {e}")
        return {}, {}, {}

def create_stable_filter_hash(month, year, shape, color, bucket):
    """Create stable hash for filter combination"""
    filter_string = f"{month}_{year}_{shape}_{color}_{bucket}"
    return hashlib.md5(filter_string.encode()).hexdigest()

@st.cache_data(show_spinner=False)
def get_gap_summary_stable(df_hash: str, filter_hash: str):
    """Stable cached GAP summary calculation"""
    try:
        # Parse filter parameters from hash
        filter_params = st.session_state.get('gap_filter_params', {})
        
        selected_month = filter_params.get('month', 'None')
        selected_year = filter_params.get('year', 'None')
        selected_shape = filter_params.get('shape', 'None')
        selected_color = filter_params.get('color', 'None')
        selected_bucket = filter_params.get('bucket', 'None')
        
        # Get master dataframe
        master_df = st.session_state.get('master_df', pd.DataFrame())
        if master_df.empty:
            return pd.DataFrame()
        
        gap_summary = []
        
        # Get filter values efficiently
        months = [selected_month] if selected_month != "None" else master_df['Month'].unique()
        years = [selected_year] if selected_year != "None" else master_df['Year'].unique()
        shapes = [selected_shape] if selected_shape != "None" else master_df['Shape key'].unique()
        colors = [selected_color] if selected_color != "None" else master_df['Color Key'].unique()
        buckets = [selected_bucket] if selected_bucket != "None" else master_df['Buckets'].unique()
        
        # Load qty dictionaries once
        max_qty_dict, min_qty_dict, max_buy_dict = load_qty_dictionaries()
        
        # Process combinations efficiently
        for month in months:
            for year in years:
                for shape in shapes:
                    for color in colors:
                        for bucket in buckets:
                            # Single filter operation
                            mask = (
                                (master_df['Month'] == month) & 
                                (master_df['Year'] == year) & 
                                (master_df['Shape key'] == shape) & 
                                (master_df['Color Key'] == color) & 
                                (master_df['Buckets'] == bucket)
                            )
                            filtered_data = master_df[mask]
                            
                            if not filtered_data.empty:
                                # Vectorized aggregations
                                max_qty = int(filtered_data['Max Qty'].max())
                                min_qty = int(filtered_data['Min Qty'].min())
                                stock_in_hand = len(filtered_data)
                                gap_value = gap_analysis(max_qty, min_qty, stock_in_hand)
                                min_selling_price = int(filtered_data['Min Selling Price'].max())
                                max_buying_price = int(filtered_data['Max Buying Price'].max())
                            else:
                                # Use cached dictionaries
                                filter_shape_color = f"{shape}_{color}"
                                max_qty = max_qty_dict.get(filter_shape_color, {}).get(f"{month}-{int(year)-2000}", {}).get(bucket, 0)
                                min_qty = min_qty_dict.get(filter_shape_color, {}).get(f"{month}-{int(year)-2000}", {}).get(bucket, 0)
                                max_buying_price = max_buy_dict.get(filter_shape_color, {}).get(f"{month}-{int(year)-2000}", {}).get(bucket, 0)
                                gap_value = gap_analysis(max_qty, min_qty, 0)
                                min_selling_price = 0
                                stock_in_hand = 0
                            
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
                                'Min Selling Price': min_selling_price,
                                'Max Buying Price': max_buying_price
                            })
        
        if gap_summary:
            result_df = pd.DataFrame(gap_summary)
            return result_df.sort_values(by=['Shape', 'Color', 'Bucket']).reset_index(drop=True)
        else:
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"Error generating gap summary table: {e}")
        return pd.DataFrame()

def get_gap_summary_table(master_df, selected_month, selected_year, selected_shape, selected_color, selected_bucket):
    """Stable GAP summary with proper caching"""
    if master_df.empty:
        return pd.DataFrame()
    
    # Store filter parameters in session state for caching
    st.session_state.gap_filter_params = {
        'month': selected_month,
        'year': selected_year,
        'shape': selected_shape,
        'color': selected_color,
        'bucket': selected_bucket
    }
    
    # Create stable hashes
    df_hash = hashlib.md5(pd.util.hash_pandas_object(master_df).values.tobytes()).hexdigest()
    filter_hash = create_stable_filter_hash(selected_month, selected_year, selected_shape, selected_color, selected_bucket)
    
    return get_gap_summary_stable(df_hash, filter_hash)

def create_gap_table_html(gap_summary_df):
    """Create styled GAP table as HTML to avoid pickle issues"""
    try:
        if gap_summary_df.empty:
            return "<p>No data available</p>"
        
        def get_row_style(row):
            """Get style for table row based on GAP value and shape"""
            if row['GAP Value'] < 0:
                return 'background-color: #ffebee; color: #c62828;'
            elif row['Shape'] == 'Cushion':
                return 'background-color: #e8f5e8; color: #2e7d32;'
            elif row['Shape'] == 'Oval':
                return 'background-color: #e3f2fd; color: #1565c0;'
            elif row['Shape'] == 'Pear':
                return 'background-color: #fce4ec; color: #ad1457;'
            elif row['Shape'] == 'Radiant':
                return 'background-color: #fff3e0; color: #ef6c00;'
            else:
                return 'background-color: #fffde7; color: #f57f17;'
        
        # Create HTML table
        html = '<table style="width: 100%; border-collapse: collapse;">'
        
        # Header
        html += '<thead><tr style="background-color: #f0f0f0; font-weight: bold;">'
        for col in gap_summary_df.columns:
            html += f'<th style="padding: 8px; border: 1px solid #ddd; text-align: left;">{col}</th>'
        html += '</tr></thead><tbody>'
        
        # Data rows
        for _, row in gap_summary_df.iterrows():
            row_style = get_row_style(row)
            html += f'<tr style="{row_style}">'
            for col in gap_summary_df.columns:
                html += f'<td style="padding: 8px; border: 1px solid #ddd;">{row[col]}</td>'
            html += '</tr>'
        
        html += '</tbody></table>'
        return html
        
    except Exception as e:
        logger.error(f"Error creating GAP table HTML: {e}")
        return "<p>Error creating table</p>"

# Optimized file history management
@lru_cache(maxsize=1)
def get_history_file_path():
    """Cached path getter"""
    history_dir = Path("history")
    history_dir.mkdir(exist_ok=True)
    return history_dir / "upload_history.json"

@st.cache_data(ttl=300, show_spinner=False)
def load_upload_history():
    """Cached upload history loading"""
    history_file = get_history_file_path()
    if history_file.exists():
        try:
            with open(history_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading upload history: {e}")
            return []
    return []

def save_upload_history(history: List[Dict]):
    """Optimized history saving"""
    history_file = get_history_file_path()
    try:
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)
        # Clear cache after saving
        load_upload_history.clear()
    except Exception as e:
        st.error(f"Error saving history: {e}")
        logger.error(f"Error saving upload history: {e}")

def display_upload_history():
    """Display upload history in sidebar with production-ready styling"""
    try:
        # Load current history
        history = load_upload_history()
        
        if not history:
            st.sidebar.info("No upload history available")
            return
        
        # Create expandable upload history section
        with st.sidebar.expander("📁 Upload History", expanded=st.session_state.show_upload_history):
            st.markdown("**Recent File Uploads**")
            
            for i, entry in enumerate(history):
                # Create a container for each upload entry
                with st.container():
                    # Parse upload time
                    try:
                        upload_time = datetime.strptime(entry['upload_time'], "%Y-%m-%d %H:%M:%S")
                        time_str = upload_time.strftime("%m/%d %H:%M")
                    except:
                        time_str = entry.get('upload_time', 'Unknown')
                    
                    # Format file size
                    file_size = entry.get('file_size', 0)
                    if file_size:
                        if file_size < 1024:
                            size_str = f"{file_size} B"
                        elif file_size < 1024**2:
                            size_str = f"{file_size/1024:.1f} KB"
                        else:
                            size_str = f"{file_size/(1024**2):.1f} MB"
                    else:
                        size_str = "Unknown"
                    
                    # Status indicator
                    status = entry.get('status', 'Unknown')
                    if status == 'Processed':
                        status_emoji = "✅"
                        status_color = "green"
                    elif status == 'Failed':
                        status_emoji = "❌"
                        status_color = "red"
                    else:
                        status_emoji = "⏳"
                        status_color = "orange"
                    
                    # Display entry with styling
                    st.markdown(
                        f"""
                        <div style="border: 1px solid #ddd; padding: 8px; margin: 4px 0; border-radius: 4px; background-color: #f9f9f9;">
                            <div style="font-weight: bold; color: #333; font-size: 14px;">
                                {status_emoji} {entry['filename']}
                            </div>
                            <div style="font-size: 12px; color: #666; margin-top: 4px;">
                                📅 {time_str} | 📊 {size_str}
                            </div>
                            <div style="font-size: 12px; color: {status_color}; margin-top: 2px;">
                                Status: {status}
                            </div>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
            
            # Summary information
            st.markdown(f"**Total uploads:** {len(history)}")
            successful_uploads = len([h for h in history if h.get('status') == 'Processed'])
            st.markdown(f"**Successful:** {successful_uploads}")
            
    except Exception as e:
        st.sidebar.error(f"Error displaying upload history: {str(e)}")
        logger.error(f"Error displaying upload history: {e}")

def format_file_size(size_bytes):
    """Format file size in human readable format"""
    if size_bytes == 0 or size_bytes is None:
        return "0 B"
    try:
        size_names = ["B", "KB", "MB", "GB"]
        i = int(np.floor(np.log(size_bytes) / np.log(1024)))
        p = pow(1024, i)
        s = round(size_bytes / p, 2)
        return f"{s} {size_names[i]}"
    except:
        return "Unknown"

# MISSING FUNCTIONS - Complete implementations

def concatenate_first_two_rows(df):
    """Stable row concatenation with error handling"""
    try:
        if len(df) < 2:
            return {}
        
        row1 = df.iloc[0].fillna('').astype(str)
        row2 = df.iloc[1].fillna('').astype(str)
        
        return {col: f"{row1[col]}_{row2[col]}" for col in df.columns}
    except Exception as e:
        logger.error(f"Error concatenating rows: {e}")
        return {}

def update_max_qty(df_max_qty, json_data_name='max_qty.pkl'):
    """Stable max qty update with error handling"""
    try:
        json_data_path = rf"C:\Users\himabirla\Downloads\CostDashboard-main (3)\CostDashboard-main\src\{json_data_name}"
        
        # Try to load existing data
        try:
            json_data = joblib.load(json_data_path)
        except:
            json_data = {}
        
        # Process column names
        columns = list(concatenate_first_two_rows(df_max_qty.iloc[0:2, 2:]).values())
        columns = ['Months', 'Buckets'] + columns
        df_max_qty.columns = columns
        df_max_qty = df_max_qty.iloc[2:].reset_index(drop=True)
        
        # Stable processing
        json_data = {}
        
        for col in df_max_qty.columns[2:]:
            json_data[col] = {}
            col_data = df_max_qty[['Months', 'Buckets', col]].set_index(['Months', 'Buckets'])[col]
            
            for month in df_max_qty['Months'].unique():
                json_data[col][month] = {}
                month_data = col_data[col_data.index.get_level_values('Months') == month]
                
                for bucket in df_max_qty['Buckets'].unique():
                    try:
                        value = month_data[month_data.index.get_level_values('Buckets') == bucket].iloc[0]
                        json_data[col][month][bucket] = value
                    except:
                        json_data[col][month][bucket] = 0
        
        # Ensure directory exists and save
        Path("src").mkdir(exist_ok=True)
        joblib.dump(json_data, json_data_path)
        
        logger.info(f"Successfully updated {json_data_name}")
        
    except Exception as e:
        logger.error(f"Error updating max qty: {e}")

def populate_max_qty(df, MONTHLY_STOCK_DATA):
    """Stable max qty population"""
    try:
        columns = list(concatenate_first_two_rows(df.iloc[0:2, 2:]).values())
        columns = ['Months', 'Buckets'] + columns
        df.columns = columns
        df = df.iloc[2:].reset_index(drop=True)
        
        max_qty_values = []
        
        for _, row in MONTHLY_STOCK_DATA.iterrows():
            join = row['Join']
            shape = row['Shape key']
            color = row['Color Key']
            bucket = row['Buckets']
            
            if pd.isna(color) or pd.isna(shape) or pd.isna(bucket):
                max_qty_values.append(0)
                continue
            
            col_name = f"{shape}_{color}"
            if col_name in df.columns:
                mask = (df['Months'] == join) & (df['Buckets'] == bucket)
                filtered_rows = df[mask]
                if not filtered_rows.empty:
                    value = filtered_rows[col_name].iloc[0]
                else:
                    value = 0
            else:
                value = 0
            
            max_qty_values.append(value)
        
        MONTHLY_STOCK_DATA['Max Qty'] = max_qty_values
        MONTHLY_STOCK_DATA['Max Qty'] = MONTHLY_STOCK_DATA['Max Qty'].apply(
            lambda x: x[0] if isinstance(x, list) and len(x) > 0 else (x if pd.notna(x) else 0)
        )
        
        return MONTHLY_STOCK_DATA
    except Exception as e:
        logger.error(f"Error populating max qty: {e}")
        MONTHLY_STOCK_DATA['Max Qty'] = 0
        return MONTHLY_STOCK_DATA

def populate_min_qty(df, MONTHLY_STOCK_DATA):
    """Stable min qty population"""
    try:
        columns = list(concatenate_first_two_rows(df.iloc[0:2, 2:]).values())
        columns = ['Months', 'Buckets'] + columns
        df.columns = columns
        df = df.iloc[2:].reset_index(drop=True)
        
        min_qty_values = []
        
        for _, row in MONTHLY_STOCK_DATA.iterrows():
            join = row['Join']
            shape = row['Shape key']
            color = row['Color Key']
            bucket = row['Buckets']
            
            if pd.isna(color) or pd.isna(shape) or pd.isna(bucket):
                min_qty_values.append(0)
                continue
            
            col_name = f"{shape}_{color}"
            if col_name in df.columns:
                mask = (df['Months'] == join) & (df['Buckets'] == bucket)
                filtered_rows = df[mask]
                if not filtered_rows.empty:
                    value = filtered_rows[col_name].iloc[0]
                else:
                    value = 0
            else:
                value = 0
            
            min_qty_values.append(value)
        
        MONTHLY_STOCK_DATA['Min Qty'] = min_qty_values
        MONTHLY_STOCK_DATA['Min Qty'] = MONTHLY_STOCK_DATA['Min Qty'].apply(
            lambda x: x[0] if isinstance(x, list) and len(x) > 0 else (x if pd.notna(x) else 0)
        )
        
        return MONTHLY_STOCK_DATA
    except Exception as e:
        logger.error(f"Error populating min qty: {e}")
        MONTHLY_STOCK_DATA['Min Qty'] = 0
        return MONTHLY_STOCK_DATA

def populate_buying_prices(df, MONTHLY_STOCK_DATA):
    """Stable buying price population"""
    try:
        columns = list(concatenate_first_two_rows(df.iloc[0:2, 2:]).values())
        columns = ['Months', 'Buckets'] + columns
        df.columns = columns
        df = df.iloc[2:].reset_index(drop=True)
        
        buying_price_values = []
        
        for _, row in MONTHLY_STOCK_DATA.iterrows():
            join = row['Join']
            shape = row['Shape key']
            color = row['Color Key']
            bucket = row['Buckets']
            
            if pd.isna(color) or pd.isna(shape) or pd.isna(bucket):
                buying_price_values.append(0)
                continue
            
            col_name = f"{shape}_{color}"
            if col_name in df.columns:
                mask = (df['Months'] == join) & (df['Buckets'] == bucket)
                filtered_rows = df[mask]
                if not filtered_rows.empty:
                    value = filtered_rows[col_name].iloc[0]
                else:
                    value = 0
            else:
                value = 0
            
            buying_price_values.append(value)
        
        MONTHLY_STOCK_DATA['Max Buying Price'] = buying_price_values
        MONTHLY_STOCK_DATA['Max Buying Price'] = MONTHLY_STOCK_DATA['Max Buying Price'].apply(
            lambda x: x[0] if isinstance(x, list) and len(x) > 0 else (x if pd.notna(x) else 0)
        )
        
        return MONTHLY_STOCK_DATA
    except Exception as e:
        logger.error(f"Error populating buying prices: {e}")
        MONTHLY_STOCK_DATA['Max Buying Price'] = 0
        return MONTHLY_STOCK_DATA

def populate_selling_prices(df, MONTHLY_STOCK_DATA):
    """Stable selling price population"""
    try:
        columns = list(concatenate_first_two_rows(df.iloc[0:2, 1:]).values())
        columns = ['Buckets'] + columns
        df.columns = columns
        df = df.iloc[2:].reset_index(drop=True)
        
        selling_price_values = []
        
        for _, row in MONTHLY_STOCK_DATA.iterrows():
            shape = row['Shape key']
            color = row['Color Key']
            bucket = row['Buckets']
            
            if pd.isna(color) or pd.isna(shape) or pd.isna(bucket):
                selling_price_values.append(0)
                continue
            
            col_name = f"{shape}_{color}"
            if col_name in df.columns:
                mask = df['Buckets'] == bucket
                filtered_rows = df[mask]
                if not filtered_rows.empty:
                    value = filtered_rows[col_name].iloc[0]
                else:
                    value = 0
            else:
                value = 0
            
            selling_price_values.append(value)
        
        MONTHLY_STOCK_DATA['Min Selling Price'] = selling_price_values
        MONTHLY_STOCK_DATA['Min Selling Price'] = MONTHLY_STOCK_DATA['Min Selling Price'].apply(
            lambda x: x[0] if isinstance(x, list) and len(x) > 0 else (x if pd.notna(x) else 0)
        )
        
        # Vectorized multiplication
        MONTHLY_STOCK_DATA['Min Selling Price'] = (
            MONTHLY_STOCK_DATA['Max Buying Price'].fillna(0) * 
            MONTHLY_STOCK_DATA['Min Selling Price'].fillna(0)
        )
        
        return MONTHLY_STOCK_DATA
    except Exception as e:
        logger.error(f"Error populating selling prices: {e}")
        MONTHLY_STOCK_DATA['Min Selling Price'] = 0
        return MONTHLY_STOCK_DATA

def calculate_buying_price_avg(df):
    """Stable buying price average calculation"""
    try:
        df = df.copy()
        df['Buying Price Avg'] = df['Max Buying Price'].fillna(0) * df['Weight'].fillna(0)
        return df
    except Exception as e:
        logger.error(f"Error calculating buying price avg: {e}")
        df['Buying Price Avg'] = 0
        return df

@st.cache_data(show_spinner=False)
def monthly_variance_stable(df_csv: str, col: str):
    """Stable monthly variance calculation"""
    try:
        from io import StringIO
        df = pd.read_csv(StringIO(df_csv))
        
        if df.empty or col not in df.columns:
            return pd.DataFrame()
        
        # Group data efficiently
        analysis = df.groupby(['Month', 'Year'], as_index=False)[col].sum()
        
        if analysis.empty:
            return pd.DataFrame()
        
        analysis['Num_Month'] = analysis['Month'].map(month_map)
        analysis = analysis.sort_values(by=['Year', 'Num_Month'])
        
        # Vectorized percentage change
        analysis['Monthly_change'] = analysis[col].pct_change().fillna(0) * 100
        analysis['Monthly_change'] = analysis['Monthly_change'].replace([np.inf, -np.inf], 0)
        
        # QoQ calculation
        qoq_changes = [0] + [
            ((analysis[col].iloc[i] - analysis[col].iloc[i-1]) / analysis[col].iloc[i-1] * 100) 
            if analysis[col].iloc[i-1] != 0 else 0
            for i in range(1, len(analysis))
        ]
        analysis['qaurter_change'] = qoq_changes
        
        # Round values
        analysis['Monthly_change'] = analysis['Monthly_change'].round(2)
        analysis['qaurter_change'] = analysis['qaurter_change'].round(2)
        
        return analysis
        
    except Exception as e:
        logger.error(f"Error calculating monthly variance: {e}")
        return pd.DataFrame()

def monthly_variance(df, col):
    """Stable monthly variance with caching"""
    if df.empty or col not in df.columns:
        return pd.DataFrame()
    
    df_csv = df.to_csv(index=False)
    return monthly_variance_stable(df_csv, col)

# Additional helper functions
def optimized_process_data_pipeline(df):
    """Optimized data processing pipeline"""
    try:
        if df is None or df.empty:
            return df
        
        df = df.copy()
        df = optimized_create_date_join(df)
        df = populate_quarter(df)
        df = optimized_calculate_avg(df)
        df = optimized_create_bucket(df)
        df = optimized_create_color_key(df)
        
        if 'Shape' in df.columns:
            df['Shape key'] = vectorized_shape_key_mapping(df['Shape'])
        
        return df
        
    except Exception as e:
        logger.error(f"Error in optimized data pipeline: {e}")
        return df

def vectorized_shape_key_mapping(shapes_series):
    """Vectorized shape key creation"""
    try:
        shapes_upper = shapes_series.str.upper().fillna('')
        result = pd.Series('Other', index=shapes_series.index)
        
        result.loc[shapes_upper.str.contains('CUSHION|MODIFIED RECTANGULAR|MODIFIED SQUARE|ROUND_CORNERED', na=False)] = 'Cushion'
        result.loc[shapes_upper.str.contains('OVAL', na=False)] = 'Oval'
        result.loc[shapes_upper.str.contains('PEAR', na=False)] = 'Pear'
        result.loc[shapes_upper.str.contains('CUT-CORNERED', na=False)] = 'Radiant'
        
        return result
    except Exception as e:
        logger.error(f"Error in vectorized shape mapping: {e}")
        return pd.Series('Other', index=shapes_series.index)

def optimized_calculate_avg(df):
    """Vectorized average calculation"""
    try:
        df = df.copy()
        df['Avg Cost Total'] = df['Weight'].fillna(0) * df['Average\nCost\n(USD)'].fillna(0)
        return df
    except Exception as e:
        logger.error(f"Error calculating average: {e}")
        df['Avg Cost Total'] = 0
        return df

def optimized_create_date_join(df):
    """Optimized date join creation"""
    try:
        df = df.copy()
        current_month = pd.to_datetime('today').month_name()
        current_year = pd.to_datetime('today').year
        
        df['Month'] = current_month
        df['Year'] = current_year
        df['Join'] = f"{current_month}-{current_year - 2000}"
        return df
    except Exception as e:
        logger.error(f"Error creating date join: {e}")
        return df

def optimized_create_color_key(df):
    """Vectorized color key creation"""
    try:
        df = df.copy()
        df['Color Key'] = df['Color'].map(color_map).fillna('')
        return df
    except Exception as e:
        logger.error(f"Error creating color key: {e}")
        df['Color Key'] = ''
        return df

def optimized_create_bucket(df):
    """Vectorized bucket creation"""
    try:
        df = df.copy()
        df['Buckets'] = 'Other'
        
        weights = df['Weight'].fillna(0)
        
        for bucket_name, (lower, upper) in stock_bucket.items():
            mask = (weights >= lower) & (weights < upper)
            df.loc[mask, 'Buckets'] = bucket_name
            
        return df
    except Exception as e:
        logger.error(f"Error creating buckets: {e}")
        df['Buckets'] = 'Other'
        return df

def gap_analysis(max_qty, min_qty, stock_in_hand):
    """Optimized gap analysis"""
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
    except:
        return 0

def get_quarter(month):
    """Optimized quarter calculation"""
    quarter_map = {
        'January': 'Q1', 'February': 'Q1', 'March': 'Q1',
        'April': 'Q2', 'May': 'Q2', 'June': 'Q2',
        'July': 'Q3', 'August': 'Q3', 'September': 'Q3',
        'October': 'Q4', 'November': 'Q4', 'December': 'Q4'
    }
    try:
        year = pd.to_datetime('today').year
        yr = year - 2000
        quarter = quarter_map.get(month, 'Q1')
        return f'{quarter}-{yr}'
    except:
        return None

def populate_quarter(df):
    """Vectorized quarter population"""
    try:
        df = df.copy()
        df['Quarter'] = df['Month'].apply(get_quarter)
        return df
    except Exception as e:
        logger.error(f"Error populating quarter: {e}")
        df['Quarter'] = None
        return df

def load_data(file):
    """Optimized load_data function"""
    try:
        if isinstance(file, str):
            if file.endswith('.pkl'):
                df = pd.read_pickle(file)
                if df is not None and not df.empty:
                    if 'Product Id' in df.columns:
                        df['Product Id'] = df['Product Id'].astype(str)
                return df
            elif file.endswith('.csv'):
                df = pd.read_csv(file)
                if 'Product Id' in df.columns:
                    df['Product Id'] = df['Product Id'].astype(str)
                return df
        else:
            if hasattr(file, 'name'):
                file_type = file.name.split('.')[-1]
            else:
                file_type = 'xlsx'
            
            if file_type in ['xlsx', 'xls']:
                df_dict = pd.read_excel(file, sheet_name=None)
                for sheet_name, df_ in df_dict.items():
                    if df_ is not None and not df_.empty:
                        if 'Product Id' in df_.columns:
                            df_['Product Id'] = df_['Product Id'].astype(str)
                return df_dict
            elif file_type == 'csv':
                df = pd.read_csv(file)
                if 'Product Id' in df.columns:
                    df['Product Id'] = df['Product Id'].astype(str)
                return df
                
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise e

def optimized_save_data(df):
    """Stable data saving with proper compression handling"""
    try:
        df = apply_data_filters(df)
        
        # Ensure directory exists
        Path("src").mkdir(exist_ok=True)
        
        file_path = r'C:\Users\himabirla\Downloads\CostDashboard-main (3)\CostDashboard-main\src\kunmings.pkl'
        
        # Save with consistent compression (or without compression for compatibility)
        df.to_pickle(file_path, compression=None)  # Changed to None for compatibility
        
        # Clear relevant caches
        load_cached_master_dataset.clear()
        
        logger.info("Data saved successfully")
    except Exception as e:
        logger.error(f"Error saving data: {e}")
        raise e

@st.cache_data(show_spinner=False)
def sort_months_stable(months_tuple):
    """Stable cached month sorting"""
    try:
        import calendar
        month_mapping = {calendar.month_name[i]: i for i in range(1, 13)}
        months_list = list(months_tuple)
        return sorted(months_list, key=lambda month: month_mapping.get(month, 13))
    except:
        return list(months_tuple)

def optimize_dataframe_groupby(df):
    """Optimized groupby with stable results"""
    if df is None or df.empty:
        return df
    
    try:
        cols = df.columns.tolist()
        grouped_df = df.groupby(['Product Id', 'Year', 'Month']).first().reset_index().loc[:, cols]
        return grouped_df
    except Exception as e:
        logger.error(f"Error in groupby operation: {e}")
        return df

def optimized_populate_monthly_stock_sheet(file):
    """Stable monthly stock sheet population"""
    try:
        df = load_data(file)
        
        if not df or not isinstance(df, dict):
            raise ValueError("Unable to load data or data is not in expected format")
        
        required_sheets = ['Monthly Stock Data', 'Buying Max Prices', 'MIN Data', 'MAX Data', 'Min Selling Price']
        missing_sheets = [sheet for sheet in required_sheets if sheet not in df]
        
        if missing_sheets:
            raise ValueError(f"Missing required sheets: {missing_sheets}")
        
        # Extract sheets
        df_stock = df['Monthly Stock Data'].copy()
        if 'avg' in df_stock.columns:
            df_stock.rename(columns={'avg': 'Avg Cost Total'}, inplace=True)
        
        df_buying = df['Buying Max Prices'].copy()
        df_min_qty = df['MIN Data'].copy()
        df_max_qty = df['MAX Data'].copy()
        df_min_sp = df['Min Selling Price'].copy()
        
        # Validate data
        if any(sheet_df.empty for sheet_df in [df_stock, df_buying, df_min_qty, df_max_qty]):
            raise ValueError("One or more dataframes are empty")
        
        # Process pipeline
        df_stock = optimized_process_data_pipeline(df_stock)
        
        # Process stock data
        df_stock = optimized_populate_stock_data(df_max_qty, df_min_qty, df_buying, df_min_sp, df_stock)
        
        # Final processing
        df_stock = apply_data_filters(df_stock)
        df_stock = df_stock.fillna(0)
        
        # Group data
        df_stock = optimize_dataframe_groupby(df_stock)
        
        return df_stock
        
    except Exception as e:
        logger.error(f"Error populating monthly stock sheet: {e}")
        raise e

def optimized_populate_stock_data(df_max_qty, df_min_qty, df_buying, df_min_sp, monthly_stock_data):
    """Stable population of stock data"""
    try:
        # Update dictionaries
        update_max_qty(df_min_qty, 'min_qty.pkl')
        update_max_qty(df_max_qty, 'max_qty.pkl')
        update_max_qty(df_buying, 'max_buy.pkl')
        
        # Apply transformations
        monthly_stock_data = populate_max_qty(df_max_qty, monthly_stock_data)
        monthly_stock_data = populate_min_qty(df_min_qty, monthly_stock_data)
        monthly_stock_data = populate_buying_prices(df_buying, monthly_stock_data)
        monthly_stock_data = calculate_buying_price_avg(monthly_stock_data)
        monthly_stock_data = populate_selling_prices(df_min_sp, monthly_stock_data)
        
        return monthly_stock_data
        
    except Exception as e:
        logger.error(f"Error in stock data population: {e}")
        return monthly_stock_data

def optimized_get_filtered_data(filter_month, filter_year, filter_shape, filter_color, filter_bucket):
    """Optimized filtered data retrieval"""
    try:
        master_df = st.session_state.get('master_df', pd.DataFrame())
        
        if master_df.empty:
            return [pd.DataFrame(), 0, "No data", 0, 0]
        
        # Single vectorized filter operation
        mask = (
            (master_df['Month'] == filter_month) & 
            (master_df['Year'] == int(filter_year)) & 
            (master_df['Shape key'] == filter_shape) &
            (master_df['Color Key'] == filter_color) &
            (master_df['Buckets'] == filter_bucket)
        )
        
        filter_data = master_df[mask]
        
        if not filter_data.empty:
            max_buying_price = filter_data['Max Buying Price'].max()
            weight_sum = filter_data['Weight'].sum()
            current_avg_cost = (filter_data['Avg Cost Total'].sum() / weight_sum * 0.9) if weight_sum > 0 else 0
            min_selling_price = filter_data['Min Selling Price'].min()
            max_qty = filter_data['Max Qty'].max() 
            min_qty = filter_data['Min Qty'].min() 
        else:
            max_qty_dict, min_qty_dict, max_buy_dict = load_qty_dictionaries()
            filter_shape_color = f"{filter_shape}_{filter_color}"
            join_key = f"{filter_month}-{int(filter_year) - 2000}"
            
            max_buying_price = max_buy_dict.get(filter_shape_color, {}).get(join_key, {}).get(filter_bucket, 0)
            latest_month_max = list(max_qty_dict[filter_shape_color].keys())[-1]
            max_qty = max_qty_dict[filter_shape_color][latest_month_max].get(filter_bucket, 0)
            min_qty = min_qty_dict.get(filter_shape_color, {}).get(join_key, {}).get(filter_bucket, 0)
            current_avg_cost = 0
            min_selling_price = 0
        
        # Calculate gap
        
        stock_in_hand = len(filter_data)
        gap_output = gap_analysis(max_qty, min_qty, stock_in_hand)
        
        return [filter_data, int(max_buying_price), int(current_avg_cost), gap_output, min_selling_price]
        
    except Exception as e:
        logger.error(f"Error getting filtered data: {e}")
        return [pd.DataFrame(), 0, "Error", 0, 0]

def optimized_get_summary_metrics(filter_data, filter_month, filter_shape, filter_year, 
                                filter_color, filter_bucket, filter_monthly_var_col):
    """Optimized summary metrics calculation"""
    try:
        # Simplified version - return stable metrics
        return [0, 0, 0]  # Placeholder for mom_variance, mom_percent_change, mom_qoq_percent_change
        
    except Exception as e:
        logger.error(f"Error getting summary metrics: {e}")
        return [0, 0, 0]

# Visualization functions with datetime fix
def create_trend_visualization(master_df, selected_shape, selected_color, selected_bucket,
                             selected_variance_column, selected_month, selected_year):
    """Create stable trend visualization"""
    try:
        # Apply filters
        filtered_df = master_df.copy()
        title_parts = []
        
        mask = pd.Series(True, index=filtered_df.index)
        
        if selected_shape and selected_shape != "None":
            mask &= (filtered_df['Shape key'] == selected_shape)
            title_parts.append(selected_shape)
        if selected_color and selected_color != "None":
            mask &= (filtered_df['Color Key'] == selected_color)
            title_parts.append(selected_color)
        if selected_bucket and selected_bucket != "None":
            mask &= (filtered_df['Buckets'] == selected_bucket)
            title_parts.append(selected_bucket)
        
        filtered_df = filtered_df[mask]
        title_suffix = " | ".join(title_parts) if title_parts else "All Data"
        
        if filtered_df.empty:
            return create_empty_figure("No data available for selected filters")
        
        # Process variance column
        variance_col = selected_variance_column
        if variance_col == 'Current Average Cost':
            variance_col = 'Buying Price Avg'
        elif variance_col in ['None', None]:
            variance_col = 'Max Buying Price'
        
        if variance_col not in filtered_df.columns:
            return create_empty_figure(f"Column '{variance_col}' not found")
        
        # Get variance analysis
        var_analysis = monthly_variance(filtered_df, variance_col)
        
        if var_analysis.empty:
            return create_empty_figure("No variance data available")
        
        # Create date column with fix for datetime warning
        var_analysis['Date'] = pd.to_datetime(
            var_analysis['Year'].astype(str) + '-' + 
            var_analysis['Num_Month'].astype(str) + '-01',
            format='%Y-%m-%d',
            errors='coerce'
        )
        
        var_analysis = var_analysis.dropna(subset=['Date']).sort_values('Date')
        
        # Convert to numpy array to avoid plotly warning
        dates = np.array(var_analysis['Date'].dt.to_pydatetime())
        
        # Create visualization
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Monthly Variance Trend', 'Quarter-over-Quarter Change'),
            vertical_spacing=0.1
        )
        
        # Add traces
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=var_analysis['Monthly_change'],
                mode='lines+markers',
                name='Monthly Change %',
                line=dict(color='#1f77b4', width=3),
                marker=dict(size=8)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=var_analysis['qaurter_change'],
                mode='lines+markers',
                name='QoQ Change %',
                line=dict(color='#ff7f0e', width=3),
                marker=dict(size=8)
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            title=f"Trend Analysis - {title_suffix}",
            height=600,
            showlegend=True,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating trend visualization: {e}")
        return create_empty_figure(f"Error: {str(e)}")

def create_summary_visualization(master_df, selected_shape, selected_color, selected_bucket,
                               selected_month, selected_year):
    """Create stable summary visualization"""
    try:
        # Apply filters
        filtered_df = master_df.copy()
        title_parts = []
        
        mask = pd.Series(True, index=filtered_df.index)
        
        if selected_shape and selected_shape != "None":
            mask &= (filtered_df['Shape key'] == selected_shape)
            title_parts.append(selected_shape)
        if selected_color and selected_color != "None":
            mask &= (filtered_df['Color Key'] == selected_color)
            title_parts.append(selected_color)
        if selected_bucket and selected_bucket != "None":
            mask &= (filtered_df['Buckets'] == selected_bucket)
            title_parts.append(selected_bucket)
        
        filtered_df = filtered_df[mask]
        title_suffix = " | ".join(title_parts) if title_parts else "All Data"
        
        if filtered_df.empty:
            return create_empty_figure("No data available for selected filters")
        
        # Group data
        summary_data = filtered_df.groupby(['Month', 'Year']).agg({
            'Avg Cost Total': 'mean',
            'Max Buying Price': 'mean',
            'Weight': 'sum',
            'Product Id': 'count'
        }).reset_index()
        
        if summary_data.empty:
            return create_empty_figure("No summary data available")
        
        # Create date column with fix for datetime warning
        summary_data['Num_Month'] = summary_data['Month'].map(month_map)
        summary_data = summary_data.dropna(subset=['Num_Month'])
        
        summary_data['Date'] = pd.to_datetime(
            summary_data['Year'].astype(str) + '-' + 
            summary_data['Num_Month'].astype(int).astype(str) + '-01',
            format='%Y-%m-%d',
            errors='coerce'
        )
        
        summary_data = summary_data.dropna(subset=['Date']).sort_values('Date')
        
        # Convert to numpy array to avoid plotly warning
        dates = np.array(summary_data['Date'].dt.to_pydatetime())
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Average Cost Trend', 'Max Buying Price Trend', 
                           'Total Weight', 'Product Count')
        )
        
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        
        # Add traces
        metrics = [
            ('Avg Cost Total', 1, 1, colors[0]),
            ('Max Buying Price', 1, 2, colors[1]),
            ('Weight', 2, 1, colors[2]),
            ('Product Id', 2, 2, colors[3])
        ]
        
        for metric, row, col, color in metrics:
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=summary_data[metric],
                    mode='lines+markers',
                    name=metric,
                    line=dict(color=color, width=2),
                    marker=dict(size=6),
                    showlegend=False
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            title=f"Summary Analytics - {title_suffix}",
            height=500,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating summary visualization: {e}")
        return create_empty_figure(f"Error: {str(e)}")

def create_empty_figure(message: str):
    """Helper to create empty figure with message"""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5, xanchor='center', yanchor='middle',
        showarrow=False, font=dict(size=16)
    )
    return fig

def main():
    """Optimized main application function with stable UI"""
    try:
        st.set_page_config(page_title="Yellow Diamond Dashboard", layout="wide")
        st.title("Yellow Diamond Dashboard")
        st.markdown("Upload Excel files to process multiple sheets and filter data.")
        
        # Initialize session state
        initialize_session_state()
        
        # Load master dataset once at startup
        if st.session_state.master_df.empty:
            with st.spinner("Loading master database..."):
                try:
                    master_df = load_cached_master_dataset()
                    if not master_df.empty:
                        st.session_state.master_df = apply_data_filters(master_df)
                        st.success("Master database loaded successfully!")
                        logger.info("Master database loaded successfully")
                    else:
                        st.warning("No existing master database found. Upload a file to create one.")
                except Exception as e:
                    st.error(f"Error loading master database: {str(e)}")
                    st.info("If you continue to see this error, try deleting the kunmings.pkl file and upload a fresh data file.")
                    logger.error(f"Error loading master database: {e}")
                    
                    # Add recovery option
                    if st.button("Clear Corrupted Database"):
                        try:
                            corrupted_file = Path(r"C:\Users\himabirla\Downloads\CostDashboard-main (3)\CostDashboard-main\src\kunmings.pkl")
                            if corrupted_file.exists():
                                corrupted_file.unlink()
                                st.success("Corrupted database cleared. Please upload a new file.")
                                st.rerun()
                        except Exception as clear_error:
                            st.error(f"Could not clear corrupted file: {str(clear_error)}")
                            logger.error(f"Error clearing corrupted file: {clear_error}")
        
        # Sidebar controls
        st.sidebar.header("Controls")
        
        # Display master database status
        if not st.session_state.master_df.empty:
            st.sidebar.success(f"Master DB: {len(st.session_state.master_df):,} records")
        else:
            st.sidebar.warning("No master database found")
        
        # File upload section - Fixed for older Streamlit versions
        uploaded_file = st.sidebar.file_uploader(
            "Upload Excel File",
            type=['xlsx', 'xls'],
            help="Upload an Excel file with multiple sheets"
        )
        
        # Process uploaded file
        if uploaded_file is not None and not st.session_state.data_processed:
            process_uploaded_file(uploaded_file)
        
        # Display upload history in sidebar
        display_upload_history()
        
        # Dashboard display
        if not st.session_state.master_df.empty:
            display_dashboard()
        else:
            st.info("No data in master database. Upload an Excel file to get started!")
        
        # Sidebar control buttons
        create_sidebar_controls(uploaded_file)
            
    except Exception as e:
        st.error(f"Critical application error: {str(e)}")
        logger.error(f"Critical application error: {e}")

def process_uploaded_file(uploaded_file):
    """Process uploaded file with stable progress tracking"""
    with st.spinner("Processing Excel file..."):
        try:
            file_size = getattr(uploaded_file, 'size', None)
            
            st.subheader("Updating Master Database")
            
            # Process file with optimized function
            new_data = optimized_populate_monthly_stock_sheet(uploaded_file)
            
            # Combine with existing data efficiently
            if not st.session_state.master_df.empty:
                combined_df = pd.concat([new_data, st.session_state.master_df], ignore_index=True)
            else:
                combined_df = new_data
            
            # Single groupby and filter operation
            st.session_state.master_df = optimize_dataframe_groupby(combined_df)
            st.session_state.master_df = apply_data_filters(st.session_state.master_df)
            
            # Save data
            optimized_save_data(st.session_state.master_df)
            
            st.session_state.data_processed = True
            
            # Update history with success status
            update_upload_history(uploaded_file, file_size, "Processed")
            
            st.success(f"Successfully processed: {uploaded_file.name}")
            st.info(f"Master database now contains {len(st.session_state.master_df):,} records")
            logger.info(f"Successfully processed file: {uploaded_file.name}")
            
            st.rerun()
            
        except Exception as e:
            # Update history with failed status
            update_upload_history(uploaded_file, getattr(uploaded_file, 'size', None), "Failed")
            st.error(f"Error processing file: {str(e)}")
            logger.error(f"Error processing file: {e}")

def update_upload_history(uploaded_file, file_size, status="Processed"):
    """Update upload history efficiently with status tracking"""
    try:
        history = load_upload_history()
        new_entry = {
            "filename": uploaded_file.name,
            "upload_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "file_size": file_size or 0,
            "status": status
        }
        history.insert(0, new_entry)
        history = history[:20]  # Keep last 20 entries
        save_upload_history(history)
        
        # Update session state to trigger UI refresh
        st.session_state.upload_history = history
        
    except Exception as e:
        logger.error(f"Error updating upload history: {e}")

def display_upload_history():
    """Display upload history in sidebar with production-ready styling"""
    try:
        # Load current history
        history = load_upload_history()
        
        if not history:
            st.sidebar.info("No upload history available")
            return
        
        # Create expandable upload history section
        with st.sidebar.expander("📁 Upload History", expanded=False):
            st.markdown("**Recent File Uploads**")
            
            # Display recent uploads (limit to 10 for UI clarity)
            recent_history = history[:10]
            
            for i, entry in enumerate(recent_history):
                # Create a container for each upload entry
                with st.container():
                    # Parse upload time
                    try:
                        upload_time = datetime.strptime(entry['upload_time'], "%Y-%m-%d %H:%M:%S")
                        time_str = upload_time.strftime("%m/%d %H:%M")
                        # Calculate time difference
                        time_diff = datetime.now() - upload_time
                        if time_diff.days > 0:
                            time_ago = f"{time_diff.days}d ago"
                        elif time_diff.seconds > 3600:
                            time_ago = f"{time_diff.seconds//3600}h ago"
                        elif time_diff.seconds > 60:
                            time_ago = f"{time_diff.seconds//60}m ago"
                        else:
                            time_ago = "Just now"
                    except:
                        time_str = entry.get('upload_time', 'Unknown')
                        time_ago = ""
                    
                    # Format file size
                    file_size = entry.get('file_size', 0)
                    size_str = format_file_size(file_size)
                    
                    # Status indicator
                    status = entry.get('status', 'Unknown')
                    if status == 'Processed':
                        status_emoji = "✅"
                        status_color = "#28a745"
                    elif status == 'Failed':
                        status_emoji = "❌"
                        status_color = "#dc3545"
                    else:
                        status_emoji = "⏳"
                        status_color = "#ffc107"
                    
                    # Display entry with enhanced styling
                    st.markdown(
                        f"""
                        <div style="
                            border: 1px solid #e1e5e9; 
                            padding: 10px; 
                            margin: 6px 0; 
                            border-radius: 6px; 
                            background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
                            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                        ">
                            <div style="
                                display: flex; 
                                align-items: center; 
                                font-weight: 600; 
                                color: #495057; 
                                font-size: 13px;
                                margin-bottom: 4px;
                            ">
                                <span style="margin-right: 6px;">{status_emoji}</span>
                                <span style="flex: 1; word-break: break-all;">{entry['filename']}</span>
                            </div>
                            <div style="
                                font-size: 11px; 
                                color: #6c757d; 
                                display: flex; 
                                justify-content: space-between;
                                margin-bottom: 3px;
                            ">
                                <span>📅 {time_str}</span>
                                <span style="color: #868e96;">{time_ago}</span>
                            </div>
                            <div style="
                                display: flex; 
                                justify-content: space-between; 
                                align-items: center;
                            ">
                                <span style="font-size: 11px; color: #6c757d;">📊 {size_str}</span>
                                <span style="
                                    font-size: 10px; 
                                    color: {status_color}; 
                                    font-weight: 500;
                                    background-color: {status_color}20;
                                    padding: 2px 6px;
                                    border-radius: 10px;
                                ">{status}</span>
                            </div>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
            
            # Summary statistics
            st.markdown("---")
            total_uploads = len(history)
            successful_uploads = len([h for h in history if h.get('status') == 'Processed'])
            failed_uploads = len([h for h in history if h.get('status') == 'Failed'])
            
            # Create metrics in columns
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total", total_uploads, delta=None)
            with col2:
                success_rate = (successful_uploads / total_uploads * 100) if total_uploads > 0 else 0
                st.metric("Success Rate", f"{success_rate:.0f}%", delta=None)
            
            if failed_uploads > 0:
                st.warning(f"⚠️ {failed_uploads} failed upload(s)")
            
            # Show more history option
            if len(history) > 10:
                st.info(f"Showing 10 of {len(history)} total uploads")
            
    except Exception as e:
        st.sidebar.error(f"Error displaying upload history: {str(e)}")
        logger.error(f"Error displaying upload history: {e}")

def format_file_size(size_bytes):
    """Format file size in human readable format"""
    if size_bytes == 0 or size_bytes is None:
        return "0 B"
    try:
        size_names = ["B", "KB", "MB", "GB"]
        i = int(np.floor(np.log(size_bytes) / np.log(1024)))
        p = pow(1024, i)
        s = round(size_bytes / p, 2)
        return f"{s} {size_names[i]}"
    except:
        return "Unknown"

def display_dashboard():
    """Display main dashboard with stable components"""
    try:
        # Create filter controls with stable keys
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        master_df = st.session_state.master_df
        
        # Get unique values with stable sorting
        with col1:
            months = ["None"] + sort_months_stable(tuple(master_df['Month'].unique()))
            selected_month = st.selectbox("Filter by Month", months, key="month_filter")
        with col2:
            years = ["None"] + sorted(list(master_df['Year'].unique()))
            selected_year = st.selectbox("Filter by Year", years, key="year_filter")
        with col3:
            shapes = ["None"] + sorted(list(master_df['Shape key'].unique()))
            selected_shape = st.selectbox("Filter by Shape", shapes, key="shape_filter")
        with col4:
            colors = ["None"] + ['WXYZ', 'FLY', 'FY', 'FIY', 'FVY']
            selected_color = st.selectbox("Filter by Color", colors, key="color_filter")
        with col5:
            buckets = ["None"] + sorted(list(stock_bucket.keys()))
            selected_bucket = st.selectbox("Filter by Bucket", buckets, key="bucket_filter")
        with col6:
            variance_columns = ["None", 'Current Average Cost', 'Max Buying Price', 'Min Selling Price']
            selected_variance_column = st.selectbox("Select Variance Column", variance_columns, key="variance_filter")
        
        # Create stable filter hash
        current_filter_hash = create_stable_filter_hash(selected_month, selected_year, selected_shape, selected_color, selected_bucket)
        
        # Only recalculate if filters changed
        if st.session_state.last_filter_hash != current_filter_hash:
            # Apply filters efficiently
            display_df = apply_stable_filters(master_df, selected_month, selected_year, selected_shape, selected_color, selected_bucket)
            st.session_state.stable_display_df = display_df
            st.session_state.last_filter_hash = current_filter_hash
        else:
            display_df = st.session_state.stable_display_df
        
        # Display summary metrics
        display_summary_metrics(display_df, selected_month, selected_year, selected_shape, 
                              selected_color, selected_bucket, selected_variance_column)
        
        # Display visualizations
        display_visualizations(master_df, selected_shape, selected_color, selected_bucket, 
                             selected_variance_column, selected_month, selected_year)
        
        # Display data table
        display_data_table(display_df)
        
        # Display download options
        display_download_options(display_df, master_df)
        
        # Display GAP summary with stable rendering - FIXED
        display_gap_summary_stable(master_df, selected_month, selected_year, selected_shape, selected_color, selected_bucket)
        
    except Exception as e:
        st.error(f"Error in dashboard display: {str(e)}")
        logger.error(f"Error in dashboard display: {e}")

def apply_stable_filters(df, month, year, shape, color, bucket):
    """Apply filters with stable output"""
    try:
        # Single filter operation
        mask = pd.Series(True, index=df.index)
        
        if month != "None":
            mask &= (df['Month'] == month)
        if year != "None":
            mask &= (df['Year'] == int(year))
        if shape != "None":
            mask &= (df['Shape key'] == shape)
        if color != "None":
            mask &= (df['Color Key'] == color)
        if bucket != "None":
            mask &= (df['Buckets'] == bucket)
        
        return df[mask].reset_index(drop=True)
    except Exception as e:
        logger.error(f"Error applying stable filters: {e}")
        return df

def display_summary_metrics(display_df, selected_month, selected_year, selected_shape, 
                          selected_color, selected_bucket, selected_variance_column):
    """Display summary metrics with stable layout"""
    st.subheader("Summary Metrics")
    
    # Check if all filters are selected
    all_filters_selected = all(
        filter_val != "None" 
        for filter_val in [selected_month, selected_year, selected_shape, selected_color, selected_bucket]
    )
    
    if all_filters_selected:
        display_detailed_metrics(display_df, selected_month, selected_year, selected_shape,
                               selected_color, selected_bucket, selected_variance_column)
    else:
        display_aggregated_metrics(display_df)

def display_detailed_metrics(filter_data, selected_month, selected_year, selected_shape,
                           selected_color, selected_bucket, selected_variance_column):
    """Display detailed metrics for specific filters"""
    try:
        # Get filtered data efficiently
        filtered_results = optimized_get_filtered_data(
            selected_month, selected_year, selected_shape, selected_color, selected_bucket)
        
        filter_data, max_buying_price, current_avg_cost, gap_output, min_selling_price = filtered_results
        
        # Get summary metrics
        mom_variance, mom_percent_change, mom_qoq_percent_change = optimized_get_summary_metrics(
            filter_data, selected_month, selected_shape, selected_year,
            selected_color, selected_bucket, selected_variance_column)
        
        # Display metrics in stable columns
        metric_cols = st.columns(7)
        
        metrics_data = [
            ("Gap Analysis", f"{gap_output}"),
            ("Max Buying Price", f"${max_buying_price:,.2f}"),
            ("Min Selling Price", f"${min_selling_price:,.2f}"),
            ("Current Avg Cost", f"${current_avg_cost:,.2f}"),
            ("MOM Variance", f"{mom_variance:,.2f}%"),
            ("MOM % Change", f"{mom_percent_change:.2f}%"),
            ("QoQ % Change", f"{mom_qoq_percent_change:.2f}%")
        ]
        
        for i, (label, value) in enumerate(metrics_data):
            with metric_cols[i]:
                st.metric(label, value)
                
    except Exception as e:
        st.error(f"Error calculating detailed metrics: {str(e)}")

def display_aggregated_metrics(display_df):
    """Display aggregated metrics"""
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
            
            st.info("Select all filters to view detailed metrics including Gap Analysis.")
        except Exception as e:
            st.error(f"Error calculating aggregated metrics: {str(e)}")
    else:
        st.warning("No data available for selected filters")

def display_gap_summary_stable(master_df, selected_month, selected_year, selected_shape, selected_color, selected_bucket):
    """Display GAP summary with stable rendering - FIXED to avoid pickle errors"""
    st.subheader("GAP Summary")
    
    try:
        # Create stable filter hash
        current_filter_hash = create_stable_filter_hash(selected_month, selected_year, selected_shape, selected_color, selected_bucket)
        
        # Only recalculate if filters changed
        if (st.session_state.cached_gap_summary_hash != current_filter_hash or 
            st.session_state.cached_gap_summary.empty):
            
            # Calculate GAP summary
            gap_summary_df = get_gap_summary_table(
                master_df, selected_month, selected_year, selected_shape, selected_color, selected_bucket
            )
            
            # Cache the results
            st.session_state.cached_gap_summary = gap_summary_df
            st.session_state.cached_gap_summary_hash = current_filter_hash
            
            # Create HTML representation instead of pandas Styler (to avoid pickle issues)
            if not gap_summary_df.empty:
                st.session_state.gap_summary_display_html = create_gap_table_html(gap_summary_df)
            else:
                st.session_state.gap_summary_display_html = "<p>No data available</p>"
        
        # Use cached data
        gap_summary_df = st.session_state.cached_gap_summary
        
        if not gap_summary_df.empty:
            # Create stable container for the table
            gap_container = st.container()
            
            with gap_container:
                # Display using either HTML or regular dataframe
                if st.session_state.gap_summary_display_html:
                    # Use HTML rendering for styled table
                    st.components.v1.html(
                        st.session_state.gap_summary_display_html,
                        height=400,
                        scrolling=True
                    )
                else:
                    # Fallback to regular dataframe
                    st.dataframe(
                        gap_summary_df,
                        use_container_width=True,
                        hide_index=True,
                        key=f"gap_table_fallback_{current_filter_hash}"
                    )
                
                # Download options with stable keys
                display_gap_download_options(gap_summary_df, current_filter_hash)
        else:
            st.info("No GAP data available for current filters.")
            
    except Exception as e:
        st.error(f"Error generating GAP summary: {str(e)}")
        logger.error(f"Error in GAP summary display: {e}")

def display_gap_download_options(gap_summary_df, filter_hash):
    """Display download options for GAP summary with stable keys"""
    st.subheader("Download GAP Summary")
    
    try:
        # Prepare download data
        gap_csv = gap_summary_df.to_csv(index=False)
        excess_df = gap_summary_df[gap_summary_df['Status'] == 'Excess']
        need_df = gap_summary_df[gap_summary_df['Status'] == 'Need']
        
        col_gap1, col_gap2, col_gap3 = st.columns(3)
        
        with col_gap1:
            st.download_button(
                "Download GAP Summary",
                data=gap_csv,
                file_name=f"gap_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key=f"download_gap_summary_{filter_hash}"
            )
        
        with col_gap2:
            if not excess_df.empty:
                excess_csv = excess_df.to_csv(index=False)
                st.download_button(
                    "Download GAP Excess",
                    data=excess_csv,
                    file_name=f"gap_excess_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    key=f"download_gap_excess_{filter_hash}"
                )
            else:
                st.button("Download GAP Excess (No Data)", disabled=True, key=f"gap_excess_disabled_{filter_hash}")
        
        with col_gap3:
            if not need_df.empty:
                need_csv = need_df.to_csv(index=False)
                st.download_button(
                    "Download GAP Need",
                    data=need_csv,
                    file_name=f"gap_need_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    key=f"download_gap_need_{filter_hash}"
                )
            else:
                st.button("Download GAP Need (No Data)", disabled=True, key=f"gap_need_disabled_{filter_hash}")
                
    except Exception as e:
        st.error(f"Error preparing download options: {str(e)}")

def display_visualizations(master_df, selected_shape, selected_color, selected_bucket, 
                         selected_variance_column, selected_month, selected_year):
    """Display visualizations with stable rendering"""
    st.subheader("Trend Analysis")
    
    tab1, tab2 = st.tabs(["Variance Trends", "Summary Analytics"])
    
    with tab1:
        if selected_variance_column != "None":
            try:
                trend_fig = create_trend_visualization(
                    master_df, selected_shape, selected_color, selected_bucket,
                    selected_variance_column, selected_month, selected_year
                )
                st.plotly_chart(trend_fig, use_container_width=True, key="trend_chart")
            except Exception as e:
                st.error(f"Error creating trend visualization: {str(e)}")
        else:
            st.info("Please select a variance column to view trend analysis.")
    
    with tab2:
        try:
            summary_fig = create_summary_visualization(
                master_df, selected_shape, selected_color, selected_bucket,
                selected_month, selected_year
            )
            st.plotly_chart(summary_fig, use_container_width=True, key="summary_chart")
        except Exception as e:
            st.error(f"Error creating summary visualization: {str(e)}")

def display_data_table(display_df):
    """Display data table with stable rendering"""
    st.subheader("Data Table")
    
    try:
        if not display_df.empty:
            # Add pagination for large datasets
            if len(display_df) > 1000:
                st.info(f"Large dataset detected ({len(display_df):,} rows). Showing first 1000 rows.")
                display_df = display_df.head(1000)
            
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No data available for selected filters")
    except Exception as e:
        st.error(f"Error displaying data table: {str(e)}")

def display_download_options(display_df, master_df):
    """Display download options with stable keys"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Download Filtered Data")
        try:
            if not display_df.empty:
                download_columns = ['Product Id', 'Shape key', 'Color Key', 'Avg Cost Total',
                                  'Min Qty', 'Max Qty', 'Buying Price Avg', 'Max Buying Price']
                available_columns = [col for col in download_columns if col in display_df.columns]
                
                download_df = display_df[available_columns]
                csv = download_df.to_csv(index=False)
                
                st.download_button(
                    label="Download Filtered Data as CSV",
                    data=csv,
                    file_name=f"filtered_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    key="download_filtered_data"
                )
                st.info(f"Records: {len(download_df):,}")
            else:
                st.info("No filtered data available for download")
        except Exception as e:
            st.error(f"Error preparing download: {str(e)}")
    
    with col2:
        st.subheader("Download Master Data")
        try:
            if not master_df.empty:
                csv = master_df.to_csv(index=False)
                
                st.download_button(
                    label="Download Master Data as CSV",
                    data=csv,
                    file_name=f"master_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    key="download_master_data"
                )
                st.info(f"Records: {len(master_df):,}")
            else:
                st.info("No master data available for download")
        except Exception as e:
            st.error(f"Error preparing master download: {str(e)}")

def create_sidebar_controls(uploaded_file):
    """Create sidebar controls with stable behavior"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("System Controls")
    
    # Control buttons with better spacing
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("🔄 Reset Process", key="reset_processing", help="Reset file processing state"):
            st.session_state.data_processed = False
            if uploaded_file is not None:
                st.session_state.master_df = load_cached_master_dataset()
            st.rerun()
    
    with col2:
        if st.button("🧹 Clear Cache", key="clear_cache", help="Clear all cached data"):
            st.cache_data.clear()
            # Clear session state caches
            st.session_state.cached_gap_summary = pd.DataFrame()
            st.session_state.cached_gap_summary_hash = None
            st.session_state.gap_summary_display_html = None
            st.success("Cache cleared!")
            st.rerun()
    
    # History management
    if st.button("🗑️ Clear Upload History", key="clear_history", help="Clear all upload history"):
        save_upload_history([])
        st.session_state.upload_history = []
        st.success("Upload history cleared!")
        st.rerun()
    
    # Additional system information
    st.sidebar.markdown("---")
    st.sidebar.markdown("**System Status**")
    
    # Display memory usage info
    try:
        history = load_upload_history()
        st.sidebar.info(f"Upload History: {len(history)} entries")
    except:
        st.sidebar.warning("Upload History: Error loading")
    
    # Cache information
    st.sidebar.info("Cache: Active for performance")

if __name__ == "__main__":
    main()
