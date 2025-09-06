# Optimized utility.py with pre-computed mappings and efficient data structures

import pandas as pd
import numpy as np
from functools import lru_cache

# Pre-computed mappings for better performance
stock_bucket = {
    '0.50-0.69': (0.5, 0.7),
    '0.70-0.89': (0.7, 0.9),
    '0.90-0.99': (0.9, 1),
    '1.00-1.25': (1, 1.26),
    '1.26-1.49': (1.26, 1.5),
    '1.50-1.74': (1.5, 1.75),
    '1.75 - 1.99': (1.75, 2),
    '2.00-2.49': (2, 2.5),
    '2.50-2.99': (2.5, 3),
    '3.00-3.49': (3, 3.5),
    '3.50-3.99': (3.5, 4),
    '4.00-4.99': (4, 5),
    '5.00-7.99': (5, 8),
    '8.00-9.99': (8, 10),
    '10.00-14.99': (10, 15)
}

month_map = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4, 
    'May': 5, 'June': 6, 'July': 7, 'August': 8, 
    'September': 9, 'October': 10, 'November': 11, 'December': 12
}

color_map = {
    'Fancy Intense Yellow': 'FIY',
    'Fancy Yellow': 'FY',
    'Fancy Light Yellow': 'FLY',
    'Fancy Vivid Yellow': 'FVY',
    'W-X': 'WXYZ',
    'Y-Z': 'WXYZ'
}


# Pre-computed reverse mappings for faster lookups
@lru_cache(maxsize=1)
def get_reverse_month_map():
    """Cached reverse month mapping"""
    return {v: k for k, v in month_map.items()}

@lru_cache(maxsize=1)
def get_reverse_color_map():
    """Cached reverse color mapping"""
    return {v: k for k, v in color_map.items()}

# Optimized bucket lookup functions
@lru_cache(maxsize=1000)
def get_weight_bucket(weight):
    """Cached weight bucket lookup"""
    try:
        weight = float(weight)
        for bucket_name, (lower, upper) in stock_bucket.items():
            if lower <= weight < upper:
                return bucket_name
        return 'Other'
    except:
        return 'Other'

# Pre-computed quarter mappings
quarter_map = {
    'January': 'Q1', 'February': 'Q1', 'March': 'Q1',
    'April': 'Q2', 'May': 'Q2', 'June': 'Q2',
    'July': 'Q3', 'August': 'Q3', 'September': 'Q3',
    'October': 'Q4', 'November': 'Q4', 'December': 'Q4'
}

@lru_cache(maxsize=12)
def get_quarter_for_month(month):
    """Cached quarter lookup"""
    try:
        year = pd.to_datetime('today').year
        yr = year - 2000
        quarter = quarter_map.get(month, 'Q1')
        return f'{quarter}-{yr}'
    except:
        return None

# Vectorized shape key mapping with pre-computed patterns
shape_patterns = {
    'Cushion': ['CUSHION', 'MODIFIED RECTANGULAR', 'MODIFIED SQUARE', 'ROUND_CORNERED'],
    'Oval': ['OVAL'],
    'Pear': ['PEAR'],
    'Radiant': ['CUT-CORNERED']
}

@lru_cache(maxsize=1000)
def get_shape_key(shape_str):
    """Cached shape key lookup"""
    try:
        if pd.isna(shape_str):
            return 'Other'
        
        shape_upper = str(shape_str).upper()
        
        for shape_key, patterns in shape_patterns.items():
            if any(pattern in shape_upper for pattern in patterns):
                return shape_key
        
        return 'Other'
    except:
        return 'Other'

# Optimized filtering constants
FILTER_CONDITIONS = {
    'color_exclude': ['U-V', 'S-T', 'Fancy Deep Yellow'],
    'weight_min': 0.5,
    'shape_exclude': ['Other']
}

# Pre-computed data validation rules
REQUIRED_COLUMNS = {
    'monthly_stock': ['Product Id', 'Weight', 'Color', 'Shape'],
    'min_data': ['Months', 'Buckets'],
    'max_data': ['Months', 'Buckets'],
    'buying_prices': ['Months', 'Buckets'],
    'selling_prices': ['Buckets']
}

@lru_cache(maxsize=100)
def validate_dataframe_structure(df_hash, df_type):
    """Cached dataframe structure validation"""
    required_cols = REQUIRED_COLUMNS.get(df_type, [])
    return all(col in df_hash for col in required_cols)

# Optimized data type conversions
DTYPE_CONVERSIONS = {
    'Product Id': 'str',
    'Weight': 'float64',
    'Year': 'int64',
    'Max Qty': 'float64',
    'Min Qty': 'float64',
    'Max Buying Price': 'float64',
    'Min Selling Price': 'float64'
}

def optimize_dtypes(df):
    """Optimize dataframe data types for better performance"""
    try:
        df = df.copy()
        for col, dtype in DTYPE_CONVERSIONS.items():
            if col in df.columns:
                if dtype == 'str':
                    df[col] = df[col].astype(str)
                elif dtype in ['float64', 'int64']:
                    df[col] = pd.to_numeric(df[col], errors='coerce').astype(dtype)
        return df
    except Exception as e:
        print(f"Error optimizing dtypes: {e}")
        return df

# Memory-efficient constants
MEMORY_EFFICIENT_COLUMNS = [
    'Product Id', 'Weight', 'Color', 'Shape', 'Month', 'Year',
    'Max Qty', 'Min Qty', 'Max Buying Price', 'Min Selling Price',
    'Avg Cost Total', 'Shape key', 'Color Key', 'Buckets'
]

# Pre-computed filter masks for common operations
@lru_cache(maxsize=100)
def get_filter_mask_template(filter_type):
    """Get pre-computed filter mask templates"""
    templates = {
        'color_filter': lambda df: ~df['Color'].isin(FILTER_CONDITIONS['color_exclude']),
        'weight_filter': lambda df: df['Weight'] >= FILTER_CONDITIONS['weight_min'],
        'shape_filter': lambda df: df['Shape key'] != 'Other'
    }
    return templates.get(filter_type)

# Optimized calculation helpers
def safe_divide_vectorized(numerator_series, denominator_series, default=0):
    """Vectorized safe division"""
    try:
        result = numerator_series / denominator_series
        result = result.fillna(default)
        result = result.replace([np.inf, -np.inf], default)
        return result
    except:
        return pd.Series(default, index=numerator_series.index)

def calculate_percentage_change_vectorized(series, default=0):
    """Vectorized percentage change calculation"""
    try:
        pct_change = series.pct_change().fillna(default) * 100
        pct_change = pct_change.replace([np.inf, -np.inf], default)
        return pct_change.round(2)
    except:
        return pd.Series(default, index=series.index)

# Batch processing utilities
class BatchProcessor:
    """Utility class for batch processing operations"""
    
    @staticmethod
    def process_shape_keys_batch(shapes_series):
        """Process shape keys in batch for better performance"""
        try:
            # Vectorized string operations
            shapes_upper = shapes_series.str.upper().fillna('')
            result = pd.Series('Other', index=shapes_series.index)
            
            # Apply all conditions at once
            for shape_key, patterns in shape_patterns.items():
                pattern_regex = '|'.join(patterns)
                mask = shapes_upper.str.contains(pattern_regex, na=False)
                result.loc[mask] = shape_key
            
            return result
        except:
            return pd.Series('Other', index=shapes_series.index)
    
    @staticmethod
    def process_color_keys_batch(colors_series):
        """Process color keys in batch"""
        try:
            return colors_series.map(color_map).fillna('')
        except:
            return pd.Series('', index=colors_series.index)
    
    @staticmethod
    def process_buckets_batch(weights_series):
        """Process weight buckets in batch"""
        try:
            result = pd.Series('Other', index=weights_series.index)
            weights = weights_series.fillna(0)
            
            for bucket_name, (lower, upper) in stock_bucket.items():
                mask = (weights >= lower) & (weights < upper)
                result.loc[mask] = bucket_name
            
            return result
        except:
            return pd.Series('Other', index=weights_series.index)

# Performance monitoring utilities
class PerformanceTracker:
    """Track performance metrics for optimization"""
    
    def __init__(self):
        self.metrics = {}
    
    def start_timer(self, operation_name):
        """Start timing an operation"""
        import time
        self.metrics[operation_name] = {'start_time': time.time()}
    
    def end_timer(self, operation_name):
        """End timing and log results"""
        import time
        if operation_name in self.metrics:
            duration = time.time() - self.metrics[operation_name]['start_time']
            self.metrics[operation_name]['duration'] = duration
            print(f"Operation '{operation_name}' took {duration:.3f} seconds")
    
    def get_summary(self):
        """Get performance summary"""
        return {op: data.get('duration', 0) for op, data in self.metrics.items()}

# Memory optimization utilities
def optimize_memory_usage(df):
    """Optimize memory usage of dataframe"""
    try:
        df = df.copy()
        
        # Optimize numeric columns
        for col in df.select_dtypes(include=[np.number]).columns:
            col_min = df[col].min()
            col_max = df[col].max()
            
            if col_min >= 0:  # Unsigned integers
                if col_max < 255:
                    df[col] = df[col].astype(np.uint8)
                elif col_max < 65535:
                    df[col] = df[col].astype(np.uint16)
                elif col_max < 4294967295:
                    df[col] = df[col].astype(np.uint32)
            else:  # Signed integers
                if col_min > np.iinfo(np.int8).min and col_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif col_min > np.iinfo(np.int16).min and col_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif col_min > np.iinfo(np.int32).min and col_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
        
        # Optimize object columns
        for col in df.select_dtypes(include=[object]).columns:
            num_unique_values = len(df[col].unique())
            num_total_values = len(df[col])
            if num_unique_values / num_total_values < 0.5:  # Less than 50% unique values
                df[col] = df[col].astype('category')
        
        return df
    except Exception as e:
        print(f"Error optimizing memory usage: {e}")
        return df

# Fast lookup tables
@lru_cache(maxsize=1)
def get_fast_lookup_tables():
    """Create fast lookup tables for common operations"""
    return {
        'month_to_num': month_map,
        'num_to_month': get_reverse_month_map(),
        'color_to_key': color_map,
        'key_to_color': get_reverse_color_map(),
        'quarter_lookup': quarter_map
    }

# Bulk data processing utilities
def process_multiple_dataframes(dataframes_dict, operations_list):
    """Process multiple dataframes with same operations efficiently"""
    try:
        results = {}
        
        for df_name, df in dataframes_dict.items():
            if df is None or df.empty:
                results[df_name] = df
                continue
            
            processed_df = df.copy()
            
            # Apply operations in sequence
            for operation in operations_list:
                if callable(operation):
                    processed_df = operation(processed_df)
                elif isinstance(operation, dict):
                    # Apply dictionary-based operations
                    func = operation.get('function')
                    kwargs = operation.get('kwargs', {})
                    if func and callable(func):
                        processed_df = func(processed_df, **kwargs)
            
            results[df_name] = processed_df
        
        return results
    except Exception as e:
        print(f"Error in bulk processing: {e}")
        return dataframes_dict

# Configuration for different environments
CONFIG = {
    'development': {
        'cache_ttl': 300,  # 5 minutes
        'chunk_size': 1000,
        'max_memory_usage': '1GB'
    },
    'production': {
        'cache_ttl': 3600,  # 1 hour
        'chunk_size': 10000,
        'max_memory_usage': '4GB'
    }
}

# Fast validation functions
@lru_cache(maxsize=100)
def validate_month_year(month, year):
    """Fast validation of month/year combinations"""
    try:
        valid_months = list(month_map.keys())
        valid_years = range(2020, 2030)  # Adjust range as needed
        return month in valid_months and int(year) in valid_years
    except:
        return False

@lru_cache(maxsize=1000)
def validate_filter_combination(shape, color, bucket):
    """Fast validation of filter combinations"""
    try:
        valid_shapes = list(shape_patterns.keys()) + ['Other']
        valid_colors = list(color_map.values())
        valid_buckets = list(stock_bucket.keys())
        
        return (shape in valid_shapes and 
                color in valid_colors and 
                bucket in valid_buckets)
    except:
        return False

# Memory-efficient data structures
class EfficientDataCache:
    """Memory-efficient data caching with LRU eviction"""
    
    def __init__(self, max_size=10):
        self.cache = {}
        self.access_order = []
        self.max_size = max_size
    
    def get(self, key):
        """Get item from cache"""
        if key in self.cache:
            # Move to end (most recently used)
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None
    
    def put(self, key, value):
        """Put item in cache with LRU eviction"""
        if key in self.cache:
            # Update existing
            self.access_order.remove(key)
        elif len(self.cache) >= self.max_size:
            # Evict least recently used
            lru_key = self.access_order.pop(0)
            del self.cache[lru_key]
        
        self.cache[key] = value
        self.access_order.append(key)
    
    def clear(self):
        """Clear cache"""
        self.cache.clear()
        self.access_order.clear()

# Create global cache instance
data_cache = EfficientDataCache(max_size=20)

# Optimized calculation functions
def calculate_metrics_vectorized(df, metrics_list):
    """Calculate multiple metrics in a vectorized manner"""
    try:
        results = {}
        
        for metric in metrics_list:
            if metric == 'avg_cost_total':
                results[metric] = (df['Weight'].fillna(0) * df['Average\nCost\n(USD)'].fillna(0)).sum()
            elif metric == 'total_weight':
                results[metric] = df['Weight'].fillna(0).sum()
            elif metric == 'product_count':
                results[metric] = len(df)
            elif metric == 'avg_buying_price':
                results[metric] = df['Max Buying Price'].fillna(0).mean()
            elif metric == 'avg_selling_price':
                results[metric] = df['Min Selling Price'].fillna(0).mean()
        
        return results
    except Exception as e:
        print(f"Error calculating vectorized metrics: {e}")
        return {metric: 0 for metric in metrics_list}

# Bulk filter application
def apply_bulk_filters(df, filter_config):
    """Apply multiple filters efficiently in a single pass"""
    try:
        if df is None or df.empty:
            return df
        
        # Start with all True mask
        mask = pd.Series(True, index=df.index)
        
        # Apply all filters using boolean indexing
        for filter_name, filter_value in filter_config.items():
            if filter_value == "None" or filter_value is None:
                continue
            
            if filter_name == 'Month' and 'Month' in df.columns:
                mask &= (df['Month'] == filter_value)
            elif filter_name == 'Year' and 'Year' in df.columns:
                mask &= (df['Year'] == int(filter_value))
            elif filter_name == 'Shape' and 'Shape key' in df.columns:
                mask &= (df['Shape key'] == filter_value)
            elif filter_name == 'Color' and 'Color Key' in df.columns:
                mask &= (df['Color Key'] == filter_value)
            elif filter_name == 'Bucket' and 'Buckets' in df.columns:
                mask &= (df['Buckets'] == filter_value)
        
        return df[mask].reset_index(drop=True)
    except Exception as e:
        print(f"Error applying bulk filters: {e}")
        return df

# Optimized aggregation functions
def fast_groupby_agg(df, group_cols, agg_dict):
    """Fast groupby aggregation with optimizations"""
    try:
        if df.empty:
            return df
        
        # Use categorical data types for grouping columns if beneficial
        df_copy = df.copy()
        for col in group_cols:
            if col in df_copy.columns and df_copy[col].dtype == 'object':
                unique_ratio = len(df_copy[col].unique()) / len(df_copy)
                if unique_ratio < 0.5:  # Less than 50% unique values
                    df_copy[col] = df_copy[col].astype('category')
        
        # Perform groupby with optimized settings
        result = df_copy.groupby(group_cols, as_index=False, sort=False).agg(agg_dict)
        
        return result
    except Exception as e:
        print(f"Error in fast groupby: {e}")
        return df

# Export optimized functions
__all__ = [
    'stock_bucket', 'month_map', 'color_map', 'quarter_map',
    'get_weight_bucket', 'get_shape_key', 'get_quarter_for_month',
    'BatchProcessor', 'EfficientDataCache', 'data_cache',
    'optimize_dtypes', 'apply_bulk_filters', 'fast_groupby_agg',
    'calculate_metrics_vectorized', 'safe_divide_vectorized',
    'calculate_percentage_change_vectorized', 'optimize_memory_usage'
]
