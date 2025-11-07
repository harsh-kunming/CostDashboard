import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple
from datetime import datetime
from dataclasses import dataclass
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

# ===== STOCK TREND ANALYZER FUNCTIONS =====
@st.cache_data(ttl=3600, show_spinner=False)
def analyze_product_stock_trends_cached(df_csv_string: str, criteria_hash: str):
    """Cached version of stock trend analysis"""
    try:
        from io import StringIO
        df = pd.read_csv(StringIO(df_csv_string))
        
        # Parse criteria from session state
        fast_moving_criteria = st.session_state.get('trend_criteria', {
            'min_activity_months': 5,
            'min_avg_cost': 3000,
            'min_total_value': 10000,
            'price_volatility': 0.2
        })
        
        # Ensure required columns exist
        if 'Average\nCost\n(USD)' in df.columns:
            df['Average_Cost_USD'] = df['Average\nCost\n(USD)']
        elif 'Avg Cost Total' in df.columns and 'Weight' in df.columns:
            # Calculate average cost from available data
            df['Average_Cost_USD'] = df['Avg Cost Total'] / df['Weight'].replace(0, np.nan)
        
        # Ensure numeric data types
        numeric_columns = ['Average_Cost_USD', 'Weight', 'Max Buying Price', 'Min Selling Price']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Create date column for trend analysis
        if 'Year' in df.columns and 'Month' in df.columns:
            df['Date'] = pd.to_datetime(
                df['Year'].astype(str) + '-' + df['Month'], 
                format='%Y-%B', 
                errors='coerce'
            )
        
        # Group by Product ID and calculate metrics
        product_analysis = df.groupby('Product Id').agg({
            'Average_Cost_USD': ['mean', 'std', 'count'],
            'Weight': 'mean',
            'Month': 'nunique',
            'Year': 'nunique',
            'Max Buying Price': 'mean',
            'Min Selling Price': 'mean',
            'Shape key': lambda x: x.mode()[0] if not x.empty else 'Unknown',
            'Color Key': lambda x: x.mode()[0] if not x.empty else 'Unknown',
            'Buckets': lambda x: x.mode()[0] if not x.empty else 'Unknown'
        }).reset_index()
        
        # Flatten column names
        product_analysis.columns = [
            'Product_ID', 'Avg_Cost', 'Cost_StdDev', 'Record_Count',
            'Avg_Weight', 'Active_Months', 'Active_Years',
            'Avg_Max_Buying_Price', 'Avg_Min_Selling_Price',
            'Primary_Shape', 'Primary_Color', 'Primary_Bucket'
        ]
        
        # Calculate additional metrics
        product_analysis['Total_Value'] = (
            product_analysis['Avg_Cost'] * 
            product_analysis['Avg_Weight'] * 
            product_analysis['Record_Count']
        )
        
        product_analysis['Price_Volatility'] = (
            product_analysis['Cost_StdDev'] / product_analysis['Avg_Cost']
        ).fillna(0)
        
        product_analysis['Profit_Margin'] = (
            (product_analysis['Avg_Min_Selling_Price'] - product_analysis['Avg_Max_Buying_Price']) / 
            product_analysis['Avg_Max_Buying_Price'] * 100
        ).fillna(0)
        
        # Calculate price trend for each product
        def calculate_price_trend(product_id):
            """Calculate if product price is trending up, down, or stable"""
            product_data = df[df['Product Id'] == product_id].copy()
            if len(product_data) < 2:
                return 'Insufficient Data'
            
            product_data = product_data.sort_values('Date')
            prices = product_data['Average_Cost_USD'].values
            
            if len(prices) > 1 and not np.all(np.isnan(prices)):
                x = np.arange(len(prices))
                valid_idx = ~np.isnan(prices)
                if np.sum(valid_idx) > 1:
                    try:
                        slope = np.polyfit(x[valid_idx], prices[valid_idx], 1)[0]
                        avg_price = np.nanmean(prices)
                        
                        if avg_price != 0:
                            normalized_slope = slope / avg_price
                            
                            if normalized_slope > 0.05:
                                return 'Upward'
                            elif normalized_slope < -0.05:
                                return 'Downward'
                    except:
                        pass
            
            return 'Stable'
        
        # Add price trend
        product_analysis['Price_Trend'] = product_analysis['Product_ID'].apply(calculate_price_trend)
        
        # Calculate movement score
        def calculate_movement_score(row):
            """Calculate movement score based on multiple factors"""
            score = 0
            
            # Activity score (0-25)
            if row['Active_Months'] >= fast_moving_criteria['min_activity_months']:
                score += 25
            else:
                score += (row['Active_Months'] / fast_moving_criteria['min_activity_months']) * 25
            
            # Value score (0-25)
            if row['Avg_Cost'] >= fast_moving_criteria['min_avg_cost']:
                score += 25
            else:
                score += (row['Avg_Cost'] / fast_moving_criteria['min_avg_cost']) * 25
            
            # Total value score (0-25)
            if row['Total_Value'] >= fast_moving_criteria['min_total_value']:
                score += 25
            else:
                score += (row['Total_Value'] / fast_moving_criteria['min_total_value']) * 25
            
            # Volatility score (0-25)
            if row['Price_Volatility'] >= fast_moving_criteria['price_volatility']:
                score += 25
            else:
                score += (row['Price_Volatility'] / fast_moving_criteria['price_volatility']) * 25
            
            return min(score, 100)
        
        # Add movement score
        product_analysis['Movement_Score'] = product_analysis.apply(calculate_movement_score, axis=1)
        
        # Categorize products based on score
        def categorize_product(score):
            if score >= 70:
                return 'Fast Moving'
            elif score >= 40:
                return 'Moderate Moving'
            else:
                return 'Slow Moving'
        
        product_analysis['Category'] = product_analysis['Movement_Score'].apply(categorize_product)
        
        # Sort by movement score (descending)
        product_analysis = product_analysis.sort_values('Movement_Score', ascending=False)
        
        # Round numerical columns for readability
        product_analysis['Avg_Cost'] = product_analysis['Avg_Cost'].round(2)
        product_analysis['Total_Value'] = product_analysis['Total_Value'].round(2)
        product_analysis['Price_Volatility'] = product_analysis['Price_Volatility'].round(3)
        product_analysis['Avg_Weight'] = product_analysis['Avg_Weight'].round(2)
        product_analysis['Movement_Score'] = product_analysis['Movement_Score'].round(1)
        product_analysis['Profit_Margin'] = product_analysis['Profit_Margin'].round(2)
        product_analysis['Avg_Max_Buying_Price'] = product_analysis['Avg_Max_Buying_Price'].round(2)
        product_analysis['Avg_Min_Selling_Price'] = product_analysis['Avg_Min_Selling_Price'].round(2)
        
        # Get lists by category
        fast_moving = product_analysis[
            product_analysis['Category'] == 'Fast Moving'
        ]['Product_ID'].tolist()
        
        moderate_moving = product_analysis[
            product_analysis['Category'] == 'Moderate Moving'
        ]['Product_ID'].tolist()
        
        slow_moving = product_analysis[
            product_analysis['Category'] == 'Slow Moving'
        ]['Product_ID'].tolist()
        
        # Calculate summary statistics
        summary_stats = {
            'total_products': len(product_analysis),
            'fast_moving_count': len(fast_moving),
            'moderate_moving_count': len(moderate_moving),
            'slow_moving_count': len(slow_moving),
            'fast_moving_percentage': round((len(fast_moving) / len(product_analysis)) * 100, 2) if len(product_analysis) > 0 else 0,
            'moderate_moving_percentage': round((len(moderate_moving) / len(product_analysis)) * 100, 2) if len(product_analysis) > 0 else 0,
            'avg_cost_all': round(product_analysis['Avg_Cost'].mean(), 2),
            'avg_cost_fast': round(
                product_analysis[product_analysis['Category'] == 'Fast Moving']['Avg_Cost'].mean(), 2
            ) if len(fast_moving) > 0 else 0,
            'avg_cost_slow': round(
                product_analysis[product_analysis['Category'] == 'Slow Moving']['Avg_Cost'].mean(), 2
            ) if len(slow_moving) > 0 else 0,
            'total_value_all': round(product_analysis['Total_Value'].sum(), 2),
            'total_value_fast': round(
                product_analysis[product_analysis['Category'] == 'Fast Moving']['Total_Value'].sum(), 2
            ) if len(fast_moving) > 0 else 0,
            'total_value_slow': round(
                product_analysis[product_analysis['Category'] == 'Slow Moving']['Total_Value'].sum(), 2
            ) if len(slow_moving) > 0 else 0,
            'avg_movement_score': round(product_analysis['Movement_Score'].mean(), 1),
            'upward_trend_count': len(product_analysis[product_analysis['Price_Trend'] == 'Upward']),
            'downward_trend_count': len(product_analysis[product_analysis['Price_Trend'] == 'Downward']),
            'stable_trend_count': len(product_analysis[product_analysis['Price_Trend'] == 'Stable'])
        }
        
        return {
            'summary_stats': summary_stats,
            'product_analysis': product_analysis,
            'fast_moving_products': fast_moving,
            'moderate_moving_products': moderate_moving,
            'slow_moving_products': slow_moving
        }
        
    except Exception as e:
        logger.error(f"Error analyzing stock trends: {e}")
        return None

def create_trend_analysis_visualizations(analysis_results, filters):
    """Create visualizations for trend analysis"""
    if not analysis_results:
        return None, None, None, None
    
    df = analysis_results['product_analysis']
    
    # Apply filters
    if filters.get('shape') and filters['shape'] != 'All':
        df = df[df['Primary_Shape'] == filters['shape']]
    if filters.get('color') and filters['color'] != 'All':
        df = df[df['Primary_Color'] == filters['color']]
    if filters.get('bucket') and filters['bucket'] != 'All':
        df = df[df['Primary_Bucket'] == filters['bucket']]
    if filters.get('category') and filters['category'] != 'All':
        df = df[df['Category'] == filters['category']]
    if filters.get('trend') and filters['trend'] != 'All':
        df = df[df['Price_Trend'] == filters['trend']]
    
    # Check if we have data after filtering
    if df.empty:
        # Return empty figure with message
        empty_fig = go.Figure()
        empty_fig.add_annotation(
            text="No data available for selected filters",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14)
        )
        return empty_fig, empty_fig, empty_fig, empty_fig
    
    # 1. Category Distribution
    category_counts = df.groupby('Category').size().reset_index(name='Count')
    if not category_counts.empty:
        category_fig = px.pie(
            category_counts,
            values='Count',
            names='Category',
            title='Product Category Distribution',
            color_discrete_map={
                'Fast Moving': '#2ecc71',
                'Moderate Moving': '#f39c12',
                'Slow Moving': '#e74c3c'
            }
        )
    else:
        category_fig = go.Figure()
        category_fig.add_annotation(text="No category data", xref="paper", yref="paper", x=0.5, y=0.5)
    
    # 2. Movement Score Distribution
    if len(df) > 0:
        score_fig = px.histogram(
            df,
            x='Movement_Score',
            nbins=min(20, len(df)),  # Adjust bins based on data size
            title='Movement Score Distribution',
            labels={'Movement_Score': 'Movement Score', 'count': 'Number of Products'},
            color_discrete_sequence=['#3498db']
        )
        score_fig.add_vline(x=70, line_dash="dash", line_color="green", annotation_text="Fast Moving Threshold")
        score_fig.add_vline(x=40, line_dash="dash", line_color="orange", annotation_text="Moderate Threshold")
    else:
        score_fig = go.Figure()
        score_fig.add_annotation(text="No score data", xref="paper", yref="paper", x=0.5, y=0.5)
    
    # 3. Price Trend Analysis
    trend_data = df.groupby(['Price_Trend', 'Category']).size().reset_index(name='Count')
    if not trend_data.empty:
        trend_fig = px.bar(
            trend_data,
            x='Price_Trend',
            y='Count',
            color='Category',
            title='Price Trends by Category',
            barmode='group',
            color_discrete_map={
                'Fast Moving': '#2ecc71',
                'Moderate Moving': '#f39c12',
                'Slow Moving': '#e74c3c'
            }
        )
    else:
        trend_fig = go.Figure()
        trend_fig.add_annotation(text="No trend data", xref="paper", yref="paper", x=0.5, y=0.5)
    
    # 4. Top Products Analysis (Scatter) - dynamically adjust to available products
    num_products = min(50, len(df))
    if num_products > 0:
        top_products_fig = px.scatter(
            df.head(num_products),
            x='Total_Value',
            y='Movement_Score',
            size='Avg_Weight',
            color='Category',
            hover_data=['Product_ID', 'Avg_Cost', 'Price_Trend', 'Profit_Margin'],
            title=f'Top {num_products} Products: Value vs Movement Score',
            color_discrete_map={
                'Fast Moving': '#2ecc71',
                'Moderate Moving': '#f39c12',
                'Slow Moving': '#e74c3c'
            }
        )
    else:
        top_products_fig = go.Figure()
        top_products_fig.add_annotation(text="No product data", xref="paper", yref="paper", x=0.5, y=0.5)
    
    return category_fig, score_fig, trend_fig, top_products_fig

def display_trend_analysis_section(master_df):
    """Display the trend analysis section with filters and results"""
    st.header("📈 Stock Trend Analysis")
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["Analysis Dashboard", "Detailed Results", "Visualizations", "Export & Reports"])
    
    with tab1:
        # Analysis criteria settings
        st.subheader("Analysis Criteria")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            min_activity_months = st.number_input(
                "Min Activity Months",
                min_value=1,
                max_value=12,
                value=5,
                help="Minimum months of activity for fast-moving classification",
                key="trend_min_activity_months"
            )
        
        with col2:
            min_avg_cost = st.number_input(
                "Min Average Cost ($)",
                min_value=0,
                max_value=100000,
                value=3000,
                step=500,
                help="Minimum average cost threshold",
                key="trend_min_avg_cost"
            )
        
        with col3:
            min_total_value = st.number_input(
                "Min Total Value ($)",
                min_value=0,
                max_value=1000000,
                value=10000,
                step=1000,
                help="Minimum total value threshold",
                key="trend_min_total_value"
            )
        
        with col4:
            price_volatility = st.slider(
                "Price Volatility",
                min_value=0.0,
                max_value=1.0,
                value=0.2,
                step=0.05,
                help="Price volatility threshold (0-1)",
                key="trend_price_volatility"
            )
        
        # Store criteria in session state
        st.session_state.trend_criteria = {
            'min_activity_months': min_activity_months,
            'min_avg_cost': min_avg_cost,
            'min_total_value': min_total_value,
            'price_volatility': price_volatility
        }
        
        # Create criteria hash for caching
        criteria_hash = hashlib.md5(
            f"{min_activity_months}_{min_avg_cost}_{min_total_value}_{price_volatility}".encode()
        ).hexdigest()
        
        # Run analysis button
        if st.button("🔍 Run Trend Analysis", type="primary"):
            with st.spinner("Analyzing product trends..."):
                # Convert dataframe to CSV string for caching
                df_csv_string = master_df.to_csv(index=False)
                
                # Run cached analysis
                analysis_results = analyze_product_stock_trends_cached(df_csv_string, criteria_hash)
                
                if analysis_results:
                    st.session_state.trend_analysis_results = analysis_results
                    st.success("Analysis completed successfully!")
                else:
                    st.error("Error running analysis. Please check your data.")
        
        # Display summary if analysis has been run
        if 'trend_analysis_results' in st.session_state and st.session_state.trend_analysis_results:
            st.markdown("---")
            st.subheader("📊 Analysis Summary")
            
            results = st.session_state.trend_analysis_results
            stats = results['summary_stats']
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Products", f"{stats['total_products']:,}")
                st.metric("Avg Movement Score", f"{stats['avg_movement_score']:.1f}")
            
            with col2:
                st.metric("Fast Moving", f"{stats['fast_moving_count']} ({stats['fast_moving_percentage']:.1f}%)")
                st.metric("Avg Cost (Fast)", f"${stats['avg_cost_fast']:,.2f}")
            
            with col3:
                st.metric("Moderate Moving", f"{stats['moderate_moving_count']} ({stats['moderate_moving_percentage']:.1f}%)")
                st.metric("Total Value (All)", f"${stats['total_value_all']:,.2f}")
            
            with col4:
                st.metric("Slow Moving", f"{stats['slow_moving_count']} ({100-stats['fast_moving_percentage']-stats['moderate_moving_percentage']:.1f}%)")
                st.metric("Avg Cost (Slow)", f"${stats['avg_cost_slow']:,.2f}")
            
            # Trend summary
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("📈 Upward Trend", stats['upward_trend_count'])
            with col2:
                st.metric("📉 Downward Trend", stats['downward_trend_count'])
            with col3:
                st.metric("➡️ Stable Trend", stats['stable_trend_count'])
    
    with tab2:
        if 'trend_analysis_results' in st.session_state and st.session_state.trend_analysis_results:
            st.subheader("Detailed Product Analysis")
            
            results = st.session_state.trend_analysis_results
            df_analysis = results['product_analysis']
            
            # Filters for detailed view
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                shape_filter = st.selectbox(
                    "Filter by Shape",
                    ['All'] + sorted(df_analysis['Primary_Shape'].unique().tolist()),
                    key="trend_detail_shape_filter"
                )
            
            with col2:
                color_filter = st.selectbox(
                    "Filter by Color",
                    ['All'] + sorted(df_analysis['Primary_Color'].unique().tolist()),
                    key="trend_detail_color_filter"
                )
            
            with col3:
                bucket_filter = st.selectbox(
                    "Filter by Bucket",
                    ['All'] + sorted(df_analysis['Primary_Bucket'].unique().tolist()),
                    key="trend_detail_bucket_filter"
                )
            
            with col4:
                category_filter = st.selectbox(
                    "Filter by Category",
                    ['All', 'Fast Moving', 'Moderate Moving', 'Slow Moving'],
                    key="trend_detail_category_filter"
                )
            
            with col5:
                trend_filter = st.selectbox(
                    "Filter by Price Trend",
                    ['All', 'Upward', 'Downward', 'Stable', 'Insufficient Data'],
                    key="trend_detail_trend_filter"
                )
            
            # Apply filters
            filtered_df = df_analysis.copy()
            
            if shape_filter != 'All':
                filtered_df = filtered_df[filtered_df['Primary_Shape'] == shape_filter]
            if color_filter != 'All':
                filtered_df = filtered_df[filtered_df['Primary_Color'] == color_filter]
            if bucket_filter != 'All':
                filtered_df = filtered_df[filtered_df['Primary_Bucket'] == bucket_filter]
            if category_filter != 'All':
                filtered_df = filtered_df[filtered_df['Category'] == category_filter]
            if trend_filter != 'All':
                filtered_df = filtered_df[filtered_df['Price_Trend'] == trend_filter]
            
            # Display options
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.info(f"Showing {len(filtered_df)} of {len(df_analysis)} products")
            
            with col2:
                # Dynamically set min and default values based on filtered data
                if len(filtered_df) > 0:
                    min_val = min(10, len(filtered_df))
                    max_val = len(filtered_df)
                    default_val = min(50, len(filtered_df))
                    show_top_n = st.number_input(
                        "Show Top N", 
                        min_value=min_val, 
                        max_value=max_val, 
                        value=default_val,
                        step=10,
                        key="trend_detail_show_top_n"
                    )
                else:
                    show_top_n = 0
                    st.warning("No products match the selected filters")
            
            # Display table with conditional formatting (only if data exists)
            if len(filtered_df) > 0 and show_top_n > 0:
                display_columns = [
                    'Product_ID', 'Category', 'Movement_Score', 'Price_Trend',
                    'Avg_Cost', 'Total_Value', 'Active_Months', 'Price_Volatility',
                    'Profit_Margin', 'Primary_Shape', 'Primary_Color', 'Primary_Bucket'
                ]
                
                # Show filtered and limited data
                display_df = filtered_df[display_columns].head(show_top_n)
                
                # Style the dataframe
                def color_category(val):
                    if val == 'Fast Moving':
                        return 'background-color: #d4edda'
                    elif val == 'Moderate Moving':
                        return 'background-color: #fff3cd'
                    elif val == 'Slow Moving':
                        return 'background-color: #f8d7da'
                    return ''
                
                def color_trend(val):
                    if val == 'Upward':
                        return 'color: green'
                    elif val == 'Downward':
                        return 'color: red'
                    return ''
                
                styled_df = display_df.style.applymap(
                    color_category, subset=['Category']
                ).applymap(
                    color_trend, subset=['Price_Trend']
                )
                
                st.dataframe(styled_df, use_container_width=True, hide_index=True)
            elif len(filtered_df) == 0:
                st.warning("No products match the selected filter criteria. Try adjusting your filters.")
            
        else:
            st.info("Please run the trend analysis first to see detailed results.")
    
    with tab3:
        if 'trend_analysis_results' in st.session_state and st.session_state.trend_analysis_results:
            st.subheader("Trend Analysis Visualizations")
            
            df_analysis = st.session_state.trend_analysis_results['product_analysis']
            
            # Create filter controls for visualizations
            viz_col1, viz_col2, viz_col3, viz_col4, viz_col5 = st.columns(5)
            
            with viz_col1:
                viz_shape_filter = st.selectbox(
                    "Filter by Shape",
                    ['All'] + sorted(df_analysis['Primary_Shape'].unique().tolist()),
                    key="viz_shape_filter"
                )
            
            with viz_col2:
                viz_color_filter = st.selectbox(
                    "Filter by Color",
                    ['All'] + sorted(df_analysis['Primary_Color'].unique().tolist()),
                    key="viz_color_filter"
                )
            
            with viz_col3:
                viz_bucket_filter = st.selectbox(
                    "Filter by Bucket",
                    ['All'] + sorted(df_analysis['Primary_Bucket'].unique().tolist()),
                    key="viz_bucket_filter"
                )
            
            with viz_col4:
                viz_category_filter = st.selectbox(
                    "Filter by Category",
                    ['All', 'Fast Moving', 'Moderate Moving', 'Slow Moving'],
                    key="viz_category_filter"
                )
            
            with viz_col5:
                viz_trend_filter = st.selectbox(
                    "Filter by Price Trend",
                    ['All', 'Upward', 'Downward', 'Stable', 'Insufficient Data'],
                    key="viz_trend_filter"
                )
            
            # Use visualization filters
            filters = {
                'shape': viz_shape_filter,
                'color': viz_color_filter,
                'bucket': viz_bucket_filter,
                'category': viz_category_filter,
                'trend': viz_trend_filter
            }
            
            # Create visualizations
            category_fig, score_fig, trend_fig, top_products_fig = create_trend_analysis_visualizations(
                st.session_state.trend_analysis_results, filters
            )
            
            # Display visualizations in 2x2 grid
            col1, col2 = st.columns(2)
            
            with col1:
                if category_fig:
                    st.plotly_chart(category_fig, use_container_width=True)
                if trend_fig:
                    st.plotly_chart(trend_fig, use_container_width=True)
            
            with col2:
                if score_fig:
                    st.plotly_chart(score_fig, use_container_width=True)
                if top_products_fig:
                    st.plotly_chart(top_products_fig, use_container_width=True)
            
            # Additional analysis charts
            st.markdown("---")
            st.subheader("Deep Dive Analysis")
            
            # Apply filters to df_analysis for additional charts
            filtered_analysis = df_analysis.copy()
            if viz_shape_filter != 'All':
                filtered_analysis = filtered_analysis[filtered_analysis['Primary_Shape'] == viz_shape_filter]
            if viz_color_filter != 'All':
                filtered_analysis = filtered_analysis[filtered_analysis['Primary_Color'] == viz_color_filter]
            if viz_bucket_filter != 'All':
                filtered_analysis = filtered_analysis[filtered_analysis['Primary_Bucket'] == viz_bucket_filter]
            if viz_category_filter != 'All':
                filtered_analysis = filtered_analysis[filtered_analysis['Category'] == viz_category_filter]
            if viz_trend_filter != 'All':
                filtered_analysis = filtered_analysis[filtered_analysis['Price_Trend'] == viz_trend_filter]
            
            if len(filtered_analysis) > 0:
                # Profit margin by category
                profit_fig = px.box(
                    filtered_analysis,
                    x='Category',
                    y='Profit_Margin',
                    title='Profit Margin Distribution by Category',
                    color='Category',
                    color_discrete_map={
                        'Fast Moving': '#2ecc71',
                        'Moderate Moving': '#f39c12',
                        'Slow Moving': '#e74c3c'
                    }
                )
                
                # Weight vs Value scatter (limit to top 100 or available)
                display_count = min(100, len(filtered_analysis))
                weight_value_fig = px.scatter(
                    filtered_analysis.head(display_count),
                    x='Avg_Weight',
                    y='Total_Value',
                    color='Movement_Score',
                    size='Active_Months',
                    hover_data=['Product_ID', 'Category'],
                    title=f'Weight vs Total Value (Top {display_count} Products)',
                    color_continuous_scale='Viridis'
                )
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.plotly_chart(profit_fig, use_container_width=True)
                
                with col2:
                    st.plotly_chart(weight_value_fig, use_container_width=True)
            else:
                st.warning("No data available for the selected filters in Deep Dive Analysis")
            
        else:
            st.info("Please run the trend analysis first to see visualizations.")
    
    with tab4:
        if 'trend_analysis_results' in st.session_state and st.session_state.trend_analysis_results:
            st.subheader("Export & Reports")
            
            results = st.session_state.trend_analysis_results
            df_analysis = results['product_analysis']
            
            # Report generation options
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📥 Download Options")
                
                # Full analysis report
                full_csv = df_analysis.to_csv(index=False)
                st.download_button(
                    "📊 Download Full Analysis (CSV)",
                    data=full_csv,
                    file_name=f"trend_analysis_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
                # Fast moving products only
                fast_moving_df = df_analysis[df_analysis['Category'] == 'Fast Moving']
                if not fast_moving_df.empty:
                    fast_csv = fast_moving_df.to_csv(index=False)
                    st.download_button(
                        "🚀 Download Fast Moving Products",
                        data=fast_csv,
                        file_name=f"fast_moving_products_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                
                # Slow moving products only
                slow_moving_df = df_analysis[df_analysis['Category'] == 'Slow Moving']
                if not slow_moving_df.empty:
                    slow_csv = slow_moving_df.to_csv(index=False)
                    st.download_button(
                        "🐌 Download Slow Moving Products",
                        data=slow_csv,
                        file_name=f"slow_moving_products_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            
            with col2:
                st.markdown("### 📋 Summary Report")
                
                # Generate summary report
                stats = results['summary_stats']
                
                summary_text = f"""
STOCK TREND ANALYSIS REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

EXECUTIVE SUMMARY
================
Total Products Analyzed: {stats['total_products']:,}
Average Movement Score: {stats['avg_movement_score']:.1f}

CATEGORY BREAKDOWN
==================
Fast Moving Products: {stats['fast_moving_count']} ({stats['fast_moving_percentage']:.1f}%)
Moderate Moving Products: {stats['moderate_moving_count']} ({stats['moderate_moving_percentage']:.1f}%)
Slow Moving Products: {stats['slow_moving_count']} ({100-stats['fast_moving_percentage']-stats['moderate_moving_percentage']:.1f}%)

PRICE TRENDS
============
Upward Trending: {stats['upward_trend_count']} products
Downward Trending: {stats['downward_trend_count']} products
Stable: {stats['stable_trend_count']} products

FINANCIAL METRICS
=================
Total Portfolio Value: ${stats['total_value_all']:,.2f}
Fast Moving Value: ${stats['total_value_fast']:,.2f}
Slow Moving Value: ${stats['total_value_slow']:,.2f}

Average Cost (All Products): ${stats['avg_cost_all']:,.2f}
Average Cost (Fast Moving): ${stats['avg_cost_fast']:,.2f}
Average Cost (Slow Moving): ${stats['avg_cost_slow']:,.2f}

ANALYSIS CRITERIA
=================
Min Activity Months: {st.session_state.trend_criteria['min_activity_months']}
Min Average Cost: ${st.session_state.trend_criteria['min_avg_cost']:,.2f}
Min Total Value: ${st.session_state.trend_criteria['min_total_value']:,.2f}
Price Volatility Threshold: {st.session_state.trend_criteria['price_volatility']:.2f}

TOP 10 FAST MOVING PRODUCTS
============================
"""
                # Add top 10 products
                top_10 = df_analysis[df_analysis['Category'] == 'Fast Moving'].head(10)
                for idx, row in top_10.iterrows():
                    summary_text += f"\n{row['Product_ID']}: Score={row['Movement_Score']:.1f}, Value=${row['Total_Value']:,.2f}, Trend={row['Price_Trend']}"
                
                # Download summary report
                st.download_button(
                    "📄 Download Summary Report (TXT)",
                    data=summary_text,
                    file_name=f"trend_analysis_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
                
                # Show preview
                with st.expander("Preview Summary Report"):
                    st.text(summary_text)
            
            # Additional export options
            st.markdown("---")
            st.markdown("### 🎯 Action Items")
            
            # Generate actionable insights
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**🚀 Fast Moving Products**")
                fast_count = stats['fast_moving_count']
                st.success(f"Maintain stock levels for {fast_count} products")
                if fast_count > 0:
                    st.write("Top recommendations:")
                    for product in results['fast_moving_products'][:3]:
                        st.write(f"• {product}")
            
            with col2:
                st.markdown("**⚠️ Moderate Moving Products**")
                moderate_count = stats['moderate_moving_count']
                st.warning(f"Monitor {moderate_count} products closely")
                if moderate_count > 0:
                    st.write("Requires attention:")
                    for product in results['moderate_moving_products'][:3]:
                        st.write(f"• {product}")
            
            with col3:
                st.markdown("**🔴 Slow Moving Products**")
                slow_count = stats['slow_moving_count']
                st.error(f"Review strategy for {slow_count} products")
                if slow_count > 0:
                    st.write("Consider action for:")
                    for product in results['slow_moving_products'][:3]:
                        st.write(f"• {product}")
        else:
            st.info("Please run the trend analysis first to generate reports.")

# ===== END OF STOCK TREND ANALYZER FUNCTIONS =====

# ===== ADVANCED INVENTORY OPTIMIZATION SYSTEM =====

# Check for optimization libraries
try:
    from scipy.optimize import linprog
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.info("scipy not available, using basic optimization")

@dataclass
class OptimizationProduct:
    """Data class for product information in optimization"""
    product_id: str
    current_qty: float
    min_qty: float
    max_qty: float
    max_buying_price: float
    min_selling_price: float
    profit_per_unit: float
    requires_restock: bool
    shortage_qty: float
    shape: str = ""
    color: str = ""
    bucket: str = ""
    weight: float = 0.0
    month: str = ""
    year: int = 0

class InventoryOptimizer:
    """
    Advanced optimization model for inventory purchase recommendations using linear programming
    """
    
    def __init__(self, df: pd.DataFrame, budget: float = float('inf')):
        """Initialize the optimizer with DataFrame and budget."""
        self.budget = budget
        self.df = self._prepare_dataframe(df)
        self.products = []
        self.optimization_result = None
        
    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare and validate the dataframe for optimization."""
        df = df.copy()
        
        # Ensure required columns exist
        required_cols = ['Product Id', 'Max Qty', 'Min Qty', 'Max Buying Price', 'Min Selling Price']
        for col in required_cols:
            if col not in df.columns:
                logger.warning(f"Missing required column: {col}")
                df[col] = 0
        
        # Convert to numeric
        numeric_cols = ['Max Qty', 'Min Qty', 'Max Buying Price', 'Min Selling Price', 'Weight']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Add current quantity if not present (simulate as percentage of max)
        if 'Current Qty' not in df.columns:
            np.random.seed(42)
            # Simulate current quantities - some products low, some adequate
            df['Current Qty'] = df.apply(
                lambda row: np.random.uniform(0, row['Max Qty'] * 0.8) 
                if np.random.random() > 0.3 else np.random.uniform(0, row['Min Qty'] * 0.5),
                axis=1
            )
        
        return df
    
    def identify_restock_products(self, filters: Optional[Dict] = None) -> List[OptimizationProduct]:
        """
        Identify products that need restocking (current < min_qty).
        
        Args:
            filters: Optional filters for shape, color, bucket, etc.
        
        Returns:
            List of products needing restock
        """
        df_filtered = self.df.copy()
        
        # Apply filters if provided
        if filters:
            if filters.get('shape') and filters['shape'] != 'All':
                df_filtered = df_filtered[df_filtered['Shape key'] == filters['shape']]
            if filters.get('color') and filters['color'] != 'All':
                df_filtered = df_filtered[df_filtered['Color Key'] == filters['color']]
            if filters.get('bucket') and filters['bucket'] != 'All':
                df_filtered = df_filtered[df_filtered['Buckets'] == filters['bucket']]
        
        # Get latest data for each product
        latest_data = df_filtered.groupby('Product Id').last().reset_index()
        
        products = []
        
        for _, row in latest_data.iterrows():
            # Calculate profit per unit
            profit = row['Min Selling Price'] - row['Max Buying Price']
            
            # Get current quantity
            current = row.get('Current Qty', 0)
            min_required = row['Min Qty']
            
            # Check if restocking needed
            requires_restock = current < min_required
            shortage = max(0, min_required - current)
            
            # Only include products with positive profit
            if profit > 0:
                product = OptimizationProduct(
                    product_id=str(row['Product Id']),
                    current_qty=current,
                    min_qty=min_required,
                    max_qty=row['Max Qty'],
                    max_buying_price=row['Max Buying Price'],
                    min_selling_price=row['Min Selling Price'],
                    profit_per_unit=profit,
                    requires_restock=requires_restock,
                    shortage_qty=shortage,
                    shape=row.get('Shape key', ''),
                    color=row.get('Color Key', ''),
                    bucket=row.get('Buckets', ''),
                    weight=row.get('Weight', 0),
                    month=row.get('Month', ''),
                    year=row.get('Year', 0)
                )
                
                if requires_restock:
                    products.append(product)
        
        self.products = products
        return products
    
    def optimize_scipy(self, max_units_per_product: int = 100) -> Dict:
        """Optimize using scipy linear programming."""
        if not self.products:
            return {"status": "No products need restocking", "recommendations": []}
        
        n_products = len(self.products)
        
        # Objective: Maximize profit (minimize negative profit)
        c = [-p.profit_per_unit for p in self.products]
        
        # Constraints
        A_ub = []
        b_ub = []
        
        # Budget constraint
        if self.budget < float('inf'):
            A_ub.append([p.max_buying_price for p in self.products])
            b_ub.append(self.budget)
        
        # Bounds for each product
        bounds = []
        for product in self.products:
            min_purchase = product.shortage_qty  # At least fill shortage
            max_purchase = min(
                max_units_per_product,
                product.max_qty - product.current_qty  # Don't exceed max
            )
            bounds.append((max(0, min_purchase), max(0, max_purchase)))
        
        # Solve
        try:
            if len(A_ub) > 0:
                result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            else:
                result = linprog(c, bounds=bounds, method='highs')
            
            if result.success:
                recommendations = []
                total_cost = 0
                total_profit = 0
                
                for i, product in enumerate(self.products):
                    qty = result.x[i]
                    if qty > 0.1:
                        cost = qty * product.max_buying_price
                        profit = qty * product.profit_per_unit
                        total_cost += cost
                        total_profit += profit
                        
                        recommendations.append({
                            'product_id': product.product_id,
                            'purchase_qty': round(qty, 0),
                            'current_qty': round(product.current_qty, 0),
                            'min_qty': product.min_qty,
                            'shortage': round(product.shortage_qty, 0),
                            'unit_cost': product.max_buying_price,
                            'unit_profit': product.profit_per_unit,
                            'total_cost': cost,
                            'expected_profit': profit,
                            'roi': (profit / cost * 100) if cost > 0 else 0,
                            'shape': product.shape,
                            'color': product.color,
                            'bucket': product.bucket,
                            'urgency': self._get_urgency(product),
                            'category': self._categorize_recommendation(product, profit, cost)
                        })
                
                return {
                    'status': 'success',
                    'method': 'scipy_optimization',
                    'total_cost': total_cost,
                    'expected_profit': total_profit,
                    'roi': (total_profit / total_cost * 100) if total_cost > 0 else 0,
                    'recommendations': sorted(recommendations, key=lambda x: x['expected_profit'], reverse=True)
                }
        except Exception as e:
            logger.error(f"Scipy optimization failed: {e}")
        
        # Fallback to greedy
        return self.optimize_greedy(max_units_per_product)
    
    def optimize_greedy(self, max_units_per_product: int = 100) -> Dict:
        """Greedy algorithm fallback for optimization."""
        if not self.products:
            return {"status": "No products need restocking", "recommendations": []}
        
        # Sort by profit per unit and urgency
        sorted_products = sorted(
            self.products, 
            key=lambda x: (x.profit_per_unit, -x.shortage_qty), 
            reverse=True
        )
        
        recommendations = []
        total_cost = 0
        total_profit = 0
        remaining_budget = self.budget
        
        for product in sorted_products:
            # Calculate purchase quantity
            max_purchase = min(
                product.shortage_qty + min(10, product.shortage_qty * 0.5),  # Buffer
                max_units_per_product,
                product.max_qty - product.current_qty
            )
            
            if self.budget < float('inf'):
                affordable_units = remaining_budget / product.max_buying_price if product.max_buying_price > 0 else 0
                max_purchase = min(max_purchase, affordable_units)
            
            if max_purchase >= product.shortage_qty:
                purchase_qty = max(product.shortage_qty, max_purchase)
                
                cost = purchase_qty * product.max_buying_price
                profit = purchase_qty * product.profit_per_unit
                
                if remaining_budget >= cost:
                    total_cost += cost
                    total_profit += profit
                    remaining_budget -= cost
                    
                    recommendations.append({
                        'product_id': product.product_id,
                        'purchase_qty': round(purchase_qty, 0),
                        'current_qty': round(product.current_qty, 0),
                        'min_qty': product.min_qty,
                        'shortage': round(product.shortage_qty, 0),
                        'unit_cost': product.max_buying_price,
                        'unit_profit': product.profit_per_unit,
                        'total_cost': cost,
                        'expected_profit': profit,
                        'roi': (profit / cost * 100) if cost > 0 else 0,
                        'shape': product.shape,
                        'color': product.color,
                        'bucket': product.bucket,
                        'urgency': self._get_urgency(product),
                        'category': self._categorize_recommendation(product, profit, cost)
                    })
        
        return {
            'status': 'success',
            'method': 'greedy_algorithm',
            'total_cost': total_cost,
            'expected_profit': total_profit,
            'roi': (total_profit / total_cost * 100) if total_cost > 0 else 0,
            'remaining_budget': remaining_budget,
            'recommendations': recommendations
        }
    
    def _get_urgency(self, product: OptimizationProduct) -> str:
        """Determine urgency level for a product."""
        shortage_pct = (product.shortage_qty / product.min_qty * 100) if product.min_qty > 0 else 0
        
        if product.current_qty == 0:
            return 'CRITICAL'
        elif shortage_pct >= 70:
            return 'HIGH'
        elif shortage_pct >= 40:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _categorize_recommendation(self, product: OptimizationProduct, profit: float, cost: float) -> str:
        """Categorize recommendation type."""
        roi = (profit / cost * 100) if cost > 0 else 0
        
        if product.current_qty == 0:
            return 'Immediate Action'
        elif roi >= 40:
            return 'High ROI'
        elif cost <= 5000 and profit >= 1000:
            return 'Quick Win'
        elif product.shortage_qty > product.min_qty * 0.5:
            return 'Urgent Restock'
        else:
            return 'Standard Restock'
    
    def optimize(self, budget: Optional[float] = None, 
                filters: Optional[Dict] = None,
                max_units_per_product: int = 100) -> Dict:
        """Main optimization function."""
        if budget is not None:
            self.budget = budget
        
        # Identify products needing restock
        self.identify_restock_products(filters)
        
        # Use scipy if available, otherwise greedy
        if SCIPY_AVAILABLE:
            result = self.optimize_scipy(max_units_per_product)
        else:
            result = self.optimize_greedy(max_units_per_product)
        
        self.optimization_result = result
        return result
    
    def get_budget_scenarios(self, budget_levels: List[float]) -> Dict:
        """Analyze different budget scenarios."""
        scenarios = {}
        
        for budget in budget_levels:
            result = self.optimize(budget=budget)
            
            if result['status'] == 'success':
                scenarios[budget] = {
                    'total_cost': result['total_cost'],
                    'expected_profit': result['expected_profit'],
                    'roi': result['roi'],
                    'num_products': len(result['recommendations']),
                    'method': result.get('method', 'unknown')
                }
        
        return scenarios
    
    def get_critical_products(self) -> List[Dict]:
        """Get products with critical shortage."""
        if not self.products:
            self.identify_restock_products()
        
        critical = []
        for product in self.products:
            if product.current_qty == 0 or product.shortage_qty > product.min_qty * 0.7:
                critical.append({
                    'product_id': product.product_id,
                    'current_qty': product.current_qty,
                    'min_qty': product.min_qty,
                    'shortage': product.shortage_qty,
                    'profit_potential': product.shortage_qty * product.profit_per_unit,
                    'investment_needed': product.shortage_qty * product.max_buying_price,
                    'urgency': self._get_urgency(product)
                })
        
        return sorted(critical, key=lambda x: x['shortage'], reverse=True)

@st.cache_data(ttl=3600, show_spinner=False)
def optimize_inventory_cached(df_csv: str, budget: Optional[float], filters_str: str, max_units: int):
    """Cached version of inventory optimization."""
    from io import StringIO
    df = pd.read_csv(StringIO(df_csv))
    
    # Parse filters
    filters = json.loads(filters_str) if filters_str else None
    
    optimizer = InventoryOptimizer(df, budget if budget else float('inf'))
    result = optimizer.optimize(filters=filters, max_units_per_product=max_units)
    
    # Get additional insights
    critical_products = optimizer.get_critical_products()
    
    # Budget scenarios
    if budget and budget < float('inf'):
        scenarios = optimizer.get_budget_scenarios([
            budget * 0.5,
            budget * 0.75,
            budget,
            budget * 1.25,
            budget * 1.5
        ])
    else:
        scenarios = optimizer.get_budget_scenarios([25000, 50000, 100000, 200000, 500000])
    
    return result, critical_products, scenarios

def display_optimized_recommendations_section(master_df):
    """Display the optimized inventory recommendations section."""
    st.header("🔬 Optimized Inventory Recommendations")
    st.markdown("Mathematical optimization using linear programming for maximum profit")
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Optimization Dashboard",
        "📊 Budget Analysis", 
        "🔍 Product Details",
        "📈 Visualization",
        "📋 Export Reports"
    ])
    
    with tab1:
        st.subheader("Inventory Optimization Settings")
        
        # Configuration columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            budget_type = st.selectbox(
                "Budget Configuration",
                ["Unlimited Budget", "Fixed Budget", "Budget Range Analysis"],
                help="Choose budget constraint type",key='Budget_Config'
            )
            
            if budget_type == "Fixed Budget":
                budget = st.number_input(
                    "Budget Amount ($)",
                    min_value=1000.0,
                    max_value=1000000.0,
                    value=100000.0,
                    step=5000.0,
                    key="opt_fixed_budget"
                )
            elif budget_type == "Budget Range Analysis":
                budget_min = st.number_input("Min Budget", value=25000.0, step=5000.0,key="opt_budget_min")
                budget_max = st.number_input("Max Budget", value=200000.0, step=5000.0,key="opt_budget_max")
                budget = budget_max  # Use max for initial optimization
            else:
                budget = None
        
        with col2:
            max_units = st.number_input(
                "Max Units per Product",
                min_value=10,
                max_value=500,
                value=100,
                step=10,
                help="Maximum units to order per product",
                key="opt_max_units"
            )
        
        with col3:
            shape_filter = st.selectbox(
                "Filter by Shape",
                ['All'] + sorted(master_df['Shape key'].unique().tolist()) if 'Shape key' in master_df.columns else ['All'],
                help="Filter products by shape",
                key="opt_shape_filter"
            )
        
        with col4:
            color_filter = st.selectbox(
                "Filter by Color",
                ['All'] + sorted(master_df['Color Key'].unique().tolist()) if 'Color Key' in master_df.columns else ['All'],
                help="Filter products by color",
                key="opt_color_filter"
            )
        
        # Advanced filters
        with st.expander("Advanced Filters"):
            adv_col1, adv_col2, adv_col3 = st.columns(3)
            
            with adv_col1:
                bucket_filter = st.selectbox(
                    "Filter by Bucket",
                    ['All'] + sorted(master_df['Buckets'].unique().tolist()) if 'Buckets' in master_df.columns else ['All'],
                    key="opt_bucket_filter"
                )
            
            with adv_col2:
                min_roi = st.slider(
                    "Minimum ROI (%)",
                    min_value=0,
                    max_value=100,
                    value=10,
                    step=5,
                    key="opt_min_roi",
                    help="Filter recommendations by minimum ROI"
                )
            
            with adv_col3:
                optimization_method = st.radio(
                    "Optimization Method",
                    ["Auto (Best Available)", "Linear Programming", "Greedy Algorithm"],
                    help="Choose optimization algorithm", key="opt_method"
                )
        
        # Prepare filters
        filters = {
            'shape': shape_filter if shape_filter != 'All' else None,
            'color': color_filter if color_filter != 'All' else None,
            'bucket': bucket_filter if bucket_filter != 'All' else None
        }
        
        # Run optimization button
        if st.button("🚀 Run Optimization", type="primary", key="run_optimization"):
            with st.spinner("Running mathematical optimization..."):
                try:
                    # Prepare data for caching
                    df_csv = master_df.to_csv(index=False)
                    filters_str = json.dumps(filters)
                    
                    # Run optimization
                    result, critical_products, scenarios = optimize_inventory_cached(
                        df_csv, budget, filters_str, max_units
                    )
                    
                    # Store results
                    st.session_state.optimization_result = result
                    st.session_state.critical_products = critical_products
                    st.session_state.budget_scenarios = scenarios
                    
                    if result and result.get('status') == 'success':
                        st.success(f"Optimization complete! Method: {result.get('method', 'Unknown')}")
                    else:
                        st.warning("Optimization completed with warnings or no products need restocking")
                except Exception as e:
                    st.error(f"Error during optimization: {str(e)}")
                    logger.error(f"Optimization error: {e}")
        
        # Display results if available
        if 'optimization_result' in st.session_state and st.session_state.optimization_result:
            result = st.session_state.optimization_result
            
            if result and result.get('status') == 'success' and result.get('recommendations'):
                # Summary metrics
                st.markdown("---")
                st.subheader("📊 Optimization Results")
                
                metric_cols = st.columns(6)
                
                with metric_cols[0]:
                    st.metric(
                        "Total Investment",
                        f"${result.get('total_cost', 0):,.0f}",
                        help="Total capital required"
                    )
                
                with metric_cols[1]:
                    st.metric(
                        "Expected Profit",
                        f"${result.get('expected_profit', 0):,.0f}",
                        f"{result.get('roi', 0):.1f}% ROI"
                    )
                
                with metric_cols[2]:
                    recommendations = result.get('recommendations', [])
                    st.metric(
                        "Products",
                        len(recommendations),
                        help="Number of products to purchase"
                    )
                
                with metric_cols[3]:
                    critical_count = len([r for r in recommendations if r.get('urgency') == 'CRITICAL'])
                    st.metric(
                        "Critical Items",
                        critical_count,
                        help="Out of stock products"
                    )
                
                with metric_cols[4]:
                    if 'remaining_budget' in result:
                        st.metric(
                            "Remaining Budget",
                            f"${result.get('remaining_budget', 0):,.0f}",
                            help="Unused budget"
                        )
                    else:
                        st.metric("Method", result.get('method', 'Unknown').replace('_', ' ').title())
                
                with metric_cols[5]:
                    avg_roi = np.mean([r.get('roi', 0) for r in recommendations]) if recommendations else 0
                    st.metric(
                        "Avg Product ROI",
                        f"{avg_roi:.1f}%",
                        help="Average ROI per product"
                    )
                
                # Category breakdown
                st.markdown("---")
                st.subheader("📦 Recommendation Categories")
                
                categories = {}
                for rec in recommendations:
                    cat = rec.get('category', 'Standard')
                    if cat not in categories:
                        categories[cat] = {'count': 0, 'investment': 0, 'profit': 0}
                    categories[cat]['count'] += 1
                    categories[cat]['investment'] += rec.get('total_cost', 0)
                    categories[cat]['profit'] += rec.get('expected_profit', 0)
                
                if categories:
                    cat_cols = st.columns(len(categories))
                    for idx, (cat_name, cat_data) in enumerate(categories.items()):
                        with cat_cols[idx]:
                            st.markdown(f"**{cat_name}**")
                            st.markdown(f"Products: **{cat_data['count']}**")
                            st.markdown(f"Investment: **${cat_data['investment']:,.0f}**")
                            st.markdown(f"Profit: **${cat_data['profit']:,.0f}**")
                
                # Recommendations table
                st.markdown("---")
                st.subheader("🛒 Optimized Purchase Recommendations")
                
                # Apply ROI filter
                filtered_recs = [r for r in recommendations if r.get('roi', 0) >= min_roi]
                
                if filtered_recs:
                    rec_df = pd.DataFrame(filtered_recs)
                    
                    # Select columns that exist
                    display_cols = []
                    possible_cols = [
                        'product_id', 'urgency', 'category', 'current_qty', 'shortage',
                        'purchase_qty', 'unit_cost', 'unit_profit', 'total_cost',
                        'expected_profit', 'roi', 'shape', 'color'
                    ]
                    for col in possible_cols:
                        if col in rec_df.columns:
                            display_cols.append(col)
                    
                    if display_cols:
                        # Style the dataframe
                        def style_urgency(val):
                            if val == 'CRITICAL':
                                return 'background-color: #ffcdd2'
                            elif val == 'HIGH':
                                return 'background-color: #ffe0b2'
                            elif val == 'MEDIUM':
                                return 'background-color: #fff9c4'
                            return ''
                        
                        styled_df = rec_df[display_cols].style
                        
                        if 'urgency' in display_cols:
                            styled_df = styled_df.applymap(style_urgency, subset=['urgency'])
                        
                        # Format numeric columns
                        format_dict = {}
                        if 'unit_cost' in display_cols:
                            format_dict['unit_cost'] = '${:.2f}'
                        if 'unit_profit' in display_cols:
                            format_dict['unit_profit'] = '${:.2f}'
                        if 'total_cost' in display_cols:
                            format_dict['total_cost'] = '${:,.2f}'
                        if 'expected_profit' in display_cols:
                            format_dict['expected_profit'] = '${:,.2f}'
                        if 'roi' in display_cols:
                            format_dict['roi'] = '{:.1f}%'
                        if 'current_qty' in display_cols:
                            format_dict['current_qty'] = '{:.0f}'
                        if 'shortage' in display_cols:
                            format_dict['shortage'] = '{:.0f}'
                        if 'purchase_qty' in display_cols:
                            format_dict['purchase_qty'] = '{:.0f}'
                        
                        if format_dict:
                            styled_df = styled_df.format(format_dict)
                        
                        st.dataframe(styled_df, use_container_width=True, hide_index=True, height=400)
                    else:
                        st.dataframe(rec_df, use_container_width=True, hide_index=True, height=400)
                else:
                    st.info(f"No recommendations meet the minimum ROI threshold of {min_roi}%")
            elif result and result.get('status') != 'success':
                st.warning(f"Optimization status: {result.get('status', 'Unknown')}")
                if result.get('message'):
                    st.info(result.get('message'))
            else:
                st.info("No products currently need restocking based on the selected filters")
        else:
            st.info("Click 'Run Optimization' to generate recommendations")
    
    with tab2:
        st.subheader("📊 Budget Scenario Analysis")
        
        if 'budget_scenarios' in st.session_state and st.session_state.budget_scenarios:
            scenarios = st.session_state.budget_scenarios
            
            if scenarios:
                # Create comparison table
                scenario_data = []
                for budget_amt, metrics in scenarios.items():
                    scenario_data.append({
                        'Budget': f"${budget_amt:,.0f}" if isinstance(budget_amt, (int, float)) else budget_amt,
                        'Investment': f"${metrics.get('total_cost', 0):,.0f}",
                        'Profit': f"${metrics.get('expected_profit', 0):,.0f}",
                        'ROI': f"{metrics.get('roi', 0):.1f}%",
                        'Products': metrics.get('num_products', 0)
                    })
                
                if scenario_data:
                    scenario_df = pd.DataFrame(scenario_data)
                    st.dataframe(scenario_df, use_container_width=True, hide_index=True)
                    
                    # Visualization
                    st.markdown("---")
                    
                    # ROI by budget chart
                    budgets = list(scenarios.keys())
                    rois = [metrics.get('roi', 0) for metrics in scenarios.values()]
                    profits = [metrics.get('expected_profit', 0) for metrics in scenarios.values()]
                    
                    fig = make_subplots(
                        rows=1, cols=2,
                        subplot_titles=('ROI by Budget', 'Profit by Budget')
                    )
                    
                    fig.add_trace(
                        go.Bar(x=[f"${b:,.0f}" if isinstance(b, (int, float)) else b for b in budgets], 
                              y=rois, name='ROI %'),
                        row=1, col=1
                    )
                    
                    fig.add_trace(
                        go.Bar(x=[f"${b:,.0f}" if isinstance(b, (int, float)) else b for b in budgets], 
                              y=profits, name='Profit $'),
                        row=1, col=2
                    )
                    
                    fig.update_layout(height=400, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Optimal budget recommendation
                    st.markdown("---")
                    st.subheader("💡 Budget Recommendation")
                    
                    # Find best ROI scenario
                    if scenarios:
                        best_roi_budget = max(scenarios.items(), key=lambda x: x[1].get('roi', 0))
                        st.success(f"**Optimal Budget for ROI**: ${best_roi_budget[0]:,.0f} → {best_roi_budget[1].get('roi', 0):.1f}% ROI")
                        
                        # Find best profit scenario
                        best_profit_budget = max(scenarios.items(), key=lambda x: x[1].get('expected_profit', 0))
                        st.info(f"**Optimal Budget for Profit**: ${best_profit_budget[0]:,.0f} → ${best_profit_budget[1].get('expected_profit', 0):,.0f} profit")
        else:
            st.info("Run optimization first to see budget analysis")
    
    with tab3:
        st.subheader("🔍 Product Deep Dive")
        
        if 'optimization_result' in st.session_state:
            result = st.session_state.optimization_result
            if result and result.get('recommendations'):
                recommendations = result.get('recommendations', [])
                
                if recommendations:
                    # Product selector
                    product_ids = [r.get('product_id', 'Unknown') for r in recommendations]
                    selected_product = st.selectbox(
                        "Select Product for Analysis",
                        product_ids,
                        format_func=lambda x: f"{x} ({next((r.get('urgency', '') for r in recommendations if r.get('product_id') == x), '')})",
                        key="opt_product_selector"
                    )
                    
                    if selected_product:
                        # Get product details
                        product = next((r for r in recommendations if r.get('product_id') == selected_product), None)
                        
                        if product:
                            # Display details
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.markdown("### 📦 Stock Information")
                                st.markdown(f"**Current Stock**: {product.get('current_qty', 0):.0f} units")
                                st.markdown(f"**Min Required**: {product.get('min_qty', 0):.0f} units")
                                st.markdown(f"**Shortage**: {product.get('shortage', 0):.0f} units")
                                st.markdown(f"**Purchase Qty**: {product.get('purchase_qty', 0):.0f} units")
                                st.markdown(f"**Urgency**: {product.get('urgency', 'Unknown')}")
                            
                            with col2:
                                st.markdown("### 💰 Financial Analysis")
                                st.markdown(f"**Unit Cost**: ${product.get('unit_cost', 0):,.2f}")
                                st.markdown(f"**Unit Profit**: ${product.get('unit_profit', 0):,.2f}")
                                st.markdown(f"**Total Investment**: ${product.get('total_cost', 0):,.2f}")
                                st.markdown(f"**Expected Profit**: ${product.get('expected_profit', 0):,.2f}")
                                st.markdown(f"**ROI**: {product.get('roi', 0):.1f}%")
                            
                            with col3:
                                st.markdown("### 📋 Product Details")
                                st.markdown(f"**Category**: {product.get('category', 'Unknown')}")
                                st.markdown(f"**Shape**: {product.get('shape', 'N/A')}")
                                st.markdown(f"**Color**: {product.get('color', 'N/A')}")
                                st.markdown(f"**Bucket**: {product.get('bucket', 'N/A')}")
                else:
                    st.info("No recommendations available")
            else:
                st.info("Run optimization first to analyze products")
        else:
            st.info("Run optimization first to analyze products")
    
    with tab4:
        st.subheader("📈 Optimization Visualizations")
        
        if 'optimization_result' in st.session_state:
            result = st.session_state.optimization_result
            if result and result.get('recommendations'):
                recommendations = result.get('recommendations', [])
                
                if recommendations:
                    # ROI Distribution
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # ROI histogram
                        roi_values = [r.get('roi', 0) for r in recommendations]
                        
                        if roi_values:
                            fig = px.histogram(
                                x=roi_values,
                                nbins=min(20, len(roi_values)),
                                title='ROI Distribution',
                                labels={'x': 'ROI (%)', 'y': 'Number of Products'}
                            )
                            fig.add_vline(x=np.mean(roi_values), line_dash="dash", annotation_text="Average")
                            st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Investment vs Profit scatter
                        fig = px.scatter(
                            x=[r.get('total_cost', 0) for r in recommendations],
                            y=[r.get('expected_profit', 0) for r in recommendations],
                            size=[r.get('purchase_qty', 1) for r in recommendations],
                            color=[r.get('urgency', 'Unknown') for r in recommendations],
                            hover_data={'Product': [r.get('product_id', '') for r in recommendations]},
                            title='Investment vs Profit',
                            labels={'x': 'Investment ($)', 'y': 'Expected Profit ($)'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No data available for visualization")
            else:
                st.info("Run optimization first to see visualizations")
        else:
            st.info("Run optimization first to see visualizations")
    
    with tab5:
        st.subheader("📋 Export Reports")
        
        if 'optimization_result' in st.session_state:
            result = st.session_state.optimization_result
            if result and result.get('recommendations'):
                recommendations = result.get('recommendations', [])
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 📥 Download Options")
                    
                    # Full recommendations
                    rec_df = pd.DataFrame(recommendations)
                    csv = rec_df.to_csv(index=False)
                    
                    st.download_button(
                        "📊 Download Full Optimization Results",
                        data=csv,
                        file_name=f"optimization_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    
                    # Critical items only
                    critical_items = [r for r in recommendations if r.get('urgency') in ['CRITICAL', 'HIGH']]
                    if critical_items:
                        critical_df = pd.DataFrame(critical_items)
                        critical_csv = critical_df.to_csv(index=False)
                        st.download_button(
                            "🚨 Download Critical Items",
                            data=critical_csv,
                            file_name=f"critical_items_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                
                with col2:
                    st.markdown("### 📄 Optimization Report")
                    
                    # Generate report
                    report = f"""
INVENTORY OPTIMIZATION REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Method: {result.get('method', 'Unknown')}

OPTIMIZATION SUMMARY
====================
Total Investment: ${result.get('total_cost', 0):,.2f}
Expected Profit: ${result.get('expected_profit', 0):,.2f}
ROI: {result.get('roi', 0):.1f}%
Products to Purchase: {len(recommendations)}

TOP 5 RECOMMENDATIONS
=====================
"""
                    for i, rec in enumerate(recommendations[:5], 1):
                        report += f"""
{i}. {rec.get('product_id', 'Unknown')}
   Urgency: {rec.get('urgency', 'Unknown')}
   Purchase: {rec.get('purchase_qty', 0):.0f} units
   Investment: ${rec.get('total_cost', 0):,.2f}
   Expected Profit: ${rec.get('expected_profit', 0):,.2f}
   ROI: {rec.get('roi', 0):.1f}%
"""
                    
                    st.download_button(
                        "📄 Download Optimization Report",
                        data=report,
                        file_name=f"optimization_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
                    
                    with st.expander("Preview Report"):
                        st.text(report)
            else:
                st.info("Run optimization first to generate reports")
        else:
            st.info("Run optimization first to generate reports")
    
    
# ===== END OF INVENTORY OPTIMIZATION SYSTEM =====

# Stable session state initialization
class StockAnalyzer:
    """
    Advanced stock analyzer for product recommendations and profit optimization.
    """
    
    def __init__(self, df: pd.DataFrame):
        """Initialize the analyzer with DataFrame."""
        self.df = self._prepare_dataframe(df)
        self.stock_trends = {}
        self.categorization = {}
        self.recommendations = []
        
    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare and clean the dataframe for analysis."""
        df = df.copy()
        
        # Ensure required columns exist
        required_columns = ['Product Id', 'Weight', 'Max Qty', 'Min Qty', 
                           'Max Buying Price', 'Min Selling Price', 'Month', 'Year']
        
        for col in required_columns:
            if col not in df.columns:
                logger.warning(f"Missing column: {col}")
                df[col] = 0
        
        # Create date column
        try:
            df['date'] = pd.to_datetime(
                df['Year'].astype(str) + '-' + df['Month'], 
                format='%Y-%B', 
                errors='coerce'
            )
        except:
            df['date'] = pd.datetime.now()
        
        # Ensure numeric columns
        numeric_cols = ['Weight', 'Max Qty', 'Min Qty', 'Max Buying Price', 
                       'Min Selling Price', 'Avg Cost Total', 'Buying Price Avg']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        return df
    
    def calculate_stock_trends(self) -> Dict:
        """Calculate stock trends for each product."""
        trends = {}
        
        for product_id in self.df['Product Id'].unique():
            product_data = self.df[self.df['Product Id'] == product_id].sort_values('date')
            
            if len(product_data) > 0:
                # Calculate average quantity
                product_data['avg_qty'] = (product_data['Max Qty'] + product_data['Min Qty']) / 2
                product_data['avg_qty'] = product_data['avg_qty'].fillna(0)
                
                # Calculate moving averages
                if len(product_data) >= 3:
                    product_data['ma_3'] = product_data['avg_qty'].rolling(window=3, min_periods=1).mean()
                else:
                    product_data['ma_3'] = product_data['avg_qty']
                
                # Calculate month-over-month change
                product_data['mom_change'] = product_data['avg_qty'].pct_change()
                product_data['mom_change'] = product_data['mom_change'].fillna(0)
                product_data['mom_change'] = product_data['mom_change'].replace([np.inf, -np.inf], 0)
                
                # Determine trend direction
                recent_trend = product_data['avg_qty'].tail(3).mean()
                historical_avg = product_data['avg_qty'].mean()
                
                if historical_avg == 0:
                    trend_direction = 'stable'
                elif recent_trend > historical_avg * 1.2:
                    trend_direction = 'increasing'
                elif recent_trend < historical_avg * 0.8:
                    trend_direction = 'decreasing'
                else:
                    trend_direction = 'stable'
                
                # Calculate velocity
                velocity = product_data['mom_change'].mean() if len(product_data) > 1 else 0
                
                # Store trend data
                trends[product_id] = {
                    'data': product_data[['date', 'Month', 'Year', 'avg_qty', 'ma_3', 'mom_change']].to_dict('records'),
                    'trend_direction': trend_direction,
                    'velocity': velocity if np.isfinite(velocity) else 0,
                    'avg_quantity': product_data['avg_qty'].mean() if np.isfinite(product_data['avg_qty'].mean()) else 0,
                    'std_quantity': product_data['avg_qty'].std() if np.isfinite(product_data['avg_qty'].std()) else 0,
                    'total_movement': product_data['avg_qty'].sum() if np.isfinite(product_data['avg_qty'].sum()) else 0,
                    'latest_qty': product_data['avg_qty'].iloc[-1] if len(product_data) > 0 else 0,
                    'peak_qty': product_data['avg_qty'].max() if len(product_data) > 0 else 0,
                    'trough_qty': product_data['avg_qty'].min() if len(product_data) > 0 else 0
                }
        
        self.stock_trends = trends
        return trends
    
    def categorize_for_recommendations(self) -> Dict:
        """Categorize products for recommendations."""
        if not self.stock_trends:
            self.calculate_stock_trends()
        
        categorization = {
            'critical_restock': [],  # Out of stock
            'urgent_restock': [],    # Very low stock (≤2 units)
            'low_stock': [],         # Below 20% of average
            'fast_depleting': [],    # Decreasing rapidly
            'optimal_stock': []      # Good stock levels
        }
        
        for product_id, trend_data in self.stock_trends.items():
            latest_qty = trend_data['latest_qty']
            avg_qty = trend_data['avg_quantity']
            trend = trend_data['trend_direction']
            velocity = trend_data['velocity']
            
            product_info = {
                'product_id': product_id,
                'current_stock': latest_qty,
                'avg_stock': avg_qty,
                'trend': trend,
                'velocity': velocity
            }
            
            # Categorize based on stock level and trend
            if latest_qty == 0:
                categorization['critical_restock'].append(product_info)
            elif latest_qty <= 2:
                categorization['urgent_restock'].append(product_info)
            elif avg_qty > 0 and latest_qty < avg_qty * 0.2:
                categorization['low_stock'].append(product_info)
            elif trend == 'decreasing' and velocity < -0.2:
                categorization['fast_depleting'].append(product_info)
            else:
                categorization['optimal_stock'].append(product_info)
        
        self.categorization = categorization
        return categorization
    
    def calculate_profit_potential(self, product_id: str) -> Tuple[float, float]:
        """Calculate profit potential and ROI for a product."""
        product_data = self.df[self.df['Product Id'] == product_id]
        
        if len(product_data) == 0:
            return 0, 0
        
        # Get the most recent data
        latest_data = product_data.sort_values('date').iloc[-1]
        
        buying_price = latest_data['Max Buying Price']
        selling_price = latest_data['Min Selling Price']
        
        profit = selling_price - buying_price
        roi = (profit / buying_price * 100) if buying_price > 0 else 0
        
        return profit, roi
    
    def recommend_stocks_to_buy(self, budget: Optional[float] = None, top_n: int = 10) -> List[Dict]:
        """Generate intelligent stock purchase recommendations."""
        if not self.stock_trends:
            self.calculate_stock_trends()
        
        if not self.categorization:
            self.categorize_for_recommendations()
        
        recommendations = []
        
        # Priority order for restocking
        priority_products = (
            self.categorization['critical_restock'] +
            self.categorization['urgent_restock'] +
            self.categorization['low_stock'] +
            self.categorization['fast_depleting']
        )
        
        # Calculate scores for each product
        for product_info in priority_products:
            product_id = product_info['product_id']
            
            # Get profit and ROI
            profit, roi = self.calculate_profit_potential(product_id)
            
            # Get product details
            product_data = self.df[self.df['Product Id'] == product_id].iloc[-1]
            trend_info = self.stock_trends[product_id]
            
            # Calculate priority score
            score = 0
            
            # Stock criticality (40%)
            if product_id in [p['product_id'] for p in self.categorization['critical_restock']]:
                score += 100 * 0.4
            elif product_id in [p['product_id'] for p in self.categorization['urgent_restock']]:
                score += 80 * 0.4
            elif product_id in [p['product_id'] for p in self.categorization['low_stock']]:
                score += 60 * 0.4
            else:
                score += 40 * 0.4
            
            # ROI weight (30%)
            score += min(roi, 100) * 0.3
            
            # Trend weight (20%)
            if trend_info['trend_direction'] == 'increasing':
                score += 100 * 0.2
            elif trend_info['trend_direction'] == 'stable':
                score += 50 * 0.2
            else:
                score += 25 * 0.2
            
            # Historical movement (10%)
            if trend_info['avg_quantity'] > 0:
                movement_score = min(trend_info['avg_quantity'] / 10, 100)
                score += movement_score * 0.1
            
            # Calculate recommended order quantity
            avg_qty = trend_info['avg_quantity']
            peak_qty = trend_info['peak_qty']
            current_qty = trend_info['latest_qty']
            
            if current_qty <= 1:
                # Critical/urgent restock - order more
                if avg_qty > 0:
                    recommended_qty = max(
                        avg_qty * 3,  # 3 months supply
                        peak_qty * 0.8 if peak_qty > 0 else avg_qty * 2,
                        10  # Minimum order
                    )
                else:
                    recommended_qty = 20  # Default for no history
            else:
                # Regular restock
                if avg_qty > 0:
                    recommended_qty = max(
                        (avg_qty * 2) - current_qty,  # 2 months supply minus current
                        10
                    )
                else:
                    recommended_qty = 10
            
            # Calculate investment required
            investment = recommended_qty * product_data['Max Buying Price']
            expected_return = recommended_qty * profit
            
            recommendation = {
                'product_id': product_id,
                'priority_score': round(score, 2),
                'current_stock': round(current_qty, 0),
                'recommended_qty': round(recommended_qty, 0),
                'unit_cost': round(product_data['Max Buying Price'], 2),
                'unit_profit': round(profit, 2),
                'roi_percentage': round(roi, 2),
                'total_investment': round(investment, 2),
                'expected_return': round(expected_return, 2),
                'trend': trend_info['trend_direction'],
                'avg_monthly_demand': round(avg_qty, 1),
                'stock_status': self._get_stock_status(product_id),
                'shape': product_data.get('Shape key', 'N/A'),
                'color': product_data.get('Color Key', 'N/A'),
                'bucket': product_data.get('Buckets', 'N/A'),
                'urgency': self._get_urgency_level(product_id)
            }
            
            recommendations.append(recommendation)
        
        # Sort by priority score
        recommendations.sort(key=lambda x: x['priority_score'], reverse=True)
        
        # Apply budget constraint if specified
        if budget:
            filtered_recommendations = []
            total_cost = 0
            
            for rec in recommendations:
                if total_cost + rec['total_investment'] <= budget:
                    filtered_recommendations.append(rec)
                    total_cost += rec['total_investment']
            
            recommendations = filtered_recommendations
        
        self.recommendations = recommendations[:top_n]
        return self.recommendations
    
    def _get_stock_status(self, product_id: str) -> str:
        """Get stock status label for a product."""
        for status, products in self.categorization.items():
            if product_id in [p['product_id'] for p in products]:
                return status.replace('_', ' ').title()
        return 'Unknown'
    
    def _get_urgency_level(self, product_id: str) -> str:
        """Determine urgency level for restocking."""
        if product_id in [p['product_id'] for p in self.categorization['critical_restock']]:
            return 'CRITICAL'
        elif product_id in [p['product_id'] for p in self.categorization['urgent_restock']]:
            return 'HIGH'
        elif product_id in [p['product_id'] for p in self.categorization['low_stock']]:
            return 'MEDIUM'
        elif product_id in [p['product_id'] for p in self.categorization['fast_depleting']]:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def get_investment_summary(self) -> Dict:
        """Generate investment summary for recommendations."""
        if not self.recommendations:
            return {}
        
        total_investment = sum(r['total_investment'] for r in self.recommendations)
        total_return = sum(r['expected_return'] for r in self.recommendations)
        avg_roi = np.mean([r['roi_percentage'] for r in self.recommendations])
        
        by_urgency = {}
        for urgency in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
            urgent_recs = [r for r in self.recommendations if r['urgency'] == urgency]
            by_urgency[urgency] = {
                'count': len(urgent_recs),
                'investment': sum(r['total_investment'] for r in urgent_recs)
            }
        
        return {
            'total_investment_required': round(total_investment, 2),
            'expected_total_return': round(total_return, 2),
            'expected_total_profit': round(total_return - total_investment, 2),
            'average_roi': round(avg_roi, 2),
            'number_of_products': len(self.recommendations),
            'by_urgency': by_urgency
        }

@st.cache_data(ttl=3600, show_spinner=False)
def generate_stock_recommendations_cached(df_csv: str, budget: Optional[float], top_n: int):
    """Cached version of stock recommendations."""
    from io import StringIO
    df = pd.read_csv(StringIO(df_csv))
    
    analyzer = StockAnalyzer(df)
    recommendations = analyzer.recommend_stocks_to_buy(budget, top_n)
    summary = analyzer.get_investment_summary()
    
    return recommendations, summary

def display_stock_recommendations_section(master_df):
    """Display the stock recommendations section."""
    st.header("🎯 Smart Stock Purchase Recommendations")
    st.markdown("AI-powered recommendations based on profit potential, stock levels, and demand trends")
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Recommendations Dashboard", 
        "📈 Detailed Analysis", 
        "💰 Investment Planner",
        "📋 Reports & Export"
    ])
    
    with tab1:
        st.subheader("Purchase Recommendations")
        
        # Configuration
        col1, col2, col3 = st.columns(3)
        
        with col1:
            budget_option = st.selectbox(
                "Budget Constraint",
                ["No Budget Limit", "Set Budget Limit"],
                help="Optionally set a budget limit for recommendations",
                key="rec_budget_option"
            )
            
            if budget_option == "Set Budget Limit":
                budget = st.number_input(
                    "Maximum Budget ($)",
                    min_value=1000.0,
                    max_value=1000000.0,
                    value=50000.0,
                    step=1000.0,
                    key="opt_budget_max"
                )
            else:
                budget = None
        
        with col2:
            top_n = st.number_input(
                "Number of Recommendations",
                min_value=5,
                max_value=50,
                value=15,
                step=5,
                help="Number of top products to recommend",
                key="opt_top_n"
            )
        
        with col3:
            min_roi = st.slider(
                "Minimum ROI Filter (%)",
                min_value=0,
                max_value=100,
                value=0,
                step=5,
                help="Filter recommendations by minimum ROI",
                key="opt_min_roi"
            )
        
        # Generate recommendations button
        if st.button("🔍 Generate Recommendations", type="primary", key="gen_recommendations"):
            with st.spinner("Analyzing inventory and generating recommendations..."):
                # Convert dataframe for caching
                df_csv = master_df.to_csv(index=False)
                
                # Get recommendations
                recommendations, summary = generate_stock_recommendations_cached(df_csv, budget, top_n)
                
                # Store in session state
                st.session_state.stock_recommendations = recommendations
                st.session_state.investment_summary = summary
                
                st.success(f"Generated {len(recommendations)} recommendations!")
        
        # Display recommendations if available
        if 'stock_recommendations' in st.session_state and st.session_state.stock_recommendations:
            recommendations = st.session_state.stock_recommendations
            summary = st.session_state.investment_summary
            
            # Apply ROI filter
            if min_roi > 0:
                recommendations = [r for r in recommendations if r['roi_percentage'] >= min_roi]
            
            # Summary metrics
            st.markdown("---")
            st.subheader("📊 Investment Summary")
            
            metric_cols = st.columns(5)
            
            with metric_cols[0]:
                st.metric(
                    "Total Investment",
                    f"${summary['total_investment_required']:,.0f}",
                    help="Total capital required for all recommendations"
                )
            
            with metric_cols[1]:
                st.metric(
                    "Expected Return",
                    f"${summary['expected_total_return']:,.0f}",
                    f"+${summary['expected_total_profit']:,.0f}",
                    help="Total expected revenue and profit"
                )
            
            with metric_cols[2]:
                st.metric(
                    "Average ROI",
                    f"{summary['average_roi']:.1f}%",
                    help="Average return on investment across all recommendations"
                )
            
            with metric_cols[3]:
                st.metric(
                    "Products",
                    len(recommendations),
                    help="Number of products recommended for purchase"
                )
            
            with metric_cols[4]:
                critical_count = summary['by_urgency'].get('CRITICAL', {}).get('count', 0)
                st.metric(
                    "Critical Items",
                    critical_count,
                    help="Products that are out of stock"
                )
            
            # Urgency breakdown
            st.markdown("---")
            st.subheader("⚡ Urgency Breakdown")
            
            urgency_cols = st.columns(4)
            urgency_colors = {
                'CRITICAL': '🔴',
                'HIGH': '🟠',
                'MEDIUM': '🟡',
                'LOW': '🟢'
            }
            
            for idx, (urgency, emoji) in enumerate(urgency_colors.items()):
                with urgency_cols[idx]:
                    urgency_data = summary['by_urgency'].get(urgency, {})
                    count = urgency_data.get('count', 0)
                    investment = urgency_data.get('investment', 0)
                    
                    st.markdown(f"**{emoji} {urgency}**")
                    st.markdown(f"Products: **{count}**")
                    st.markdown(f"Investment: **${investment:,.0f}**")
            
            # Recommendations table
            st.markdown("---")
            st.subheader("🛒 Purchase Recommendations")
            
            # Create DataFrame for display
            display_data = []
            for idx, rec in enumerate(recommendations, 1):
                display_data.append({
                    'Rank': idx,
                    'Product ID': rec['product_id'],
                    'Urgency': rec['urgency'],
                    'Current Stock': f"{rec['current_stock']:.0f}",
                    'Recommended Qty': f"{rec['recommended_qty']:.0f}",
                    'Unit Cost': f"${rec['unit_cost']:,.2f}",
                    'Unit Profit': f"${rec['unit_profit']:,.2f}",
                    'ROI %': f"{rec['roi_percentage']:.1f}%",
                    'Investment': f"${rec['total_investment']:,.2f}",
                    'Expected Return': f"${rec['expected_return']:,.2f}",
                    'Trend': rec['trend'].title(),
                    'Shape': rec['shape'],
                    'Color': rec['color']
                })
            
            recommendations_df = pd.DataFrame(display_data)
            
            # Style the dataframe
            def highlight_urgency(row):
                if row['Urgency'] == 'CRITICAL':
                    return ['background-color: #ffebee'] * len(row)
                elif row['Urgency'] == 'HIGH':
                    return ['background-color: #fff3e0'] * len(row)
                elif row['Urgency'] == 'MEDIUM':
                    return ['background-color: #fffde7'] * len(row)
                else:
                    return [''] * len(row)
            
            styled_df = recommendations_df.style.apply(highlight_urgency, axis=1)
            
            st.dataframe(
                styled_df,
                use_container_width=True,
                hide_index=True,
                height=400
            )
        else:
            st.info("Click 'Generate Recommendations' to see purchase suggestions")
    
    with tab2:
        st.subheader("Detailed Product Analysis")
        
        if 'stock_recommendations' in st.session_state and st.session_state.stock_recommendations:
            # Product selector
            product_ids = [rec['product_id'] for rec in st.session_state.stock_recommendations]
            
            selected_product = st.selectbox(
                "Select Product for Detailed Analysis",
                product_ids,
                format_func=lambda x: f"{x} - {next((r['urgency'] for r in st.session_state.stock_recommendations if r['product_id'] == x), '')}",
                key="rec_product_selector"
            )
            
            if selected_product:
                # Get product details
                product_rec = next(r for r in st.session_state.stock_recommendations if r['product_id'] == selected_product)
                
                # Display detailed metrics
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### Stock Information")
                    st.markdown(f"**Current Stock:** {product_rec['current_stock']:.0f} units")
                    st.markdown(f"**Recommended Order:** {product_rec['recommended_qty']:.0f} units")
                    st.markdown(f"**Average Monthly Demand:** {product_rec['avg_monthly_demand']:.1f} units")
                    st.markdown(f"**Stock Status:** {product_rec['stock_status']}")
                    st.markdown(f"**Trend:** {product_rec['trend'].title()}")
                
                with col2:
                    st.markdown("### Financial Analysis")
                    st.markdown(f"**Unit Cost:** ${product_rec['unit_cost']:,.2f}")
                    st.markdown(f"**Unit Profit:** ${product_rec['unit_profit']:,.2f}")
                    st.markdown(f"**ROI:** {product_rec['roi_percentage']:.1f}%")
                    st.markdown(f"**Total Investment:** ${product_rec['total_investment']:,.2f}")
                    st.markdown(f"**Expected Return:** ${product_rec['expected_return']:,.2f}")
                
                # Historical trend chart
                st.markdown("---")
                st.markdown("### Historical Stock Trend")
                
                # Get historical data for the product
                product_history = master_df[master_df['Product Id'] == selected_product].copy()
                
                if not product_history.empty:
                    product_history['avg_stock'] = (product_history['Max Qty'] + product_history['Min Qty']) / 2
                    product_history['date'] = pd.to_datetime(
                        product_history['Year'].astype(str) + '-' + product_history['Month'],
                        format='%Y-%B'
                    )
                    product_history = product_history.sort_values('date')
                    
                    # Create trend chart
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=product_history['date'],
                        y=product_history['avg_stock'],
                        mode='lines+markers',
                        name='Average Stock',
                        line=dict(color='#2196F3', width=2)
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=product_history['date'],
                        y=product_history['Max Qty'],
                        mode='lines',
                        name='Max Quantity',
                        line=dict(color='#4CAF50', width=1, dash='dash')
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=product_history['date'],
                        y=product_history['Min Qty'],
                        mode='lines',
                        name='Min Quantity',
                        line=dict(color='#FF9800', width=1, dash='dash')
                    ))
                    
                    fig.update_layout(
                        title=f"Stock Trend for {selected_product}",
                        xaxis_title="Date",
                        yaxis_title="Quantity",
                        height=400,
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Generate recommendations first to see detailed analysis")
    
    with tab3:
        st.subheader("💰 Investment Planning Tool")
        
        if 'stock_recommendations' in st.session_state and st.session_state.stock_recommendations:
            recommendations = st.session_state.stock_recommendations
            
            # Investment scenarios
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Investment Scenarios")
                
                scenario = st.radio(
                    "Select Investment Strategy",
                    ["Conservative (Critical Only)", 
                     "Balanced (Critical + High Priority)",
                     "Aggressive (All Recommendations)",
                     "Custom Selection"],
                     key="investment_scenario"
                )
                
                if scenario == "Conservative (Critical Only)":
                    selected_recs = [r for r in recommendations if r['urgency'] == 'CRITICAL']
                elif scenario == "Balanced (Critical + High Priority)":
                    selected_recs = [r for r in recommendations if r['urgency'] in ['CRITICAL', 'HIGH']]
                elif scenario == "Aggressive (All Recommendations)":
                    selected_recs = recommendations
                else:  # Custom
                    product_ids = st.multiselect(
                        "Select Products to Include",
                        [r['product_id'] for r in recommendations],
                        format_func=lambda x: f"{x} ({next((r['urgency'] for r in recommendations if r['product_id'] == x), '')})"
                    )
                    selected_recs = [r for r in recommendations if r['product_id'] in product_ids]
            
            with col2:
                st.markdown("### Scenario Summary")
                
                if selected_recs:
                    total_investment = sum(r['total_investment'] for r in selected_recs)
                    total_return = sum(r['expected_return'] for r in selected_recs)
                    total_profit = total_return - total_investment
                    avg_roi = np.mean([r['roi_percentage'] for r in selected_recs])
                    
                    st.metric("Products Selected", len(selected_recs))
                    st.metric("Total Investment", f"${total_investment:,.2f}")
                    st.metric("Expected Profit", f"${total_profit:,.2f}")
                    st.metric("Average ROI", f"{avg_roi:.1f}%")
                else:
                    st.info("Select products to see investment summary")
            
            # ROI Analysis Chart
            if selected_recs:
                st.markdown("---")
                st.markdown("### ROI Analysis")
                
                # Create ROI comparison chart
                roi_data = pd.DataFrame([
                    {
                        'Product': r['product_id'],
                        'ROI %': r['roi_percentage'],
                        'Investment': r['total_investment'],
                        'Profit': r['expected_return'] - r['total_investment']
                    }
                    for r in selected_recs
                ])
                
                # Sort by ROI
                roi_data = roi_data.sort_values('ROI %', ascending=True)
                
                # Create horizontal bar chart
                fig = px.bar(
                    roi_data,
                    x='ROI %',
                    y='Product',
                    orientation='h',
                    title='Return on Investment by Product',
                    color='ROI %',
                    color_continuous_scale='RdYlGn',
                    hover_data=['Investment', 'Profit']
                )
                
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Investment allocation pie chart
                st.markdown("### Investment Allocation")
                
                allocation_fig = px.pie(
                    values=[r['total_investment'] for r in selected_recs],
                    names=[r['product_id'] for r in selected_recs],
                    title='Investment Distribution by Product'
                )
                
                st.plotly_chart(allocation_fig, use_container_width=True)
        else:
            st.info("Generate recommendations first to use the investment planner")
    
    with tab4:
        st.subheader("📋 Reports & Export")
        
        if 'stock_recommendations' in st.session_state and st.session_state.stock_recommendations:
            recommendations = st.session_state.stock_recommendations
            summary = st.session_state.investment_summary
            
            # Report options
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Download Reports")
                
                # Full recommendations CSV
                rec_df = pd.DataFrame(recommendations)
                csv = rec_df.to_csv(index=False)
                
                st.download_button(
                    "📊 Download Full Recommendations (CSV)",
                    data=csv,
                    file_name=f"stock_recommendations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
                # Critical items only
                critical_df = rec_df[rec_df['urgency'] == 'CRITICAL']
                if not critical_df.empty:
                    critical_csv = critical_df.to_csv(index=False)
                    st.download_button(
                        "🔴 Download Critical Items Only",
                        data=critical_csv,
                        file_name=f"critical_restock_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            
            with col2:
                st.markdown("### Executive Summary")
                
                # Generate executive summary
                exec_summary = f"""
STOCK PURCHASE RECOMMENDATIONS - EXECUTIVE SUMMARY
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

INVESTMENT OVERVIEW
==================
Total Investment Required: ${summary['total_investment_required']:,.2f}
Expected Total Return: ${summary['expected_total_return']:,.2f}
Expected Profit: ${summary['expected_total_profit']:,.2f}
Average ROI: {summary['average_roi']:.1f}%
Number of Products: {summary['number_of_products']}

URGENCY BREAKDOWN
=================
Critical (Out of Stock): {summary['by_urgency']['CRITICAL']['count']} products - ${summary['by_urgency']['CRITICAL']['investment']:,.2f}
High Priority: {summary['by_urgency']['HIGH']['count']} products - ${summary['by_urgency']['HIGH']['investment']:,.2f}
Medium Priority: {summary['by_urgency']['MEDIUM']['count']} products - ${summary['by_urgency']['MEDIUM']['investment']:,.2f}
Low Priority: {summary['by_urgency']['LOW']['count']} products - ${summary['by_urgency']['LOW']['investment']:,.2f}

TOP 5 RECOMMENDATIONS
=====================
"""
                for i, rec in enumerate(recommendations[:5], 1):
                    exec_summary += f"""
{i}. {rec['product_id']}
   - Urgency: {rec['urgency']}
   - Current Stock: {rec['current_stock']:.0f} units
   - Recommended Order: {rec['recommended_qty']:.0f} units
   - Investment: ${rec['total_investment']:,.2f}
   - ROI: {rec['roi_percentage']:.1f}%
"""
                
                st.download_button(
                    "📄 Download Executive Summary",
                    data=exec_summary,
                    file_name=f"executive_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
                
                # Preview summary
                with st.expander("Preview Executive Summary"):
                    st.text(exec_summary)
            
            # Action items
            st.markdown("---")
            st.markdown("### 🎯 Recommended Actions")
            
            action_cols = st.columns(3)
            
            with action_cols[0]:
                st.markdown("**🔴 Immediate Action Required**")
                critical_items = [r for r in recommendations if r['urgency'] == 'CRITICAL']
                if critical_items:
                    for item in critical_items[:3]:
                        st.write(f"• Order {item['recommended_qty']:.0f} units of {item['product_id']}")
                else:
                    st.write("No critical items")
            
            with action_cols[1]:
                st.markdown("**🟠 High Priority Orders**")
                high_items = [r for r in recommendations if r['urgency'] == 'HIGH']
                if high_items:
                    for item in high_items[:3]:
                        st.write(f"• Order {item['recommended_qty']:.0f} units of {item['product_id']}")
                else:
                    st.write("No high priority items")
            
            with action_cols[2]:
                st.markdown("**💰 Best ROI Opportunities**")
                sorted_by_roi = sorted(recommendations, key=lambda x: x['roi_percentage'], reverse=True)
                for item in sorted_by_roi[:3]:
                    st.write(f"• {item['product_id']}: {item['roi_percentage']:.1f}% ROI")
        else:
            st.info("Generate recommendations first to create reports")

# ===== END OF ADVANCED STOCK ANALYZER & RECOMMENDATIONS =====

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
        'show_upload_history': False,
        'trend_analysis_results': None,
        'trend_criteria': {
            'min_activity_months': 5,
            'min_avg_cost': 3000,
            'min_total_value': 10000,
            'price_volatility': 0.2
        },
        'optimization_result': None,
        'critical_products': [],
        'budget_scenarios': {}
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

# Cache configurations with stable keys
@st.cache_data(ttl=3600, show_spinner=False)
def load_cached_master_dataset():
    """Load master dataset with proper compression handling"""
    try:
        master_file_path = Path(r"C:\streamlit-app\src\kunmings.pkl")
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
        max_qty_dict = joblib.load(r'C:\streamlit-app\src\max_qty.pkl')
        min_qty_dict = joblib.load(r'C:\streamlit-app\src\min_qty.pkl')
        max_buy_dict = joblib.load(r'C:\streamlit-app\src\max_buy.pkl')
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

# ===== MISSING PRODUCTS ANALYSIS FUNCTIONS =====

@st.cache_data(ttl=3600, show_spinner=False)
def get_missing_products_analysis_cached(df_csv: str, month: str, year: int, shape: str, color: str, bucket: str):
    """Cached version of missing products analysis"""
    try:
        from io import StringIO
        df = pd.read_csv(StringIO(df_csv))
        
        # Get all unique product IDs
        all_product_ids = set(df['Product Id'].unique())
        
        # Filter for selected month and year
        mask = (df['Month'] == month) & (df['Year'] == year)
        
        # Apply additional filters if selected
        if shape != "None":
            mask &= (df['Shape key'] == shape)
        if color != "None":
            mask &= (df['Color Key'] == color)
        if bucket != "None":
            mask &= (df['Buckets'] == bucket)
        
        current_month_products = set(df[mask]['Product Id'].unique())
        
        # Find missing products
        missing_products = all_product_ids - current_month_products
        
        if not missing_products:
            return pd.DataFrame(), {}
        
        # Get details of missing products from their last occurrence
        missing_details = []
        
        for product_id in missing_products:
            # Get last occurrence of this product
            product_data = df[df['Product Id'] == product_id].copy()
            
            if not product_data.empty:
                # Sort by year and month to get the most recent data
                product_data['date'] = pd.to_datetime(
                    product_data['Year'].astype(str) + '-' + product_data['Month'], 
                    format='%Y-%B', 
                    errors='coerce'
                )
                product_data = product_data.sort_values('date')
                last_record = product_data.iloc[-1]
                
                # Calculate months since last seen
                last_date = last_record['date']
                current_date = pd.to_datetime(f"{year}-{month}", format='%Y-%B')
                months_missing = (current_date.year - last_date.year) * 12 + (current_date.month - last_date.month)
                
                missing_details.append({
                    'Product ID': product_id,
                    'Last Seen Month': last_record['Month'],
                    'Last Seen Year': int(last_record['Year']),
                    'Months Missing': months_missing,
                    'Shape': last_record.get('Shape key', 'Unknown'),
                    'Color': last_record.get('Color Key', 'Unknown'),
                    'Bucket': last_record.get('Buckets', 'Unknown'),
                    'Last Weight': round(last_record.get('Weight', 0), 2),
                    'Last Avg Cost': round(last_record.get('Avg Cost Total', 0), 2),
                    'Last Max Qty': int(last_record.get('Max Qty', 0)),
                    'Last Min Qty': int(last_record.get('Min Qty', 0)),
                    'Max Buying Price': round(last_record.get('Max Buying Price', 0), 2),
                    'Min Selling Price': round(last_record.get('Min Selling Price', 0), 2)
                })
        
        missing_df = pd.DataFrame(missing_details)
        
        if not missing_df.empty:
            # Sort by months missing (descending) to show longest missing first
            missing_df = missing_df.sort_values('Months Missing', ascending=False)
            
            # Categorize missing products
            categories = {
                'Critical': missing_df[missing_df['Months Missing'] >= 3],
                'Warning': missing_df[(missing_df['Months Missing'] >= 2) & (missing_df['Months Missing'] < 3)],
                'Recent': missing_df[missing_df['Months Missing'] < 2]
            }
            
            # Calculate statistics
            stats = {
                'total_missing': len(missing_df),
                'critical_missing': len(categories['Critical']),
                'warning_missing': len(categories['Warning']),
                'recent_missing': len(categories['Recent']),
                'avg_months_missing': round(missing_df['Months Missing'].mean(), 1),
                'max_months_missing': int(missing_df['Months Missing'].max()),
                'total_value_missing': round(missing_df['Last Avg Cost'].sum(), 2)
            }
            
            return missing_df, stats
        
        return pd.DataFrame(), {}
        
    except Exception as e:
        logger.error(f"Error in missing products analysis: {e}")
        return pd.DataFrame(), {}

def display_missing_products_analysis(master_df, selected_month, selected_year, selected_shape, selected_color, selected_bucket):
    """Display missing products analysis section"""
    try:
        st.markdown("---")
        st.subheader("🔍 Missing Products Analysis")
        st.markdown(f"**Products not found in {selected_month} {selected_year}**")
        
        # Convert dataframe for caching
        df_csv = master_df.to_csv(index=False)
        
        # Get missing products analysis
        missing_df, stats = get_missing_products_analysis_cached(
            df_csv, selected_month, selected_year, 
            selected_shape, selected_color, selected_bucket
        )
        
        if missing_df.empty:
            st.success("✅ All products are present in the selected month!")
            return
        
        # Display summary metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                "Total Missing",
                f"{stats['total_missing']}",
                help="Total number of products not found in selected month"
            )
        
        with col2:
            st.metric(
                "Critical (≥3 months)",
                f"{stats['critical_missing']}",
                help="Products missing for 3 or more months"
            )
        
        with col3:
            st.metric(
                "Warning (2 months)",
                f"{stats['warning_missing']}",
                help="Products missing for 2 months"
            )
        
        with col4:
            st.metric(
                "Avg Months Missing",
                f"{stats['avg_months_missing']}",
                help="Average number of months products have been missing"
            )
        
        with col5:
            st.metric(
                "Est. Value Missing",
                f"${stats['total_value_missing']:,.0f}",
                help="Total last known value of missing products"
            )
        
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["📊 Summary View", "📋 Detailed Table", "📈 Analysis Charts"])
        
        with tab1:
            # Categorized display
            if stats['critical_missing'] > 0:
                st.markdown("### 🔴 Critical - Missing ≥3 Months")
                critical_df = missing_df[missing_df['Months Missing'] >= 3].head(10)
                
                # Display as dataframe with key columns
                critical_cols = [
                    'Product ID', 'Months Missing', 'Last Seen Month', 'Last Seen Year',
                    'Shape', 'Color', 'Bucket', 'Last Weight', 'Last Avg Cost',
                    'Last Max Qty', 'Last Min Qty', 'Max Buying Price'
                ]
                # Filter to only include columns that exist in the dataframe
                available_cols = [col for col in critical_cols if col in critical_df.columns]
                st.dataframe(critical_df[available_cols], use_container_width=True, hide_index=True)
            
            if stats['warning_missing'] > 0:
                st.markdown("### 🟡 Warning - Missing 2 Months")
                warning_df = missing_df[(missing_df['Months Missing'] >= 2) & (missing_df['Months Missing'] < 3)].head(5)
                warning_cols = ['Product ID', 'Last Seen Month', 'Last Seen Year', 'Shape', 'Color', 'Last Avg Cost']
                st.dataframe(warning_df[warning_cols], use_container_width=True, hide_index=True)
            
            if stats['recent_missing'] > 0:
                st.markdown("### 🟢 Recent - Missing <2 Months")
                recent_df = missing_df[missing_df['Months Missing'] < 2].head(5)
                recent_cols = ['Product ID', 'Last Seen Month', 'Shape', 'Color', 'Last Avg Cost']
                st.dataframe(recent_df[recent_cols], use_container_width=True, hide_index=True)
        
        with tab2:
            # Full detailed table
            st.markdown("### Complete Missing Products List")
            
            # Add filters for the table
            col1, col2, col3 = st.columns(3)
            with col1:
                min_months = st.number_input(
                    "Min Months Missing",
                    min_value=0,
                    max_value=int(missing_df['Months Missing'].max()),
                    value=0,
                    key="missing_min_months"
                )
            with col2:
                shape_filter = st.selectbox(
                    "Filter by Shape",
                    ['All'] + sorted(missing_df['Shape'].unique().tolist()),
                    key="missing_shape_filter"
                )
            with col3:
                color_filter = st.selectbox(
                    "Filter by Color",
                    ['All'] + sorted(missing_df['Color'].unique().tolist()),
                    key="missing_color_filter"
                )
            
            # Apply filters
            filtered_missing = missing_df[missing_df['Months Missing'] >= min_months]
            if shape_filter != 'All':
                filtered_missing = filtered_missing[filtered_missing['Shape'] == shape_filter]
            if color_filter != 'All':
                filtered_missing = filtered_missing[filtered_missing['Color'] == color_filter]
            
            # Style the dataframe
            def style_missing_months(val):
                if val >= 3:
                    return 'background-color: #ffcdd2'
                elif val >= 2:
                    return 'background-color: #fff9c4'
                else:
                    return 'background-color: #c8e6c9'
            
            styled_df = filtered_missing.style.applymap(
                style_missing_months, 
                subset=['Months Missing']
            )
            
            st.dataframe(styled_df, use_container_width=True, hide_index=True, height=400)
            
            # Download button
            csv = filtered_missing.to_csv(index=False)
            st.download_button(
                "📥 Download Missing Products Report",
                data=csv,
                file_name=f"missing_products_{selected_month}_{selected_year}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="download_missing_products"
            )
        
        with tab3:
            # Analysis charts
            if len(missing_df) > 0:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Distribution by months missing
                    months_dist = missing_df['Months Missing'].value_counts().sort_index()
                    fig = px.bar(
                        x=months_dist.index,
                        y=months_dist.values,
                        title="Distribution by Months Missing",
                        labels={'x': 'Months Missing', 'y': 'Number of Products'},
                        color=months_dist.values,
                        color_continuous_scale='Reds'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Top 10 by value
                    top_value = missing_df.nlargest(10, 'Last Avg Cost')
                    fig = px.bar(
                        top_value,
                        x='Last Avg Cost',
                        y='Product ID',
                        orientation='h',
                        title="Top 10 Missing Products by Value",
                        color='Months Missing',
                        color_continuous_scale='RdYlGn_r'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Distribution by shape
                shape_dist = missing_df.groupby('Shape').agg({
                    'Product ID': 'count',
                    'Last Avg Cost': 'sum'
                }).round(2)
                shape_dist.columns = ['Count', 'Total Value']
                
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=('Missing Products by Shape', 'Missing Value by Shape'),
                    specs=[[{'type': 'pie'}, {'type': 'pie'}]]
                )
                
                fig.add_trace(
                    go.Pie(labels=shape_dist.index, values=shape_dist['Count'], name='Count'),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Pie(labels=shape_dist.index, values=shape_dist['Total Value'], name='Value'),
                    row=1, col=2
                )
                
                fig.update_layout(title="Missing Products Analysis by Shape", height=400)
                st.plotly_chart(fig, use_container_width=True)
                
    except Exception as e:
        st.error(f"Error in missing products analysis: {str(e)}")
        logger.error(f"Error in missing products analysis: {e}")

# ===== END OF MISSING PRODUCTS ANALYSIS FUNCTIONS =====

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
        json_data_path = rf"C:\streamlit-app\src\{json_data_name}"
        
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
        drop_cols = df.isna().sum()[df.isna().sum()>len(df)*.8].index.tolist()
        df.drop(columns=drop_cols,inplace=True)
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
        drop_cols = df.isna().sum()[df.isna().sum()>len(df)*.8].index.tolist()
        df.drop(columns=drop_cols,inplace=True)
        columns = list(concatenate_first_two_rows(df.iloc[0:2, 2:]).values())
        columns = ['Months', 'Buckets'] + columns
        df.columns = columns
        df = df.iloc[2:,:].reset_index(drop=True)
        
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
        drop_cols = df.isna().sum()[df.isna().sum()>len(df)*.8].index.tolist()
        df.drop(columns=drop_cols,inplace=True)
        columns = list(concatenate_first_two_rows(df.iloc[0:2, 2:]).values())
        columns = ['Months', 'Buckets'] + columns
        df.columns = columns
        df = df.iloc[2:,:].reset_index(drop=True)
        
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
        drop_cols = df.isna().sum()[df.isna().sum()>len(df)*.8].index.tolist()
        df.drop(columns=drop_cols,inplace=True)
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
def calculate_rolling_qoq_change(df, value_column, month_column='month', year_column=None, agg_method='sum'):
    """
    Calculate rolling quarter-on-quarter percentage change with sliding 3-month windows.
    
    Q1: Jan, Feb, Mar (months 1, 2, 3)
    Q2: Feb, Mar, Apr (months 2, 3, 4)
    Q3: Mar, Apr, May (months 3, 4, 5)
    ... and so on
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    value_column : str
        Name of the column to calculate percentage change on
    month_column : str, default='month'
        Name of the column containing month numbers (1-12)
    year_column : str, optional
        Name of the column containing year. If None, assumes all data is from same year
    agg_method : str, default='sum'
        Aggregation method for grouping by month ('sum', 'mean', etc.)
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with rolling quarters and their QoQ percentage changes
    """
    
    # Create a copy to avoid modifying original dataframe
    df_copy = df.copy()
    
    # If year column is provided, create a period column for proper sorting
    if year_column:
        df_copy['period'] = df_copy[year_column] * 100 + df_copy[month_column]
        group_cols = [year_column, month_column]
    else:
        df_copy['period'] = df_copy[month_column]
        group_cols = [month_column]
    
    # Aggregate data by month (and year if provided)
    if agg_method == 'sum':
        monthly_data = df_copy.groupby(group_cols)[value_column].sum().reset_index()
    elif agg_method == 'mean':
        monthly_data = df_copy.groupby(group_cols)[value_column].mean().reset_index()
    elif agg_method == 'count':
        monthly_data = df_copy.groupby(group_cols)[value_column].count().reset_index()
    else:
        monthly_data = df_copy.groupby(group_cols)[value_column].agg(agg_method).reset_index()
    
    # Sort by period
    if year_column:
        monthly_data = monthly_data.sort_values([year_column, month_column])
        monthly_data['period'] = monthly_data[year_column] * 100 + monthly_data[month_column]
    else:
        monthly_data = monthly_data.sort_values(month_column)
    
    # Calculate rolling 3-month sum/mean
    monthly_data['rolling_quarter_value'] = monthly_data[value_column].rolling(window=3, min_periods=3).sum()
    
    # Create quarter labels
    if year_column:
        monthly_data['quarter_label'] = (
            monthly_data[year_column].astype(str) + '-' + 
            'Q' + monthly_data[month_column].astype(str)
        )
    else:
        monthly_data['quarter_label'] = 'Q' + monthly_data[month_column].astype(str)
    
    # Calculate quarter-on-quarter percentage change
    monthly_data['qoq_pct_change'] = monthly_data['rolling_quarter_value'].pct_change() * 100
    
    # Round for better readability
    monthly_data['qoq_pct_change'] = monthly_data['qoq_pct_change'].round(2)
    
    # Keep only rows where we have a full 3-month window
    result = monthly_data[monthly_data['rolling_quarter_value'].notna()].copy()
    
    # Select relevant columns
    if year_column:
        result = result[[year_column, month_column, 'quarter_label', 
                        'rolling_quarter_value', 'qoq_pct_change']]
    else:
        result = result[[month_column, 'quarter_label', 
                        'rolling_quarter_value', 'qoq_pct_change']]
    
    return result
@st.cache_data(show_spinner=False)
def monthly_variance_stable(df_csv: str, col: str,Year:str):
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
        
        result = calculate_rolling_qoq_change(analysis,col,'Num_Month','Year')
        result = result.replace([np.inf, -np.inf, np.nan], 0)
        
        analysis = analysis.merge(result,on=['Year',"Num_Month"],how="left")
        analysis.rename(columns={'qoq_pct_change':'qaurter_change'},inplace=True)
        
        # Round values
        analysis['Monthly_change'] = analysis['Monthly_change'].round(2)
        analysis['qaurter_change'] = analysis['qaurter_change'].round(2)
        
        return analysis
        
    except Exception as e:
        logger.error(f"Error calculating monthly variance: {e}")
        return pd.DataFrame()

def monthly_variance(df, col,Year='Year'):
    """Stable monthly variance with caching"""
    if df.empty or col not in df.columns:
        return pd.DataFrame()
    
    df_csv = df.to_csv(index=False)
    return monthly_variance_stable(df_csv, col,Year)

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
        
        file_path = r'C:\streamlit-app\src\kunmings.pkl'
        
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

def optimized_get_filtered_data(filter_month, filter_year, filter_shape, filter_color, filter_bucket,var_col):
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
        var_mask = (
            (master_df['Shape key'] == filter_shape) &
            (master_df['Color Key'] == filter_color) &
            (master_df['Buckets'] == filter_bucket)
        ) 
        
        filter_data = master_df[mask]
        var_data = master_df[var_mask]
        
        if not filter_data.empty:
            max_buying_price = filter_data['Max Buying Price'].max()
            weight_sum = filter_data['Weight'].sum()
            current_avg_cost = (filter_data['Avg Cost Total'].sum() / weight_sum * 0.9) if weight_sum > 0 else 0
            min_selling_price = filter_data['Min Selling Price'].min()
            max_qty = filter_data['Max Qty'].max() 
            min_qty = filter_data['Min Qty'].min() 
            variance_data = monthly_variance(var_data,var_col)
            var_filter_mask = (
            (variance_data['Month'] == filter_month) & 
            (variance_data['Year'] == int(filter_year))
            )
            variance_filtered_data = variance_data[var_filter_mask]
            avg = variance_filtered_data[var_col].mean() if not variance_filtered_data.empty else 0
            if avg != 0 :
                deviation = (filter_data[var_col] - avg)/avg
                mom_variance = deviation.mean() * 100 
            else:
                mom_variance = 0
            Monthly_change = float(variance_filtered_data['Monthly_change'].values[0]) if not variance_filtered_data.empty else 0
            qaurter_change = float(variance_filtered_data['qaurter_change'].values[0]) if not variance_filtered_data.empty else 0


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
            qaurter_change = 0
            Monthly_change = 0
            mom_variance = 0
        
        # Calculate gap
        
        stock_in_hand = len(filter_data)
        gap_output = gap_analysis(max_qty, min_qty, stock_in_hand)
        
        return [filter_data, int(max_buying_price), int(current_avg_cost), gap_output, min_selling_price, Monthly_change, qaurter_change,mom_variance]
        
    except Exception as e:
        logger.error(f"Error getting filtered data: {e}")
        return [pd.DataFrame(), 0, "Error", 0, 0,0,0,0]

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
                            corrupted_file = Path(r"C:\streamlit-app\src\kunmings.pkl")
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
        # Create main tabs for different sections
        main_tab1, main_tab2, main_tab3 = st.tabs([
            "📊 Dashboard & GAP Analysis", 
            "📈 Stock Trend Analysis",
            "🔬 Optimized Inventory Recommendations"
        ])
        
        with main_tab1:
            # Original dashboard content
            display_original_dashboard()
        
        with main_tab2:
            # Trend analysis section
            if not st.session_state.master_df.empty:
                display_trend_analysis_section(st.session_state.master_df)
            else:
                st.info("No data available. Please upload an Excel file to run trend analysis.")
        
        with main_tab3:
            # Optimized recommendations section
            if not st.session_state.master_df.empty:
                display_optimized_recommendations_section(st.session_state.master_df)
            else:
                st.info("No data available. Please upload an Excel file to generate optimized recommendations.")
                
    except Exception as e:
        st.error(f"Error in dashboard display: {str(e)}")
        logger.error(f"Error in dashboard display: {e}")

def display_original_dashboard():
    """Display the original dashboard with filters and GAP analysis"""
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
        
        # Display missing products analysis for selected month
        if selected_month != "None" and selected_year != "None":
            display_missing_products_analysis(master_df, selected_month, int(selected_year), 
                                             selected_shape, selected_color, selected_bucket)
        
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
        st.error(f"Error in original dashboard display: {str(e)}")
        logger.error(f"Error in original dashboard display: {e}")

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
    variance_column = selected_variance_column
    if variance_column == 'Current Average Cost':
        variance_column = 'Buying Price Avg'
    elif variance_column in ['None',None]:
        variance_column = 'Max Buying Price'
    else:
        variance_column = selected_variance_column
    if all_filters_selected:
        display_detailed_metrics(display_df, selected_month, selected_year, selected_shape,
                               selected_color, selected_bucket, variance_column)
    else:
        display_aggregated_metrics(display_df)

def display_detailed_metrics(filter_data, selected_month, selected_year, selected_shape,
                           selected_color, selected_bucket, selected_variance_column):
    """Display detailed metrics for specific filters"""
    try:
        # Get filtered data efficiently
        filtered_results = optimized_get_filtered_data(
            selected_month, selected_year, selected_shape, selected_color, selected_bucket,selected_variance_column)
        
        filter_data, max_buying_price, current_avg_cost, gap_output, min_selling_price,mom_percent_change,mom_qoq_percent_change,mom_variance = filtered_results
        
        # # Get summary metrics
        # mom_variance, mom_percent_change, mom_qoq_percent_change = optimized_get_summary_metrics(
        #     filter_data, selected_month, selected_shape, selected_year,
        #     selected_color, selected_bucket, selected_variance_column)
        
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
