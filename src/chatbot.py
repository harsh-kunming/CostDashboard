"""
AI Chatbot Module for Cost Dashboard
Uses HuggingFace Inference API with GPT-OSS-120B
"""

import os
import pandas as pd
import streamlit as st
import requests
import json
import time
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime
from collections import deque

# Import new enhanced components
from enhanced_query_analyzer import EnhancedQueryAnalyzer
from deque_window_processor import DequeWindowProcessor

logger = logging.getLogger(__name__)

# HuggingFace Configuration - New Router API
# ORIGINAL: HF_API_URL = "https://router.huggingface.co/v1/chat/completions"
# Alternative endpoint if router.huggingface.co is not accessible:
HF_API_URL = os.environ.get('HF_API_URL', "https://router.huggingface.co/v1/chat/completions")
HF_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"  # Using a supported model
# HF_MODEL = "openai/gpt-oss-120b"
# Try to get token from Streamlit secrets first, then environment variable
try:
    HF_API_TOKEN = st.secrets.get('HUGGINGFACE_API_TOKEN', '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')
except:
    HF_API_TOKEN = os.environ.get('HUGGINGFACE_API_TOKEN', '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')

# Conversation settings
MAX_HISTORY_MESSAGES = 10  # Keep last 10 messages
MAX_CONTEXT_ROWS = 1000  # Sliding window size - send 100 rows at a time
WINDOW_STEP_SIZE = 50  # How many rows to step when sliding window


class ConversationManager:
    """Manages conversation history with truncation"""

    def __init__(self, max_messages: int = MAX_HISTORY_MESSAGES):
        self.max_messages = max_messages

    def add_message(self, role: str, content: str, conversation_history: List[Dict]) -> List[Dict]:
        """Add a message and maintain history limit"""
        conversation_history.append({"role": role, "content": content})

        # Keep only recent messages (sliding window)
        if len(conversation_history) > self.max_messages:
            # Keep system message if it exists, then recent messages
            if conversation_history[0]["role"] == "system":
                conversation_history = [conversation_history[0]] + conversation_history[-(self.max_messages-1):]
            else:
                conversation_history = conversation_history[-self.max_messages:]

        return conversation_history

    def format_conversation_context(self, conversation_history: List[Dict]) -> str:
        """Format conversation history for display"""
        context = ""
        for msg in conversation_history:
            if msg["role"] == "user":
                context += f"User: {msg['content']}\n"
            elif msg["role"] == "assistant":
                context += f"Assistant: {msg['content']}\n"
        return context


class DataSchemaGenerator:
    """Generates schema information for dataframes"""

    @staticmethod
    def get_dataframe_schema(df: pd.DataFrame, df_name: str) -> str:
        """Generate schema description for a dataframe"""
        if df is None or df.empty:
            return f"{df_name}: No data available\n"

        # CRITICAL FIX: Sanitize dataframe first to handle unhashable types (dict, list, etc.)
        # This prevents "TypeError: unhashable type: 'dict'" when calling nunique() or unique()
        try:
            df = DataFilterer.sanitize_dataframe(df)
        except Exception as sanitize_err:
            logger.warning(f"Error sanitizing dataframe in get_dataframe_schema: {sanitize_err}")
            # Continue with original df if sanitization fails

        schema = f"\n{df_name} Schema:\n"
        schema += f"- Total Rows: {len(df)}\n"
        schema += f"- Columns: {list(df.columns)}\n"
        schema += f"- Column Types:\n"

        for col in df.columns:
            dtype = df[col].dtype

            # Safe nunique() call - now works because df is sanitized
            try:
                unique_count = df[col].nunique()
            except Exception as e:
                logger.warning(f"Error getting unique count for column {col}: {e}")
                unique_count = 0

            null_count = df[col].isnull().sum()

            schema += f"  * {col}: {dtype} (unique: {unique_count}, nulls: {null_count})\n"

            # Add sample values for categorical columns
            if unique_count < 20 and unique_count > 0:
                try:
                    sample_values = df[col].dropna().unique()[:10].tolist()
                    schema += f"    Sample values: {sample_values}\n"
                except Exception as e:
                    logger.warning(f"Error getting sample values for column {col}: {e}")

        return schema

    @staticmethod
    def analyze_data_structure(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Deep analysis of dataframe structure to understand data semantics.
        Returns a dictionary with detailed column information.
        """
        if df is None or df.empty:
            return {}

        # CRITICAL FIX: Sanitize dataframe first to handle unhashable types (dict, list, etc.)
        # This prevents "TypeError: unhashable type: 'dict'" when calling nunique() or unique()
        try:
            df = DataFilterer.sanitize_dataframe(df)
        except Exception as sanitize_err:
            logger.warning(f"Error sanitizing dataframe in analyze_data_structure: {sanitize_err}")
            # Continue with original df if sanitization fails

        analysis = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'columns_info': {}
        }

        for col in df.columns:
            # Safe operations with proper error handling
            try:
                unique_count = int(df[col].nunique())
            except Exception as e:
                logger.warning(f"Error getting unique count for column {col}: {e}")
                unique_count = 0

            col_info = {
                'dtype': str(df[col].dtype),
                'unique_count': unique_count,
                'null_count': int(df[col].isnull().sum()),
                'null_percentage': float(df[col].isnull().sum() / len(df) * 100),
                'is_numeric': pd.api.types.is_numeric_dtype(df[col]),
                'is_categorical': unique_count < 50,
                'sample_values': []
            }

            # Add sample values with error handling
            if col_info['unique_count'] < 20 and col_info['unique_count'] > 0:
                try:
                    col_info['sample_values'] = df[col].dropna().unique()[:10].tolist()
                except Exception as e:
                    logger.warning(f"Error getting sample values for column {col}: {e}")
                    col_info['sample_values'] = []

            # Add statistical info for numeric columns
            if col_info['is_numeric']:
                try:
                    col_info['min'] = float(df[col].min()) if not df[col].isna().all() else None
                    col_info['max'] = float(df[col].max()) if not df[col].isna().all() else None
                    col_info['mean'] = float(df[col].mean()) if not df[col].isna().all() else None
                    col_info['median'] = float(df[col].median()) if not df[col].isna().all() else None
                except Exception as e:
                    logger.warning(f"Error calculating statistics for column {col}: {e}")

            analysis['columns_info'][col] = col_info

        # Save analysis to local file
        try:
            analysis_file = 'data_structure_analysis.json'
            with open(analysis_file, 'w') as f:
                json.dump(analysis, f, indent=2, default=str)
            logger.info(f"✅ Data structure analysis saved to {analysis_file}")
        except Exception as e:
            logger.warning(f"Could not save data structure analysis: {e}")

        return analysis

    @staticmethod
    def get_all_schemas(master_df: pd.DataFrame, max_qty_df: pd.DataFrame,
                       min_qty_df: pd.DataFrame, max_buy_df: pd.DataFrame) -> str:
        """Get schemas for all dataframes"""
        schemas = "=== AVAILABLE DATASETS ===\n"
        schemas += DataSchemaGenerator.get_dataframe_schema(master_df, "Master Dataset (kunmings)")
        schemas += DataSchemaGenerator.get_dataframe_schema(max_qty_df, "Maximum Quantity Dataset")
        schemas += DataSchemaGenerator.get_dataframe_schema(min_qty_df, "Minimum Quantity Dataset")
        schemas += DataSchemaGenerator.get_dataframe_schema(max_buy_df, "Maximum Buy Dataset")
        return schemas


class ColumnIdentifier:
    """Identifies relevant columns based on user query"""

    @staticmethod
    def identify_relevant_columns(user_query: str, data_analysis: Dict[str, Any]) -> List[str]:
        """
        Analyze user query to identify which columns are most relevant.
        Uses keyword matching and semantic understanding.
        """
        if not data_analysis or 'columns_info' not in data_analysis:
            return []

        user_query_lower = user_query.lower()
        relevant_columns = []
        column_scores = {}

        # Keywords mapping to potential column relevance
        keyword_patterns = {
            'quantity': ['quantity', 'qty', 'stock', 'inventory', 'count', 'amount'],
            'cost': ['cost', 'price', 'value', 'expense', 'budget'],
            'shape': ['shape', 'cut', 'form', 'type'],
            'color': ['color', 'colour', 'hue', 'shade'],
            'date': ['date', 'time', 'month', 'year', 'period', 'when'],
            'product': ['product', 'item', 'sku', 'id'],
            'bucket': ['bucket', 'category', 'group', 'class'],
            'clarity': ['clarity', 'grade', 'quality'],
            'carat': ['carat', 'weight', 'size'],
            'gap': ['gap', 'difference', 'shortage', 'deficit']
        }

        # Score each column based on query keywords
        for col_name, col_info in data_analysis['columns_info'].items():
            score = 0
            col_name_lower = col_name.lower()

            # Check if column name or keywords appear in query
            for concept, keywords in keyword_patterns.items():
                if any(keyword in user_query_lower for keyword in keywords):
                    if any(keyword in col_name_lower for keyword in keywords):
                        score += 10

            # Boost score if column name appears directly in query
            if col_name_lower in user_query_lower:
                score += 20

            # Boost score for commonly queried columns
            if any(key in col_name_lower for key in ['quantity', 'cost', 'price', 'date', 'shape', 'color']):
                score += 5

            if score > 0:
                column_scores[col_name] = score

        # Get top scored columns (minimum score of 5)
        relevant_columns = [col for col, score in sorted(column_scores.items(),
                           key=lambda x: x[1], reverse=True) if score >= 5]

        # Always include key identifier columns if available
        for key_col in ['Product Id', 'product_id', 'id', 'ID']:
            for col in data_analysis['columns_info'].keys():
                if key_col.lower() in col.lower() and col not in relevant_columns:
                    relevant_columns.insert(0, col)
                    break

        # Log identified columns
        logger.info(f"Identified {len(relevant_columns)} relevant columns: {relevant_columns[:10]}")

        return relevant_columns


class QueryAnalyzer:
    """Analyzes user query to determine data needs"""

    def __init__(self, api_token: str):
        self.api_token = api_token
        self.headers = {"Authorization": f"Bearer {api_token}"}
        self.column_identifier = ColumnIdentifier()

    def analyze_query(self, user_query: str, data_schemas: str,
                     conversation_context: str, data_analysis: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Analyze user query to determine:
        - Which datasets are needed
        - What filters to apply
        - What type of analysis is requested
        - Which columns are relevant
        """

        # Identify relevant columns first
        relevant_columns = []
        if data_analysis:
            relevant_columns = self.column_identifier.identify_relevant_columns(user_query, data_analysis)

        system_prompt = f"""You are a data query analyzer. Your job is to understand the user's query and determine:
1. Which datasets are needed (master_df, max_qty_df, min_qty_df, max_buy_df)
2. What filters should be applied (column name, operator, value)
3. What type of analysis is requested (trend, comparison, summary, prediction, etc.)

Available Data Schemas:
{data_schemas}

Identified Relevant Columns (based on query analysis):
{relevant_columns}

Previous Conversation Context:
{conversation_context}

Respond in JSON format:
{{
    "datasets_needed": ["master_df"],
    "filters": [
        {{"column": "column_name", "operator": "==", "value": "some_value"}},
        {{"column": "another_column", "operator": "<", "value": 2}}
    ],
    "analysis_type": "summary|trend|comparison|prediction|filtering",
    "needs_visualization": true|false,
    "visualization_type": "bar|line|scatter|pie|none",
    "relevant_columns": {relevant_columns}
}}

User Query: {user_query}

Respond ONLY with valid JSON, no additional text."""

        try:
            response = requests.post(
                HF_API_URL,
                headers=self.headers,
                json={
                    "messages": [
                        {
                            "role": "user",
                            "content": system_prompt
                        }
                    ],
                    "model": HF_MODEL,
                    "max_tokens": 500,
                    "temperature": 0.1
                },
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                # New API format: {"choices": [{"message": {"content": "..."}}]}
                if 'choices' in result and len(result['choices']) > 0:
                    generated_text = result['choices'][0]['message']['content']
                    # Try to extract JSON from response
                    try:
                        # Find JSON in the response
                        start_idx = generated_text.find('{')
                        end_idx = generated_text.rfind('}') + 1
                        if start_idx != -1 and end_idx > start_idx:
                            json_str = generated_text[start_idx:end_idx]
                            return json.loads(json_str)
                    except json.JSONDecodeError:
                        logger.error(f"Failed to parse JSON from response: {generated_text}")
            else:
                logger.error(f"API error: {response.status_code} - {response.text}")

            # Return default if parsing fails
            return {
                "datasets_needed": ["master_df"],
                "filters": [],
                "analysis_type": "summary",
                "needs_visualization": False,
                "visualization_type": "none",
                "relevant_columns": relevant_columns
            }

        except Exception as e:
            logger.error(f"Error analyzing query: {e}")
            # Return default analysis
            return {
                "datasets_needed": ["master_df"],
                "filters": [],
                "analysis_type": "summary",
                "needs_visualization": False,
                "visualization_type": "none",
                "relevant_columns": relevant_columns
            }


class DataFilterer:
    """Filters and prepares data based on analysis"""

    @staticmethod
    def sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """
        Sanitize DataFrame to handle unhashable types (dict, list, set, tuple).
        This prevents 'unhashable type' errors during pandas operations.
        """
        if df is None or df.empty:
            return df

        sanitized_df = df.copy()

        for col in sanitized_df.columns:
            try:
                # Check if column might contain unhashable types
                if sanitized_df[col].dtype == 'object':
                    # Check first non-null value
                    first_valid = sanitized_df[col].dropna().head(1)
                    if len(first_valid) > 0:
                        first_value = first_valid.iloc[0]
                        if isinstance(first_value, (dict, list, set, tuple)):
                            logger.warning(f"Column {col} contains unhashable types, converting to JSON strings")
                            sanitized_df[col] = sanitized_df[col].apply(
                                lambda x: json.dumps(x, default=str) if isinstance(x, (dict, list, set, tuple)) else x
                            )
            except Exception as e:
                logger.warning(f"Error checking column {col} for unhashable types: {e}")
                # If we can't check safely, convert to string as fallback
                try:
                    sanitized_df[col] = sanitized_df[col].astype(str)
                except:
                    pass

        return sanitized_df

    @staticmethod
    def apply_filters(df: pd.DataFrame, filters: List[Dict]) -> pd.DataFrame:
        """Apply filters to dataframe"""
        if df is None or df.empty:
            return df

        try:
            # CRITICAL: Sanitize DataFrame first to prevent unhashable type errors
            filtered_df = DataFilterer.sanitize_dataframe(df)

            for filter_dict in filters:
                try:
                    column = filter_dict.get('column')
                    operator = filter_dict.get('operator', '==')
                    value = filter_dict.get('value')

                    if column not in filtered_df.columns:
                        logger.warning(f"Column {column} not found in dataframe")
                        continue

                    # Apply filter based on operator
                    if operator == '==':
                        filtered_df = filtered_df[filtered_df[column] == value]
                    elif operator == '!=':
                        filtered_df = filtered_df[filtered_df[column] != value]
                    elif operator == '<':
                        filtered_df = filtered_df[filtered_df[column] < value]
                    elif operator == '>':
                        filtered_df = filtered_df[filtered_df[column] > value]
                    elif operator == '<=':
                        filtered_df = filtered_df[filtered_df[column] <= value]
                    elif operator == '>=':
                        filtered_df = filtered_df[filtered_df[column] >= value]
                    elif operator == 'contains':
                        filtered_df = filtered_df[filtered_df[column].astype(str).str.contains(str(value), case=False, na=False)]

                except Exception as e:
                    logger.error(f"Error applying filter {filter_dict}: {e}")
                    continue

            return filtered_df

        except Exception as e:
            logger.error(f"Critical error in apply_filters: {e}")
            logger.exception("Full traceback:")
            # Return sanitized version as fallback
            return DataFilterer.sanitize_dataframe(df)

    @staticmethod
    def prepare_data_context(filtered_df: pd.DataFrame, max_rows: int = MAX_CONTEXT_ROWS,
                           relevant_columns: List[str] = None, window_start: int = 0) -> str:
        """
        Prepare filtered data as context for LLM using sliding window approach.

        Args:
            filtered_df: DataFrame to process
            max_rows: Maximum rows per window (window size)
            relevant_columns: List of relevant column names to focus on
            window_start: Starting row index for the sliding window

        Returns:
            String representation of the windowed data with context
        """
        if filtered_df is None or filtered_df.empty:
            return "No data found matching the filters."

        try:
            total_rows = len(filtered_df)

            # Focus on relevant columns if specified
            if relevant_columns:
                available_cols = [col for col in relevant_columns if col in filtered_df.columns]
                if available_cols:
                    logger.info(f"Focusing on {len(available_cols)} relevant columns: {available_cols}")
                    data_to_send = filtered_df[available_cols].copy()
                else:
                    data_to_send = filtered_df.copy()
            else:
                data_to_send = filtered_df.copy()

            # CRITICAL: Sanitize data to handle unhashable types (dict, list, etc.)
            for col in data_to_send.columns:
                try:
                    # Check if column contains unhashable types
                    if data_to_send[col].dtype == 'object':
                        # Convert any dict/list values to strings
                        data_to_send[col] = data_to_send[col].apply(
                            lambda x: json.dumps(x, default=str) if isinstance(x, (dict, list, set, tuple)) else x
                        )
                except Exception as col_err:
                    logger.warning(f"Error sanitizing column {col}: {col_err}")
                    # Fallback: convert entire column to string
                    data_to_send[col] = data_to_send[col].astype(str)

            # SLIDING WINDOW IMPLEMENTATION
            # Calculate window boundaries
            if max_rows is None or max_rows <= 0:
                # If no limit, use all data (backward compatibility)
                window_start = 0
                window_end = total_rows
                windowed_data = data_to_send
            else:
                # Apply sliding window
                window_start = max(0, min(window_start, total_rows - 1))
                window_end = min(window_start + max_rows, total_rows)
                windowed_data = data_to_send.iloc[window_start:window_end].copy()

            # Prepare context with window information
            context = f"Dataset Window (showing rows {window_start+1}-{window_end} of {total_rows} total rows):\n"

            if window_start > 0:
                context += f"⬅️ Previous {window_start} rows not shown\n"
            if window_end < total_rows:
                remaining = total_rows - window_end
                context += f"➡️ Next {remaining} rows available\n"

            context += "\n"

            # Show which columns are being focused on
            if relevant_columns and available_cols:
                context += f"Focusing on columns: {', '.join(available_cols)}\n\n"

            # Convert windowed data to readable format
            context += windowed_data.to_string(index=False)

            # Add summary statistics for the ENTIRE dataset (not just window)
            context += f"\n\n=== Summary Statistics (Full Dataset - {total_rows} rows) ===\n"
            numeric_cols = data_to_send.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                context += data_to_send[numeric_cols].describe().to_string()

            # Add window-specific statistics
            context += f"\n\n=== Window Statistics (Current {len(windowed_data)} rows) ===\n"
            window_numeric_cols = windowed_data.select_dtypes(include=['number']).columns
            if len(window_numeric_cols) > 0:
                context += windowed_data[window_numeric_cols].describe().to_string()

            logger.info(f"Prepared data context with sliding window: rows {window_start+1}-{window_end} of {total_rows}")

            return context

        except Exception as e:
            logger.error(f"Error preparing data context: {e}")
            # Return a safe fallback
            return f"Dataset with {len(filtered_df)} rows and {len(filtered_df.columns)} columns. Columns: {', '.join(map(str, filtered_df.columns))}"


class ResponseGenerator:
    """Generates responses using LLM with data context"""

    def __init__(self, api_token: str, max_retries: int = 3, retry_delay: float = 2.0):
        self.api_token = api_token
        self.headers = {"Authorization": f"Bearer {api_token}"}
        self.max_retries = max_retries
        self.retry_delay = retry_delay  # Initial delay in seconds for exponential backoff

    def generate_response(self, user_query: str, data_context: str,
                         conversation_history: List[Dict], analysis_info: Dict,
                         stream: bool = False, thinking_placeholder=None,
                         response_placeholder=None) -> str:
        """Generate response based on user query and data context

        Args:
            user_query: User's question
            data_context: Data context to analyze
            conversation_history: Previous conversation
            analysis_info: Analysis metadata
            stream: Whether to stream the response
            thinking_placeholder: Streamlit placeholder for thinking content
            response_placeholder: Streamlit placeholder for response content
        """

        # Build conversation context
        conv_context = ""
        for msg in conversation_history[-5:]:  # Last 5 messages
            if msg["role"] == "user":
                conv_context += f"User: {msg['content']}\n"
            elif msg["role"] == "assistant":
                conv_context += f"Assistant: {msg['content']}\n"

        system_prompt = f"""You are an intelligent data analyst assistant for a diamond inventory cost dashboard.
You help users understand their inventory data, identify trends, and make business decisions.

Previous Conversation:
{conv_context}

Current Query: {user_query}

Available Data:
{data_context}

Analysis Type: {analysis_info.get('analysis_type', 'summary')}

Based on the data provided above, answer the user's query with insights and recommendations.
Be concise but informative. If you see trends or patterns, mention them.
If you recommend actions, explain why.

IMPORTANT: Before answering, think through your reasoning step by step. Start your response with:
<thinking>
[Your step-by-step reasoning here - analyze the data, identify patterns, formulate insights]
</thinking>

Then provide your final answer after the thinking section."""

        try:
            if stream and thinking_placeholder and response_placeholder:
                # Streaming mode
                return self._generate_response_streaming(
                    system_prompt,
                    thinking_placeholder,
                    response_placeholder
                )
            else:
                # Non-streaming mode (backward compatible)
                response = requests.post(
                    HF_API_URL,
                    headers=self.headers,
                    json={
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are an intelligent data analyst assistant for a diamond inventory cost dashboard."
                            },
                            {
                                "role": "user",
                                "content": system_prompt
                            }
                        ],
                        "model": HF_MODEL,
                        "max_tokens": 800,
                        "temperature": 0.7
                    },
                    timeout=60
                )

                if response.status_code == 200:
                    result = response.json()
                    # New API format: {"choices": [{"message": {"content": "..."}}]}
                    if 'choices' in result and len(result['choices']) > 0:
                        full_response = result['choices'][0]['message']['content']
                        # Extract thinking and response parts
                        return self._extract_final_response(full_response)
                    else:
                        logger.error(f"Unexpected API response format: {result}")
                        return 'I apologize, but I could not generate a response.'
                else:
                    logger.error(f"API error: {response.status_code} - {response.text}")
                    return f"I encountered an error while processing your request. Please try again."

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"I encountered an error: {str(e)}"

        return "I apologize, but I could not generate a response. Please try again."

    def _generate_response_streaming(self, system_prompt: str,
                                    thinking_placeholder, response_placeholder) -> str:
        """Generate streaming response with thinking display"""
        try:
            response = requests.post(
                HF_API_URL,
                headers=self.headers,
                json={
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are an intelligent data analyst assistant for a diamond inventory cost dashboard."
                        },
                        {
                            "role": "user",
                            "content": system_prompt
                        }
                    ],
                    "model": HF_MODEL,
                    "max_tokens": 1200,
                    "temperature": 0.7,
                    "stream": True  # Enable streaming
                },
                timeout=120,
                stream=True  # Important: stream the response
            )

            if response.status_code == 200:
                full_text = ""
                thinking_text = ""
                response_text = ""
                in_thinking = False

                # Process streaming response
                for line in response.iter_lines():
                    if line:
                        line = line.decode('utf-8')
                        if line.startswith('data: '):
                            data_str = line[6:]  # Remove 'data: ' prefix
                            if data_str.strip() == '[DONE]':
                                break

                            try:
                                data = json.loads(data_str)
                                if 'choices' in data and len(data['choices']) > 0:
                                    delta = data['choices'][0].get('delta', {})
                                    content = delta.get('content', '')

                                    if content:
                                        full_text += content

                                        # Parse thinking tags
                                        if '<thinking>' in full_text and not in_thinking:
                                            in_thinking = True
                                            thinking_text = full_text.split('<thinking>')[1]
                                        elif '</thinking>' in full_text and in_thinking:
                                            in_thinking = False
                                            parts = full_text.split('</thinking>')
                                            thinking_text = full_text.split('<thinking>')[1].split('</thinking>')[0]
                                            response_text = parts[1] if len(parts) > 1 else ""
                                        elif in_thinking:
                                            thinking_text = full_text.split('<thinking>')[1]
                                        else:
                                            if '</thinking>' in full_text:
                                                response_text = full_text.split('</thinking>')[1]
                                            elif '<thinking>' not in full_text:
                                                response_text = full_text

                                        # Update UI in real-time
                                        if thinking_text.strip():
                                            thinking_placeholder.markdown(
                                                f"**🔍 Analyzing Query & Data...**\n\n{thinking_text.strip()}"
                                            )

                                        if response_text.strip():
                                            response_placeholder.markdown(response_text.strip())

                            except json.JSONDecodeError:
                                continue

                # Return final response (without thinking tags)
                return self._extract_final_response(full_text)
            else:
                logger.error(f"Streaming API error: {response.status_code}")
                return f"I encountered an error while processing your request."

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            return f"I encountered an error: {str(e)}"

    def _extract_final_response(self, full_response: str) -> str:
        """Extract the final response without thinking tags"""
        from deque_window_processor import extract_thinking_and_response

        _, clean_response = extract_thinking_and_response(full_response)
        return clean_response

    def _extract_thinking_and_response(self, full_response: str) -> Tuple[Optional[str], str]:
        """Extract both thinking and response for storage in chat history"""
        from deque_window_processor import extract_thinking_and_response

        thinking, clean_response = extract_thinking_and_response(full_response)
        return thinking, clean_response

    def process_window_batch(self, user_query: str, window_data: str,
                            window_number: int, total_windows: int,
                            stream: bool = False, thinking_placeholder=None) -> str:
        """
        Process a single window of data and generate meaningful intermediate result with retry logic.
        This is the MAP phase of map-reduce.

        Args:
            user_query: Original user query
            window_data: Data for this specific window
            window_number: Current window index (1-based)
            total_windows: Total number of windows
            stream: Enable streaming with thinking display
            thinking_placeholder: Streamlit placeholder for thinking display

        Returns:
            Meaningful intermediate analysis result for this window (not garbage)
        """
        system_prompt = f"""You are a data analyst extracting meaningful insights from a data window.

User's Question: "{user_query}"

Window Data (Window {window_number}/{total_windows}):
{window_data}

YOUR TASK: Extract ONLY meaningful insights from this window that directly help answer the user's question.

IMPORTANT INSTRUCTIONS:
1. First, show your thought process in <thinking> tags:
   <thinking>
   - What patterns do I see in this data?
   - What calculations are needed?
   - What's relevant to the user's query?
   </thinking>

2. Then provide your analysis:
   - Focus ONLY on data relevant to the user's query
   - Provide specific numbers, values, and facts
   - Identify clear patterns, trends, or anomalies
   - Calculate precise statistics (totals, averages, min/max, counts)
   - If no relevant data, state that clearly and briefly
   - DO NOT provide a final answer - this is intermediate analysis only
   - Be precise and factual - no fluff

Output Format:
<thinking>
[Your reasoning process here]
</thinking>

Key Findings: [List specific relevant findings with numbers]
Statistics: [Precise calculations if applicable]
Notable Patterns: [Only if clear patterns exist in this window]"""

        # Dynamically calculate max_tokens
        base_tokens = 350
        data_complexity_bonus = min(250, len(window_data) // 500)
        query_complexity_bonus = min(150, len(user_query.split()) * 5)
        max_tokens = base_tokens + data_complexity_bonus + query_complexity_bonus

        logger.info(f"   Window {window_number}: Using {max_tokens} max_tokens for analysis")

        # Retry logic with exponential backoff
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    HF_API_URL,
                    headers=self.headers,
                    json={
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are a data analyst extracting meaningful insights. Show your thinking in <thinking> tags."
                            },
                            {
                                "role": "user",
                                "content": system_prompt
                            }
                        ],
                        "model": HF_MODEL,
                        "max_tokens": max_tokens,
                        "temperature": 0.1
                    },
                    timeout=60
                )

                if response.status_code == 200:
                    result = response.json()
                    if 'choices' in result and len(result['choices']) > 0:
                        window_result = result['choices'][0]['message']['content'].strip()

                        # Display thinking if streaming
                        if stream and thinking_placeholder:
                            window_result = self._display_thinking_if_present(
                                window_result, thinking_placeholder, window_number
                            )
                        else:
                            # Extract final response without thinking tags
                            window_result = self._extract_final_response(window_result)

                        # Filter out meaningless responses
                        if window_result and len(window_result) > 20:
                            logger.info(f"✅ Window {window_number}/{total_windows} processed successfully")
                            return window_result
                        else:
                            logger.warning(f"Window {window_number}: Response too short")
                            return f"Window {window_number}: No meaningful insights found for the query in this data segment."
                    else:
                        logger.error(f"Unexpected API response for window {window_number}")

                # Handle specific error codes
                elif response.status_code == 402:
                    error_msg = self._handle_payment_required_error(response, window_number, attempt)
                    last_error = error_msg
                    if attempt < self.max_retries - 1:
                        logger.warning(f"   Retrying window {window_number} after 402 error (attempt {attempt + 1}/{self.max_retries})")
                        continue
                    return error_msg

                elif response.status_code in [429, 503]:  # Rate limit or service unavailable
                    if attempt < self.max_retries - 1:
                        wait_time = self.retry_delay * (2 ** attempt)
                        logger.warning(f"   Status {response.status_code} for window {window_number}, waiting {wait_time}s before retry")
                        time.sleep(wait_time)
                        continue
                    last_error = f"Window {window_number}: API overloaded (status {response.status_code}). Please try again later."
                    return last_error

                else:
                    logger.error(f"API error for window {window_number}: {response.status_code} - {response.text[:200]}")
                    return f"Window {window_number}: Could not analyze - API returned status {response.status_code}"

            except requests.exceptions.Timeout:
                logger.error(f"Window {window_number}: Request timeout (attempt {attempt + 1})")
                last_error = f"Window {window_number}: Request timeout"
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))
                    continue

            except Exception as e:
                logger.error(f"Error processing window {window_number} (attempt {attempt + 1}): {e}")
                last_error = f"Window {window_number}: Analysis error - {str(e)}"
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))
                    continue

        # All retries exhausted
        return last_error or f"Window {window_number}: Failed after {self.max_retries} attempts"

    def _handle_payment_required_error(self, response, window_number: int, attempt: int) -> str:
        """Handle 402 Payment Required error with detailed information"""
        try:
            error_details = response.json() if response.text else {}
            error_message = error_details.get('error', {}).get('message', 'Payment or quota issue')
        except:
            error_message = 'Payment or quota issue'

        logger.error(f"⚠️  PAYMENT REQUIRED (402) for window {window_number}")
        logger.error(f"   Error details: {error_message}")
        logger.error(f"   API Response: {response.text[:500]}")
        logger.error(f"   Possible causes:")
        logger.error(f"   - HuggingFace API token quota exhausted")
        logger.error(f"   - Billing tier limit reached")
        logger.error(f"   - Invalid or expired API token")
        logger.error(f"   - Model requires paid tier")

        return f"""Window {window_number}: ⚠️ API Payment/Quota Issue

**Error**: {error_message}

**Possible Solutions**:
1. Check HuggingFace API token quota at https://huggingface.co/settings/tokens
2. Verify billing status and upgrade if needed
3. Try a different model (free tier available)
4. Use a different API token
5. Wait for quota reset

**Technical Details**: HTTP 402 (Payment Required) - Attempt {attempt + 1}"""

    def _display_thinking_if_present(self, text: str, thinking_placeholder, window_number: int) -> str:
        """Extract and display thinking tags if present, return text without thinking"""
        if '<thinking>' in text and '</thinking>' in text:
            thinking_content = text.split('<thinking>')[1].split('</thinking>')[0].strip()
            response_content = text.split('</thinking>')[1].strip()

            if thinking_content and thinking_placeholder:
                thinking_placeholder.markdown(
                    f"**📊 Analyzing Data Window {window_number}...**\n\n{thinking_content}"
                )

            return response_content
        return text

    def generate_final_response_from_deque(self, user_query: str,
                                           deque_results: deque,
                                           conversation_history: List[Dict],
                                           analysis_info: Dict,
                                           stream: bool = False,
                                           thinking_placeholder=None,
                                           response_placeholder=None) -> str:
        """
        Generate comprehensive final response from accumulated deque results.
        This is the REDUCE phase of map-reduce.

        The LLM takes the complete deque as context and performs comprehensive analysis
        to give a meaningful, well-structured answer to the user.

        Args:
            user_query: Original user query
            deque_results: Deque containing all meaningful intermediate window results
            conversation_history: Previous conversation messages
            analysis_info: Query analysis metadata

        Returns:
            Final comprehensive answer based on complete deque analysis
        """
        # Build conversation context
        conv_context = ""
        for msg in conversation_history[-5:]:  # Last 5 messages
            if msg["role"] == "user":
                conv_context += f"User: {msg['content']}\n"
            elif msg["role"] == "assistant":
                conv_context += f"Assistant: {msg['content']}\n"

        # Combine all window results from deque - only meaningful ones
        accumulated_insights = "\n\n".join([
            f"=== Segment {idx+1} Analysis ===\n{result}"
            for idx, result in enumerate(deque_results)
        ])

        system_prompt = f"""You are an expert data analyst for a diamond inventory cost dashboard providing a comprehensive final answer.

CONTEXT:
Previous Conversation:
{conv_context}

User's Question: "{user_query}"

Analysis Type: {analysis_info.get('analysis_type', 'comprehensive analysis')}

Total Data Segments Analyzed: {len(deque_results)}

ACCUMULATED INSIGHTS FROM ALL DATA SEGMENTS:
{accumulated_insights}

YOUR TASK - Provide a COMPREHENSIVE FINAL ANSWER:

1. ANSWER THE QUESTION DIRECTLY:
   - Start with a clear, direct answer to the user's query
   - Use specific numbers, values, and facts from the accumulated insights

2. SYNTHESIZE CROSS-SEGMENT INSIGHTS:
   - Identify patterns and trends that emerge across ALL segments
   - Compare and contrast findings from different segments
   - Calculate aggregate statistics and distributions

3. PROVIDE COMPLETE ANALYSIS:
   - Key findings with supporting data (quantities, prices, trends)
   - Notable patterns or anomalies in the inventory
   - Statistical summaries relevant to the query
   - Business implications if applicable

4. ACTIONABLE RECOMMENDATIONS:
   - Based on the data, suggest specific actions
   - For inventory issues: recommend restocking, reducing stock, etc.
   - For trends: highlight opportunities or risks

5. STRUCTURE YOUR RESPONSE:
   - Use clear sections with markdown formatting
   - Lead with the most important information
   - Use bullet points for clarity
   - Be thorough but concise

IMPORTANT: This is the FINAL answer the user will see. It must be:
- Complete and self-contained (user sees ONLY this answer)
- Based on ALL the accumulated insights from the deque
- Clear, professional, and actionable
- NOT just a summary of windows, but a synthesized answer to their question

Before providing your final answer, think through your reasoning step by step. Start with:
<thinking>
[Your analytical reasoning: how you're synthesizing the data, what patterns you see, your logic]
</thinking>

Then provide your final comprehensive answer."""

        try:
            # Calculate appropriate max_tokens based on accumulated insights and query complexity
            base_tokens = 700
            insights_bonus = min(700, len(accumulated_insights) // 100)  # Scale with amount of data
            query_complexity = min(300, len(user_query.split()) * 10)  # Complex questions need detailed answers
            max_tokens = base_tokens + insights_bonus + query_complexity

            logger.info(f"   REDUCE phase: Using {max_tokens} max_tokens for final synthesis")

            if stream and thinking_placeholder and response_placeholder:
                # Streaming mode
                return self._generate_final_response_streaming(
                    system_prompt,
                    max_tokens,
                    thinking_placeholder,
                    response_placeholder,
                    deque_results,
                    accumulated_insights
                )

            response = requests.post(
                HF_API_URL,
                headers=self.headers,
                json={
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are an expert data analyst who synthesizes insights from large datasets to provide comprehensive, actionable answers. Focus on clarity and value."
                        },
                        {
                            "role": "user",
                            "content": system_prompt
                        }
                    ],
                    "model": HF_MODEL,
                    "max_tokens": max_tokens,
                    "temperature": 0.4,  # Balanced for synthesis with accuracy
                },
                timeout=90  # Longer timeout for comprehensive synthesis
            )

            if response.status_code == 200:
                result = response.json()
                if 'choices' in result and len(result['choices']) > 0:
                    final_answer = result['choices'][0]['message']['content'].strip()

                    # Extract final response without thinking tags
                    final_answer = self._extract_final_response(final_answer)

                    # Validate that we got a substantial answer
                    if final_answer and len(final_answer) > 50:
                        logger.info(f"✅ Final comprehensive response generated: {len(final_answer)} characters")
                        return final_answer
                    else:
                        logger.warning("Final answer too short, using fallback")
                else:
                    logger.error(f"Unexpected API response format: {result}")

                # Fallback: return structured accumulated insights
                return f"""Based on comprehensive analysis of {len(deque_results)} data segments:

{accumulated_insights}

---
Note: Complete synthesis unavailable. Above are the detailed findings from each data segment."""

            else:
                logger.error(f"API error: {response.status_code} - {response.text}")
                return f"""Analysis completed for {len(deque_results)} segments (API error {response.status_code}):

{accumulated_insights}

---
Unable to generate synthesized answer due to API error, but detailed segment analysis is shown above."""

        except Exception as e:
            logger.error(f"Error generating final response: {e}")
            return f"""Comprehensive analysis completed for {len(deque_results)} segments:

{accumulated_insights}

---
Note: Synthesis step encountered an error ({str(e)}), but detailed findings from all segments are shown above."""

    def _generate_final_response_streaming(self, system_prompt: str, max_tokens: int,
                                          thinking_placeholder, response_placeholder,
                                          deque_results, accumulated_insights) -> str:
        """Generate streaming final response with thinking display"""
        try:
            response = requests.post(
                HF_API_URL,
                headers=self.headers,
                json={
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are an expert data analyst who synthesizes insights from large datasets to provide comprehensive, actionable answers."
                        },
                        {
                            "role": "user",
                            "content": system_prompt
                        }
                    ],
                    "model": HF_MODEL,
                    "max_tokens": max_tokens,
                    "temperature": 0.4,
                    "stream": True  # Enable streaming
                },
                timeout=120,
                stream=True
            )

            if response.status_code == 200:
                full_text = ""
                thinking_text = ""
                response_text = ""
                in_thinking = False

                # Process streaming response
                for line in response.iter_lines():
                    if line:
                        line = line.decode('utf-8')
                        if line.startswith('data: '):
                            data_str = line[6:]
                            if data_str.strip() == '[DONE]':
                                break

                            try:
                                data = json.loads(data_str)
                                if 'choices' in data and len(data['choices']) > 0:
                                    delta = data['choices'][0].get('delta', {})
                                    content = delta.get('content', '')

                                    if content:
                                        full_text += content

                                        # Parse thinking tags
                                        if '<thinking>' in full_text and not in_thinking:
                                            in_thinking = True
                                            thinking_text = full_text.split('<thinking>')[1]
                                        elif '</thinking>' in full_text and in_thinking:
                                            in_thinking = False
                                            parts = full_text.split('</thinking>')
                                            thinking_text = full_text.split('<thinking>')[1].split('</thinking>')[0]
                                            response_text = parts[1] if len(parts) > 1 else ""
                                        elif in_thinking:
                                            thinking_text = full_text.split('<thinking>')[1]
                                        else:
                                            if '</thinking>' in full_text:
                                                response_text = full_text.split('</thinking>')[1]
                                            elif '<thinking>' not in full_text:
                                                response_text = full_text

                                        # Update UI in real-time
                                        if thinking_text.strip():
                                            thinking_placeholder.markdown(
                                                f"**🔄 Synthesizing Final Answer...**\n\n{thinking_text.strip()}"
                                            )

                                        if response_text.strip():
                                            response_placeholder.markdown(response_text.strip())

                            except json.JSONDecodeError:
                                continue

                # Extract and return final response (without thinking tags)
                if '</thinking>' in full_text:
                    parts = full_text.split('</thinking>')
                    return parts[1].strip() if len(parts) > 1 else full_text
                return full_text

            else:
                logger.error(f"Streaming API error: {response.status_code}")
                return f"""Analysis completed for {len(deque_results)} segments:

{accumulated_insights}"""

        except Exception as e:
            logger.error(f"Streaming error in final response: {e}")
            return f"""Comprehensive analysis completed for {len(deque_results)} segments:

{accumulated_insights}"""


class ChatbotOrchestrator:
    """Main orchestrator that coordinates all chatbot components"""

    def __init__(self, api_token: str):
        self.conversation_manager = ConversationManager()
        self.schema_generator = DataSchemaGenerator()
        self.query_analyzer = QueryAnalyzer(api_token)
        self.data_filterer = DataFilterer()
        self.response_generator = ResponseGenerator(api_token)
        self.data_analyzed = False  # Track if initial data analysis is complete

        # Initialize enhanced components
        self.enhanced_analyzer = EnhancedQueryAnalyzer(api_token, HF_API_URL, HF_MODEL)
        self.deque_processor = DequeWindowProcessor(api_token, HF_API_URL, HF_MODEL, window_size=MAX_CONTEXT_ROWS)

        # Streaming placeholders
        self.streaming_enabled = False
        self.thinking_placeholder = None
        self.response_placeholder = None

        # Store last thinking content for persistence in chat history
        self.last_thinking_content = None

    def enable_streaming(self, thinking_placeholder=None, response_placeholder=None):
        """Enable streaming mode with placeholders for thinking and response"""
        self.streaming_enabled = True
        self.thinking_placeholder = thinking_placeholder
        self.response_placeholder = response_placeholder
        logger.info("Streaming mode enabled with placeholders")

    def get_last_thinking(self) -> Optional[str]:
        """Get the last thinking content captured during processing"""
        return self.last_thinking_content

    def clear_last_thinking(self):
        """Clear the stored thinking content"""
        self.last_thinking_content = None

    def initialize_data_understanding(self, master_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform initial data understanding and save to local machine.
        This should be called once when the chatbot is first loaded.
        """
        if self.data_analyzed:
            return {}

        logger.info("🔍 Performing initial data analysis from kunmings.pkl...")
        data_analysis = self.schema_generator.analyze_data_structure(master_df)

        # Create a summary report
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_rows': data_analysis.get('total_rows', 0),
            'total_columns': data_analysis.get('total_columns', 0),
            'column_names': list(data_analysis.get('columns_info', {}).keys()),
            'numeric_columns': [col for col, info in data_analysis.get('columns_info', {}).items()
                              if info.get('is_numeric', False)],
            'categorical_columns': [col for col, info in data_analysis.get('columns_info', {}).items()
                                  if info.get('is_categorical', False)],
            'columns_with_nulls': [col for col, info in data_analysis.get('columns_info', {}).items()
                                 if info.get('null_count', 0) > 0]
        }

        # Save summary
        try:
            summary_file = 'kunmings_data_summary.json'
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            logger.info(f"✅ Data summary saved to {summary_file}")
            print(f"📊 Data understanding complete! Summary saved to: {summary_file}")
        except Exception as e:
            logger.warning(f"Could not save data summary: {e}")

        self.data_analyzed = True
        return data_analysis

    def process_query_enhanced(self, user_query: str, master_df: pd.DataFrame,
                               max_qty_df: pd.DataFrame, min_qty_df: pd.DataFrame,
                               max_buy_df: pd.DataFrame, conversation_history: List[Dict]) -> Tuple[str, Optional[pd.DataFrame], Dict]:
        """
        ENHANCED query processing with smart column/value identification and deque sliding window.

        WORKFLOW:
        1. LLM identifies if query mentions a COLUMN NAME or a VALUE
           Example: "Analyze trend of Cushion Diamond"
           - LLM sees "Cushion" is not a column, but a VALUE in "Shape key" column
           - Result: filter_column="Shape key", filter_value="Cushion"

        2. Filter the dataframe based on identified column and value
        3. Apply deque sliding window processing (Map-Reduce):
           - MAP: Process each window → LLM → intermediate result → append to deque
           - REDUCE: All deque results → LLM → final comprehensive answer

        Args:
            user_query: User's natural language question
            master_df, max_qty_df, min_qty_df, max_buy_df: DataFrames
            conversation_history: Previous messages

        Returns:
            (final_answer, filtered_df, analysis_info)
        """
        try:
            logger.info("=" * 80)
            logger.info("🚀 ENHANCED QUERY PROCESSING with Smart Column/Value Identification")
            logger.info("=" * 80)

            # Step 1: Analyze data structure
            logger.info("📊 Step 1: Analyzing data structure...")
            if self.thinking_placeholder:
                self.thinking_placeholder.markdown(
                    "**Step 1/7:** 📊 Analyzing data structure...\n\n"
                    "- Understanding dataset schema\n"
                    "- Identifying available columns and data types"
                )
            data_analysis = self.schema_generator.analyze_data_structure(master_df)
            logger.info(f"   Data: {data_analysis['total_rows']} rows, {data_analysis['total_columns']} columns")

            # Step 2: Use enhanced query analyzer to identify intent and filter parameters
            logger.info("🔍 Step 2: Analyzing query intent and identifying filters...")
            if self.thinking_placeholder:
                self.thinking_placeholder.markdown(
                    "**Step 2/7:** 🔍 Understanding your question...\n\n"
                    f"- Query: \"{user_query}\"\n"
                    "- Identifying filter criteria and analysis intent\n"
                    "- Determining which columns are relevant"
                )
            query_understanding = self.enhanced_analyzer.analyze_query(user_query, data_analysis)

            intent = query_understanding.get('intent', 'analysis')
            filter_column = query_understanding.get('filter_column')
            filter_value = query_understanding.get('filter_value')
            operation_type = query_understanding.get('operation_type')

            logger.info(f"   Query Understanding:")
            logger.info(f"   - Intent: {intent}")
            logger.info(f"   - Filter Column: {filter_column}")
            logger.info(f"   - Filter Value: {filter_value}")
            logger.info(f"   - Operation: {operation_type}")
            logger.info(f"   - Reasoning: {query_understanding.get('reasoning', 'N/A')}")

            # Determine which dataframe to use
            primary_df = master_df  # Default to master

            # Step 3: Handle based on intent
            if intent == "filter_only":
                # FILTER_ONLY: Just filter and return data without deque processing
                logger.info("📋 Step 3: Filter-only query detected - skipping deque processing")
                logger.info("   Applying filter(s) and returning data directly...")
                if self.thinking_placeholder:
                    self.thinking_placeholder.markdown(
                        "**Step 3/7:** 📋 Applying filters to dataset...\n\n"
                        f"- Filter column: {filter_column}\n"
                        f"- Filter value: {filter_value}\n"
                        "- This is a simple filter query - no deep analysis needed"
                    )

                # Get all filters from query_understanding
                filters = query_understanding.get('filters', [])

                # Apply all filters using the DataFilterer method
                if filters:
                    try:
                        filtered_df = self.data_filterer.apply_filters(primary_df, filters)
                        logger.info(f"   ✅ Applied {len(filters)} filter(s)")
                        for filter_item in filters:
                            logger.info(f"      - {filter_item['column']} {filter_item.get('operator', '==')} {filter_item['value']}")
                        logger.info(f"   Result: {len(filtered_df)} rows")

                        # Generate simple response
                        if len(filtered_df) > 0:
                            if len(filters) == 1:
                                final_answer = f"Found {len(filtered_df)} records where {filters[0]['column']} is '{filters[0]['value']}'.\n\n"
                            else:
                                final_answer = f"Found {len(filtered_df)} records matching the following filters:\n"
                                for filter_item in filters:
                                    final_answer += f"- {filter_item['column']} = {filter_item['value']}\n"
                                final_answer += "\n"

                            final_answer += f"**Data Summary:**\n"
                            final_answer += f"- Total Records: {len(filtered_df)}\n"
                            final_answer += f"- Filters Applied: {len(filters)}\n"
                            for filter_item in filters:
                                final_answer += f"  - {filter_item['column']} = {filter_item['value']}\n"
                            final_answer += "\nThe filtered data is displayed below."
                        else:
                            if len(filters) == 1:
                                final_answer = f"No records found where {filters[0]['column']} is '{filters[0]['value']}'."
                            else:
                                final_answer = f"No records found matching all the specified filters."
                    except Exception as filter_err:
                        logger.error(f"Error applying filters: {filter_err}")
                        filtered_df = pd.DataFrame()
                        final_answer = f"Error applying filters: {str(filter_err)}"
                else:
                    # No filter specified, return all data
                    filtered_df = primary_df.copy()
                    final_answer = f"Showing all {len(filtered_df)} records from the dataset."
                    logger.info(f"   No specific filter - returning all {len(filtered_df)} rows")

                logger.info("=" * 80)
                logger.info(f"✅ FILTER-ONLY PROCESSING COMPLETE")
                logger.info(f"   Filtered rows: {len(filtered_df)}")
                logger.info("=" * 80)

                # Build analysis info for return
                analysis_info = {
                    "intent": intent,
                    "filter_column": filter_column,
                    "filter_value": filter_value,
                    "filters": filters,  # Include all filters
                    "operation_type": "filter_only",
                    "analysis_focus": "display",
                    "windows_processed": 0,
                    "total_rows": len(filtered_df)
                }

                return final_answer, filtered_df, analysis_info

            else:
                # ANALYSIS: Apply full deque window processing
                logger.info("🔄 Step 3: Analysis query detected - starting deque sliding window processing...")
                if self.thinking_placeholder:
                    self.thinking_placeholder.markdown(
                        "**Step 3/7:** 🔄 Processing data with sliding window analysis...\n\n"
                        "- Breaking data into manageable windows\n"
                        "- Each window will be analyzed independently\n"
                        "- Results will be synthesized into final answer"
                    )

                # Use deque processor with streaming enabled
                final_answer, filtered_df, results_deque = self.deque_processor.process_with_deque(
                    user_query=user_query,
                    df=primary_df,
                    filter_column=filter_column,
                    filter_value=filter_value,
                    relevant_columns=None,  # Process all columns
                    stream=self.streaming_enabled,  # Enable streaming for REDUCE phase
                    thinking_placeholder=self.thinking_placeholder,  # For displaying thinking in UI
                    response_placeholder=self.response_placeholder  # For displaying response in UI
                )

                logger.info("=" * 80)
                logger.info(f"✅ ANALYSIS PROCESSING COMPLETE")
                logger.info(f"   Windows processed: {len(results_deque)}")
                logger.info(f"   Filtered rows: {len(filtered_df)}")
                logger.info(f"   Final answer length: {len(final_answer)} characters")
                logger.info(f"   Final answer preview: {final_answer[:200] if final_answer else 'EMPTY'}...")
                logger.info("=" * 80)

                # Capture thinking from deque processor (it was extracted and stored during processing)
                if hasattr(self.deque_processor, 'last_thinking_content') and self.deque_processor.last_thinking_content:
                    self.last_thinking_content = self.deque_processor.last_thinking_content
                    logger.info(f"   Captured thinking content from processor: {len(self.last_thinking_content)} characters")

                # Build analysis info for return
                analysis_info = {
                    "intent": intent,
                    "filter_column": filter_column,
                    "filter_value": filter_value,
                    "operation_type": operation_type,
                    "analysis_focus": query_understanding.get('analysis_focus'),
                    "windows_processed": len(results_deque),
                    "total_rows": len(filtered_df)
                }

                return final_answer, filtered_df, analysis_info

        except Exception as e:
            logger.error(f"❌ Error in enhanced query processing: {e}")
            logger.exception("Full traceback:")
            error_msg = f"I encountered an error processing your query: {str(e)}. Please try rephrasing."
            return error_msg, pd.DataFrame(), {"error": str(e)}

    def process_query_with_deque_window(self, user_query: str, master_df: pd.DataFrame,
                                        max_qty_df: pd.DataFrame, min_qty_df: pd.DataFrame,
                                        max_buy_df: pd.DataFrame, conversation_history: List[Dict]) -> Tuple[str, Optional[pd.DataFrame], Dict]:
        """
        Process user query using deque-based sliding window approach (Map-Reduce pattern).

        WORKFLOW:
        1. Filter dataframe based on user query
        2. Split filtered data into windows
        3. MAP PHASE: Process each window → LLM → intermediate result → append to deque
        4. REDUCE PHASE: All deque results → LLM → final answer

        Args:
            user_query: The user's question
            master_df, max_qty_df, min_qty_df, max_buy_df: DataFrames to query
            conversation_history: Previous conversation messages

        Returns:
            (response_text, filtered_dataframe, analysis_info)
        """
        try:
            logger.info("🔄 Starting deque-based sliding window processing...")

            # Step 1: Analyze data structure
            logger.info("Analyzing data structure...")
            data_analysis = self.schema_generator.analyze_data_structure(master_df)

            # Step 2: Generate schemas
            data_schemas = self.schema_generator.get_all_schemas(
                master_df, max_qty_df, min_qty_df, max_buy_df
            )

            # Step 3: Get conversation context
            conv_context = self.conversation_manager.format_conversation_context(conversation_history)

            # Step 4: Analyze query
            logger.info("Analyzing user query...")
            analysis_info = self.query_analyzer.analyze_query(
                user_query, data_schemas, conv_context, data_analysis
            )
            logger.info(f"Query analysis: {analysis_info.get('analysis_type', 'unknown')}")

            # Step 5: Get appropriate dataset
            datasets = {
                'master_df': master_df,
                'max_qty_df': max_qty_df,
                'min_qty_df': min_qty_df,
                'max_buy_df': max_buy_df
            }

            datasets_needed = analysis_info.get('datasets_needed', ['master_df'])
            primary_dataset = datasets.get(datasets_needed[0], master_df)

            # Step 6: Apply filters
            logger.info("Applying filters to dataset...")
            filters = analysis_info.get('filters', [])
            try:
                filtered_df = self.data_filterer.apply_filters(primary_dataset, filters)
                logger.info(f"✅ Filtered dataset: {len(filtered_df)} rows")
            except Exception as filter_err:
                logger.error(f"Error applying filters: {filter_err}")
                filtered_df = self.data_filterer.sanitize_dataframe(primary_dataset)

            if filtered_df.empty:
                return "No data found matching your filters.", filtered_df, analysis_info

            # Step 7: DEQUE SLIDING WINDOW PROCESSING
            total_rows = len(filtered_df)
            relevant_columns = analysis_info.get('relevant_columns', [])

            # Focus on relevant columns
            if relevant_columns:
                available_cols = [col for col in relevant_columns if col in filtered_df.columns]
                if available_cols:
                    data_to_process = filtered_df[available_cols].copy()
                else:
                    data_to_process = filtered_df.copy()
            else:
                data_to_process = filtered_df.copy()

            # Sanitize data
            data_to_process = self.data_filterer.sanitize_dataframe(data_to_process)

            # Calculate windows
            window_size = MAX_CONTEXT_ROWS
            num_windows = (total_rows + window_size - 1) // window_size  # Ceiling division

            logger.info(f"📊 Processing {total_rows} rows in {num_windows} windows (window size: {window_size})")

            # Initialize deque for storing intermediate results
            results_deque = deque()

            # MAP PHASE: Process each window
            logger.info("🗺️  MAP PHASE: Processing windows...")
            for window_idx in range(num_windows):
                window_start = window_idx * window_size
                window_end = min(window_start + window_size, total_rows)

                logger.info(f"Processing window {window_idx + 1}/{num_windows} (rows {window_start + 1}-{window_end})...")

                # Extract window data
                window_df = data_to_process.iloc[window_start:window_end].copy()

                # Convert window to string representation
                window_context = f"Rows {window_start + 1}-{window_end} of {total_rows}:\n"
                window_context += window_df.to_string(index=False)

                # Add statistics for this window
                numeric_cols = window_df.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    window_context += "\n\nWindow Statistics:\n"
                    window_context += window_df[numeric_cols].describe().to_string()

                # Process window through LLM (MAP)
                window_result = self.response_generator.process_window_batch(
                    user_query=user_query,
                    window_data=window_context,
                    window_number=window_idx + 1,
                    total_windows=num_windows
                )

                # Append result to deque
                results_deque.append(window_result)
                logger.info(f"✅ Window {window_idx + 1}/{num_windows} result added to deque")

            logger.info(f"✅ MAP PHASE complete. Deque contains {len(results_deque)} results")

            # REDUCE PHASE: Generate final answer from deque
            logger.info("🔽 REDUCE PHASE: Generating final answer from deque...")
            final_response = self.response_generator.generate_final_response_from_deque(
                user_query=user_query,
                deque_results=results_deque,
                conversation_history=conversation_history,
                analysis_info=analysis_info,
                stream=self.streaming_enabled,
                thinking_placeholder=self.thinking_placeholder,
                response_placeholder=self.response_placeholder
            )

            logger.info("✅ REDUCE PHASE complete. Final answer generated.")

            return final_response, filtered_df, analysis_info

        except Exception as e:
            logger.error(f"Critical error in process_query_with_deque_window: {e}")
            logger.exception("Full traceback:")
            error_response = f"I encountered an error processing your query: {str(e)}. Please try rephrasing."
            return error_response, pd.DataFrame(), {"analysis_type": "error", "error": str(e)}

    def process_query(self, user_query: str, master_df: pd.DataFrame,
                     max_qty_df: pd.DataFrame, min_qty_df: pd.DataFrame,
                     max_buy_df: pd.DataFrame, conversation_history: List[Dict],
                     window_start: int = 0) -> Tuple[str, Optional[pd.DataFrame], Dict]:
        """
        Process user query end-to-end with sliding window support.

        Args:
            user_query: The user's question
            master_df, max_qty_df, min_qty_df, max_buy_df: DataFrames to query
            conversation_history: Previous conversation messages
            window_start: Starting row index for sliding window (default: 0)

        Returns:
            (response_text, filtered_dataframe, analysis_info)
        """

        try:
            # Step 0: Analyze data structure from master dataset
            logger.info("Analyzing data structure from master dataset...")
            data_analysis = self.schema_generator.analyze_data_structure(master_df)

            # Step 1: Generate data schemas
            data_schemas = self.schema_generator.get_all_schemas(
                master_df, max_qty_df, min_qty_df, max_buy_df
            )

            # Step 2: Get conversation context
            conv_context = self.conversation_manager.format_conversation_context(conversation_history)

            # Step 3: Analyze query to determine data needs and identify relevant columns
            logger.info("Analyzing user query...")
            analysis_info = self.query_analyzer.analyze_query(
                user_query, data_schemas, conv_context, data_analysis
            )
            logger.info(f"Query analysis complete: {analysis_info.get('analysis_type', 'unknown')}")

            # Step 4: Get appropriate datasets
            datasets = {
                'master_df': master_df,
                'max_qty_df': max_qty_df,
                'min_qty_df': min_qty_df,
                'max_buy_df': max_buy_df
            }

            # Use master_df by default
            datasets_needed = analysis_info.get('datasets_needed', ['master_df'])
            primary_dataset = datasets.get(datasets_needed[0], master_df)

            # Step 5: Apply filters with error handling
            logger.info("Applying filters to dataset...")
            filters = analysis_info.get('filters', [])
            try:
                filtered_df = self.data_filterer.apply_filters(primary_dataset, filters)
                logger.info(f"Filtered dataset has {len(filtered_df)} rows")
            except Exception as filter_err:
                logger.error(f"Error applying filters: {filter_err}")
                logger.exception("Filter error traceback:")
                # Use sanitized version of original dataset as fallback
                filtered_df = self.data_filterer.sanitize_dataframe(primary_dataset)

            # Step 6: Prepare data context for LLM with focus on relevant columns and sliding window
            logger.info(f"Preparing data context for LLM with sliding window starting at row {window_start}...")
            relevant_columns = analysis_info.get('relevant_columns', [])
            try:
                data_context = self.data_filterer.prepare_data_context(
                    filtered_df,
                    max_rows=MAX_CONTEXT_ROWS,
                    relevant_columns=relevant_columns,
                    window_start=window_start
                )
            except Exception as context_err:
                logger.error(f"Error preparing data context: {context_err}")
                logger.exception("Context error traceback:")
                data_context = f"Dataset with {len(filtered_df)} rows and {len(filtered_df.columns)} columns."

            # Step 7: Generate response using LLM
            logger.info("Generating response from LLM...")
            try:
                response = self.response_generator.generate_response(
                    user_query, data_context, conversation_history, analysis_info,
                    stream=self.streaming_enabled,
                    thinking_placeholder=self.thinking_placeholder,
                    response_placeholder=self.response_placeholder
                )
            except Exception as response_err:
                logger.error(f"Error generating response: {response_err}")
                logger.exception("Response error traceback:")
                response = f"I found {len(filtered_df)} matching records but encountered an error generating the response. Please try rephrasing your question."

            # Log the workflow
            logger.info(f"Query processed successfully with {len(relevant_columns)} relevant columns")

            return response, filtered_df, analysis_info

        except Exception as e:
            logger.error(f"Critical error in process_query: {e}")
            logger.exception("Full traceback:")
            # Return error response with empty results
            error_response = f"I encountered an error processing your query: {str(e)}. Please try rephrasing your question."
            return error_response, pd.DataFrame(), {"analysis_type": "error", "error": str(e)}


def _convert_dataframe_to_safe_dict(df: pd.DataFrame, max_rows: int = None) -> dict:
    """
    Convert DataFrame to a fully JSON-serializable dict without numpy types.
    Format: {column-name: [column values]}
    This ensures compatibility with Streamlit's session state and caching.
    """
    import numpy as np

    try:
        # DEBUG: Save original dataframe info
        debug_info = {
            'original_shape': df.shape,
            'original_columns': list(df.columns),
            'column_types': {str(col): str(df[col].dtype) for col in df.columns},
            'timestamp': str(datetime.now())
        }

        # Save debug info to file
        debug_file = 'chatbot_debug_dataframe_original.json'
        with open(debug_file, 'w') as f:
            json.dump(debug_info, f, indent=2, default=str)
        logger.info(f"Saved original dataframe debug info to {debug_file}")
    except Exception as e:
        logger.warning(f"Could not save debug info: {e}")

    # Use whole dataframe - no row limits
    limited_df = df.copy()

    # Reset index to ensure it's a simple integer index
    limited_df = limited_df.reset_index(drop=True)

    # Convert DataFrame to dictionary format: {column-name: [column values]}
    result = {}

    for col in limited_df.columns:
        try:
            # Ensure column name is string
            col_str = str(col)

            # Convert column values to list with proper type handling
            col_values = []

            for value in limited_df[col]:
                try:
                    # CRITICAL: Handle dict, list, set, tuple values (unhashable types)
                    if isinstance(value, (dict, list, set, tuple)):
                        # Convert unhashable types to JSON string
                        col_values.append(json.dumps(value, default=str))
                    # Handle different data types
                    elif pd.isna(value):
                        col_values.append(None)
                    elif isinstance(value, (np.integer, np.int64, np.int32)):
                        col_values.append(int(value))
                    elif isinstance(value, (np.floating, np.float64, np.float32)):
                        if np.isinf(value):
                            col_values.append(None)
                        else:
                            col_values.append(float(value))
                    elif isinstance(value, np.bool_):
                        col_values.append(bool(value))
                    elif isinstance(value, (pd.Timestamp, datetime)):
                        col_values.append(str(value))
                    elif isinstance(value, (str, int, float, bool)):
                        col_values.append(value)
                    else:
                        # Convert everything else to string
                        col_values.append(str(value))
                except Exception as val_err:
                    logger.warning(f"Error converting value in column {col_str}: {val_err}")
                    col_values.append(str(value) if value is not None else None)

            result[col_str] = col_values

        except Exception as e:
            logger.error(f"Error processing column {col}: {e}")
            # Fallback: convert entire column to string list
            result[str(col)] = [str(v) if pd.notna(v) else None for v in limited_df[col]]

    # Add metadata
    result_with_metadata = {
        'columns': result,
        'row_count': int(len(df)),
        'column_names': list(result.keys())
    }

    # IMPORTANT: Save the dictionary to local machine as requested by user
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f'dataframe_dict_{timestamp}.json'
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        logger.info(f"✅ Dictionary saved to local machine: {output_file}")
        print(f"📁 Data dictionary saved to: {output_file}")
        print(f"📋 Format: {{column-name: [column values]}}")
    except Exception as e:
        logger.warning(f"Could not save dictionary to local machine: {e}")

    # DEBUG: Save converted result with metadata
    try:
        debug_file = 'chatbot_debug_dataframe_converted.json'
        with open(debug_file, 'w') as f:
            json.dump(result_with_metadata, f, indent=2, default=str)
        logger.info(f"Saved converted dataframe to {debug_file}")
    except Exception as e:
        logger.warning(f"Could not save converted result: {e}")

    return result_with_metadata


def _sanitize_for_session_state(obj: Any) -> Any:
    """
    Recursively sanitize objects to ensure they're JSON-serializable.
    Converts DataFrames, numpy types, and other complex objects to simple Python types.
    """
    import numpy as np

    try:
        if obj is None:
            return None
        elif isinstance(obj, pd.DataFrame):
            # Convert DataFrame to serializable dict
            with open("data.json", "w") as f:
                json.dump(_convert_dataframe_to_safe_dict(obj),f)
            return _convert_dataframe_to_safe_dict(obj)
        elif isinstance(obj, dict):
            # Recursively sanitize dictionary values
            result = {}

            # CRITICAL FIX: Safely iterate over items, handling corrupt dictionaries
            try:
                items = list(obj.items())
            except Exception as items_err:
                # If we can't iterate (e.g., unhashable keys), try alternative approach
                logger.error(f"Cannot iterate over dictionary items: {items_err}")
                # Convert entire dict to string as last resort
                return str(obj)

            for idx, (k, v) in enumerate(items):
                # CRITICAL: Ensure key is ALWAYS a string (never dict, list, or other unhashable)
                # This is the main cause of "unhashable type: 'dict'" errors
                try:
                    if isinstance(k, (dict, list, set, tuple)):
                        # If key is unhashable, use index-based naming instead of hashing
                        safe_key = f"col_{idx}_{str(type(k).__name__)}"
                        logger.warning(f"Found unhashable key type {type(k)}, converting to {safe_key}")
                    elif k is None:
                        safe_key = f"col_{idx}_none"
                    else:
                        safe_key = str(k)
                except Exception as key_err:
                    # Ultimate fallback for key conversion
                    safe_key = f"col_{idx}"
                    logger.error(f"Error processing key at index {idx}: {key_err}")

                # Sanitize the value
                try:
                    if isinstance(v, (np.integer, np.floating)):
                        result[safe_key] = float(v) if isinstance(v, np.floating) else int(v)
                    elif isinstance(v, np.bool_):
                        result[safe_key] = bool(v)
                    elif isinstance(v, np.ndarray):
                        result[safe_key] = v.tolist()
                    elif isinstance(v, pd.DataFrame):
                        # Convert nested DataFrames
                        with open("data.json", "w") as f:
                            json.dump(_convert_dataframe_to_safe_dict(v),f)
                        result[safe_key] = _convert_dataframe_to_safe_dict(v)
                    else:
                        result[safe_key] = _sanitize_for_session_state(v)
                except Exception as e:
                    logger.warning(f"Error sanitizing value for key {safe_key}: {e}, converting to string")
                    try:
                        result[safe_key] = str(v)
                    except:
                        result[safe_key] = "error_converting_value"
            return result
        elif isinstance(obj, (list, tuple)):
            # Recursively sanitize list/tuple items
            return [_sanitize_for_session_state(item) for item in obj]
        elif isinstance(obj, set):
            # Convert set to list (sets are not JSON-serializable)
            return [_sanitize_for_session_state(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            # Convert numpy numeric types to Python types
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (int, float, str, bool)):
            # These types are already serializable
            return obj
        elif isinstance(obj, (datetime,)):
            # Convert datetime to string
            return str(obj)
        else:
            # Convert other types to string
            logger.debug(f"Converting unknown type {type(obj)} to string")
            return str(obj)
    except Exception as e:
        logger.error(f"Error in _sanitize_for_session_state: {e}, object type: {type(obj)}")
        # Ultimate fallback - return string representation
        return str(obj)


def render_chatbot_sidebar(master_df: pd.DataFrame, max_qty_df: pd.DataFrame,
                           min_qty_df: pd.DataFrame, max_buy_df: pd.DataFrame):
    """
    Render chatbot interface in sidebar
    """

    # Initialize session state for chatbot
    if 'chatbot_history' not in st.session_state:
        st.session_state.chatbot_history = []

    # Initialize sliding window position
    if 'chatbot_window_start' not in st.session_state:
        st.session_state.chatbot_window_start = 0

    # Clean up any legacy data that might cause unhashable errors
    # This ensures all messages are properly serialized
    try:
        if st.session_state.chatbot_history:
            # DEBUG: Save raw history before cleaning
            try:
                debug_file = 'chatbot_debug_history_raw.json'
                with open(debug_file, 'w') as f:
                    json.dump(st.session_state.chatbot_history, f, indent=2, default=str)
                logger.info(f"Saved raw chat history to {debug_file}")
            except Exception as debug_err:
                logger.warning(f"Could not save raw history: {debug_err}")

            cleaned_history = []
            for idx, msg in enumerate(st.session_state.chatbot_history):
                try:
                    if not isinstance(msg, dict):
                        logger.warning(f"Skipping non-dict message at index {idx}: {type(msg)}")
                        continue

                    # Create a clean copy with only required fields
                    cleaned_msg = {
                        'role': str(msg.get('role', 'user')),
                        'content': str(msg.get('content', ''))
                    }

                    # Handle data_preview if it exists and is valid
                    if 'data_preview' in msg and msg['data_preview'] is not None:
                        preview = msg['data_preview']
                        logger.info(f"Processing data_preview at index {idx}, type: {type(preview)}")

                        # DEBUG: Save the preview data
                        try:
                            debug_preview_file = f'chatbot_debug_preview_{idx}.json'
                            with open(debug_preview_file, 'w') as f:
                                json.dump(preview, f, indent=2, default=str)
                            logger.info(f"Saved preview data to {debug_preview_file}")
                        except Exception as debug_err:
                            logger.warning(f"Could not save preview data: {debug_err}")

                        # Accept both old and new format
                        if isinstance(preview, dict) and ('columns' in preview or 'data' in preview):
                            # Sanitize the preview data to ensure no unhashable types
                            try:
                                cleaned_msg['data_preview'] = _sanitize_for_session_state(preview)
                                logger.info(f"Successfully sanitized data_preview at index {idx}")
                            except Exception as sanitize_err:
                                logger.error(f"Error sanitizing preview at index {idx}: {sanitize_err}")
                                # Skip this preview if sanitization fails
                        else:
                            # Skip invalid preview data
                            logger.warning(f"Skipping invalid data_preview format at index {idx}")

                    # Sanitize the entire message to ensure compatibility
                    cleaned_msg = _sanitize_for_session_state(cleaned_msg)
                    cleaned_history.append(cleaned_msg)
                except Exception as msg_err:
                    logger.error(f"Error processing message at index {idx}: {msg_err}")
                    # Skip this message but continue with others
                    continue

            # DEBUG: Save cleaned history
            try:
                debug_file = 'chatbot_debug_history_cleaned.json'
                with open(debug_file, 'w') as f:
                    json.dump(cleaned_history, f, indent=2, default=str)
                logger.info(f"Saved cleaned chat history to {debug_file}")
            except Exception as debug_err:
                logger.warning(f"Could not save cleaned history: {debug_err}")

            st.session_state.chatbot_history = cleaned_history
    except Exception as e:
        logger.error(f"Error cleaning chat history, resetting: {e}")
        logger.exception("Full traceback:")
        st.session_state.chatbot_history = []

    if 'chatbot_enabled' not in st.session_state:
        st.session_state.chatbot_enabled = True  # Enable by default

    if 'chatbot_initialized' not in st.session_state:
        st.session_state.chatbot_initialized = False

    # Chatbot toggle
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🤖 AI Assistant")

    # Check if API token is available and valid
    if not HF_API_TOKEN or HF_API_TOKEN == "your_token_here":
        st.sidebar.warning("⚠️ HuggingFace API token not configured. Using default token.")
        st.sidebar.info("For better performance, get your own token from https://huggingface.co/settings/tokens")
        # Don't return - allow using the fallback token

    # Toggle chatbot
    chatbot_enabled = st.sidebar.checkbox("Enable AI Assistant", value=st.session_state.chatbot_enabled)
    st.session_state.chatbot_enabled = chatbot_enabled

    if not chatbot_enabled:
        return

    # Show data availability status
    if master_df is None or master_df.empty:
        st.sidebar.info("📊 No data loaded. Upload an Excel file to enable data-based queries.")
    else:
        st.sidebar.success(f"📊 Data loaded: {len(master_df):,} records")

    # Initialize data understanding on first load
    if not st.session_state.chatbot_initialized and master_df is not None and not master_df.empty:
        try:
            with st.spinner("🔍 Understanding data structure from kunmings.pkl..."):
                orchestrator = ChatbotOrchestrator(HF_API_TOKEN)
                orchestrator.initialize_data_understanding(master_df)
                st.session_state.chatbot_initialized = True
                # st.sidebar.success("✅ Data structure analyzed successfully!")
        except Exception as e:
            logger.error(f"Error initializing data understanding: {e}")
            st.sidebar.warning("⚠️ Could not analyze data structure")

    # Display chat history in sidebar
    st.sidebar.markdown("#### Chat")

    # Chat container with scrollable area
    chat_container = st.sidebar.container()

    with chat_container:
        # Display conversation history
        for idx, message in enumerate(st.session_state.chatbot_history):
            try:
                role = message.get("role", "user")
                content = message.get("content", "")

                if role == "user":
                    st.markdown(f"**You:** {content}")
                elif role == "assistant":
                    st.markdown(f"**AI:** {content}")

                    # Show thinking section if available (collapsible, like Claude)
                    if "thinking" in message and message["thinking"]:
                        thinking_content = message["thinking"]
                        with st.expander("💭 View AI Thinking Process", expanded=False):
                            st.markdown(f"```\n{thinking_content}\n```")

                # Show data preview if available
                if "data_preview" in message and message["data_preview"] is not None:
                    preview_data = message["data_preview"]

                    # Handle sanitized dict format - NEW FORMAT: {column-name: [values]}
                    if isinstance(preview_data, dict):
                        try:
                            logger.info(f"Attempting to display data preview for message {idx}")

                            # DEBUG: Save preview data being rendered
                            try:
                                debug_render_file = f'chatbot_debug_render_{idx}.json'
                                with open(debug_render_file, 'w') as f:
                                    json.dump(preview_data, f, indent=2, default=str)
                                logger.info(f"Saved render preview data to {debug_render_file}")
                            except Exception as debug_err:
                                logger.warning(f"Could not save render debug data: {debug_err}")

                            # Convert from {column-name: [values]} format to DataFrame
                            if 'columns' in preview_data:
                                # New format: {'columns': {col: [values]}, 'row_count': N}
                                preview_df = pd.DataFrame(preview_data['columns'])
                                row_count = preview_data.get('row_count', len(preview_df))
                            elif 'data' in preview_data:
                                # Old format: {'data': [row_dicts], 'row_count': N}
                                preview_df = pd.DataFrame(preview_data['data'])
                                row_count = preview_data.get('row_count', len(preview_df))
                            else:
                                # Direct format: {column: [values]}
                                preview_df = pd.DataFrame(preview_data)
                                row_count = len(preview_df)

                            logger.info(f"Preview DataFrame shape: {preview_df.shape}, dtypes: {preview_df.dtypes.to_dict()}")

                            if not preview_df.empty:
                                # Clean the dataframe to ensure all values are Python native types
                                # This prevents MediaFileHandler errors with unhashable types
                                for col in preview_df.columns:
                                    try:
                                        col_dtype = preview_df[col].dtype
                                        logger.debug(f"Processing column {col} with dtype {col_dtype}")

                                        # Convert any remaining numpy types to Python types
                                        if col_dtype == 'object':
                                            # For object columns, ensure everything is a string
                                            preview_df[col] = preview_df[col].apply(
                                                lambda x: str(x) if x is not None else None
                                            )
                                        else:
                                            # Convert numeric columns to native Python types
                                            preview_df[col] = preview_df[col].apply(
                                                lambda x: x.item() if hasattr(x, 'item') else (x if x is not None else None)
                                            )
                                    except Exception as col_err:
                                        logger.error(f"Error processing column {col}: {col_err}")
                                        # Fallback: convert to string
                                        preview_df[col] = preview_df[col].astype(str)

                                # Create a clean copy for display
                                display_df = preview_df.head(10).copy()

                                # DEBUG: Save the display dataframe
                                try:
                                    debug_display_file = f'chatbot_debug_display_{idx}.csv'
                                    display_df.to_csv(debug_display_file, index=False)
                                    logger.info(f"Saved display DataFrame to {debug_display_file}")
                                except Exception as debug_err:
                                    logger.warning(f"Could not save display DataFrame: {debug_err}")

                                with st.expander(f"📊 Data Preview ({row_count} rows total)"):
                                    # Use key parameter to avoid caching issues
                                    # Ensure key uses only hashable types (str, int)
                                    # Use timestamp to ensure uniqueness
                                    safe_key = f"preview_{int(idx)}_{int(row_count)}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
                                    logger.info(f"Displaying dataframe with key: {safe_key}")

                                    try:
                                        st.dataframe(
                                            display_df,
                                            use_container_width=True,
                                            key=safe_key
                                        )
                                    except Exception as display_err:
                                        logger.error(f"Error displaying dataframe: {display_err}")
                                        # Fallback: show as text
                                        st.text(display_df.to_string())
                        except Exception as e:
                            logger.error(f"Could not display data preview at index {idx}: {e}")
                            logger.exception("Full traceback:")
                            st.warning(f"Could not display data preview: {str(e)}")

            except Exception as e:
                logger.warning(f"Error displaying message {idx}: {e}")
                continue


    # Chat input
    user_input = st.sidebar.text_input(
        "Ask me anything about your inventory:",
        key="chatbot_input",
        placeholder="e.g., Show stocks with quantity < 2"
    )

    # Send button
    col1, col2 = st.sidebar.columns([3, 1])

    with col1:
        send_button = st.button("Send", key="chatbot_send", use_container_width=True)

    with col2:
        clear_button = st.button("🗑️", key="chatbot_clear", help="Clear chat history")

    # Clear chat history
    if clear_button:
        st.session_state.chatbot_history = []
        st.rerun()

    # Process user input
    if send_button and user_input.strip():
        try:
            # Ensure chatbot_history is a list
            if not isinstance(st.session_state.chatbot_history, list):
                st.session_state.chatbot_history = []

            # Add user message to history (sanitized)
            logger.info(f"Adding user message to history: {user_input[:100]}...")
            user_message = {
                "role": "user",
                "content": str(user_input).strip()
            }

            # Sanitize to ensure compatibility with session state
            try:
                user_message = _sanitize_for_session_state(user_message)
                logger.info("Successfully sanitized user message")
            except Exception as sanitize_err:
                logger.error(f"Error sanitizing user message: {sanitize_err}")
                # Fallback to simple string message
                user_message = {
                    "role": "user",
                    "content": str(user_input).strip()
                }

            st.session_state.chatbot_history.append(user_message)
            logger.info(f"Chat history now has {len(st.session_state.chatbot_history)} messages")

            # Show processing indicator with progress info
            with st.spinner("Thinking"):
                # Import thinking display component
                from thinking_display import create_thinking_placeholder

                # Create placeholder for thinking (will be shown in expander)
                thinking_container = st.sidebar.empty()

                # Create thinking display and wrapper
                thinking_display, thinking_wrapper = create_thinking_placeholder(
                    container=thinking_container,
                    title="🧠 View AI Thinking Process",
                    expanded=True
                )

                # Create placeholder for response
                response_placeholder = st.sidebar.empty()

                # Will be populated during streaming
                try:
                    # Initialize orchestrator
                    orchestrator = ChatbotOrchestrator(HF_API_TOKEN)

                    # Enable streaming in orchestrator with proper thinking display
                    orchestrator.enable_streaming(
                        thinking_placeholder=thinking_wrapper,
                        response_placeholder=response_placeholder
                    )

                    # Process query with ENHANCED analyzer (identifies column vs value) + deque window
                    response, filtered_df, analysis_info = orchestrator.process_query_enhanced(
                        user_input,
                        master_df,
                        max_qty_df,
                        min_qty_df,
                        max_buy_df,
                        st.session_state.chatbot_history
                    )

                    # Add assistant response to history with fully sanitized data
                    # Convert DataFrame to serializable dict to avoid unhashable type errors
                    data_preview_dict = None
                    if filtered_df is not None and not filtered_df.empty:
                        logger.info(f"Converting filtered DataFrame to safe dict. Shape: {filtered_df.shape}")

                        # DEBUG: Save filtered dataframe info
                        try:
                            debug_filtered_file = 'chatbot_debug_filtered_df.csv'
                            filtered_df.head(100).to_csv(debug_filtered_file, index=False)
                            logger.info(f"Saved filtered DataFrame to {debug_filtered_file}")
                        except Exception as debug_err:
                            logger.warning(f"Could not save filtered DataFrame: {debug_err}")

                        # Use helper function to ensure proper conversion
                        # This prevents "unhashable type: 'dict'" and MediaFileHandler errors
                        try:
                            with open("data.json", "w") as f:
                                json.dump(_convert_dataframe_to_safe_dict(filtered_df), f)
                            data_preview_dict = _convert_dataframe_to_safe_dict(filtered_df)
                            logger.info(f"Successfully converted DataFrame to safe dict with {data_preview_dict['row_count']} total rows and {len(data_preview_dict.get('columns', {}))} columns")

                            # DEBUG: Verify the converted dict is JSON-serializable
                            try:
                                json_test = json.dumps(data_preview_dict)
                                logger.info(f"Verified data_preview_dict is JSON-serializable (size: {len(json_test)} chars)")
                            except Exception as json_err:
                                logger.error(f"data_preview_dict is NOT JSON-serializable: {json_err}")
                                data_preview_dict = None
                        except Exception as convert_err:
                            logger.error(f"Error converting DataFrame to safe dict: {convert_err}")
                            logger.exception("Full traceback:")
                            data_preview_dict = None

                    # Create assistant message with only JSON-serializable data
                    assistant_message = {
                        "role": "assistant",
                        "content": str(response)
                    }

                    # Add thinking content if it was captured
                    thinking_content = orchestrator.get_last_thinking()
                    if thinking_content:
                        logger.info(f"Adding thinking content to assistant message ({len(thinking_content)} chars)")
                        assistant_message["thinking"] = str(thinking_content)
                        # Clear after storing
                        orchestrator.clear_last_thinking()

                    # Only add data_preview if it exists and is valid
                    if data_preview_dict is not None:
                        logger.info("Adding data_preview to assistant message")
                        assistant_message["data_preview"] = data_preview_dict

                        # DEBUG: Save the assistant message before sanitization
                        try:
                            debug_msg_file = 'chatbot_debug_assistant_msg_before.json'
                            with open(debug_msg_file, 'w') as f:
                                json.dump(assistant_message, f, indent=2, default=str)
                            logger.info(f"Saved assistant message (before sanitization) to {debug_msg_file}")
                        except Exception as debug_err:
                            logger.warning(f"Could not save assistant message before: {debug_err}")

                    # Sanitize the entire message to ensure no unhashable types
                    try:
                        logger.info("Sanitizing assistant message for session state")
                        assistant_message = _sanitize_for_session_state(assistant_message)
                        logger.info("Successfully sanitized assistant message")

                        # DEBUG: Save the assistant message after sanitization
                        try:
                            debug_msg_file = 'chatbot_debug_assistant_msg_after.json'
                            with open(debug_msg_file, 'w') as f:
                                json.dump(assistant_message, f, indent=2, default=str)
                            logger.info(f"Saved assistant message (after sanitization) to {debug_msg_file}")
                        except Exception as debug_err:
                            logger.warning(f"Could not save assistant message after: {debug_err}")

                        # Verify it's JSON-serializable before adding to history
                        try:
                            json.dumps(assistant_message)
                            logger.info("Verified assistant message is JSON-serializable")
                        except Exception as json_err:
                            logger.error(f"Assistant message is NOT JSON-serializable after sanitization: {json_err}")
                            # Remove data_preview if it's causing issues
                            assistant_message.pop("data_preview", None)
                    except Exception as sanitize_err:
                        logger.error(f"Error sanitizing assistant message: {sanitize_err}")
                        logger.exception("Full traceback:")
                        # Fallback to simple message without preview
                        assistant_message = {
                            "role": "assistant",
                            "content": str(response)
                        }

                    logger.info("Appending assistant message to chat history")
                    st.session_state.chatbot_history.append(assistant_message)

                    # Maintain history limit
                    if len(st.session_state.chatbot_history) > MAX_HISTORY_MESSAGES:
                        st.session_state.chatbot_history = st.session_state.chatbot_history[-MAX_HISTORY_MESSAGES:]

                except Exception as e:
                    error_msg = f"Chatbot error: {str(e)}"
                    logger.error(error_msg)

                    # Add error message to history (sanitized)
                    # Ensure error message is simple string only
                    error_response = {
                        "role": "assistant",
                        "content": f"I encountered an error: {str(e)}. Please try again or rephrase your question."
                    }

                    # Sanitize error response to ensure compatibility
                    error_response = _sanitize_for_session_state(error_response)

                    # Ensure chatbot_history is still a list before appending
                    if isinstance(st.session_state.chatbot_history, list):
                        st.session_state.chatbot_history.append(error_response)
                    else:
                        st.session_state.chatbot_history = [error_response]

        except Exception as outer_error:
            # Catch any errors in message handling itself
            error_str = str(outer_error)
            logger.error(f"Error handling chat message: {error_str}")
            st.error(f"An error occurred processing your message: {error_str}. Please try clearing the chat and trying again.")

        # Rerun to update UI
        st.rerun()

    # Help section
    with st.sidebar.expander("💡 Example Questions"):
        st.markdown("""
        - Show me stocks with quantity less than 2
        - What are the top 10 items by cost?
        - Analyze trends for Cushion diamonds
        - Which shapes are most profitable?
        - Compare Round vs Princess diamonds
        - What should I focus on buying?
        """)


def render_chatbot_main(master_df: pd.DataFrame, max_qty_df: pd.DataFrame,
                        min_qty_df: pd.DataFrame, max_buy_df: pd.DataFrame):
    """
    Render chatbot interface in main page area with toggle button at top
    Always renders UI regardless of data availability
    """

    # Ensure all dataframes are valid (not None)
    if master_df is None:
        master_df = pd.DataFrame()
    if max_qty_df is None:
        max_qty_df = pd.DataFrame()
    if min_qty_df is None:
        min_qty_df = pd.DataFrame()
    if max_buy_df is None:
        max_buy_df = pd.DataFrame()

    logger.info(f"render_chatbot_main called with master_df shape: {master_df.shape}")

    # Session state is now initialized early in main app - no need to reinitialize
    # This ensures chatbot state persists across tab switches and filter interactions

    # Clean up any legacy data that might cause unhashable errors
    try:
        if st.session_state.chatbot_history:
            cleaned_history = []
            for idx, msg in enumerate(st.session_state.chatbot_history):
                try:
                    if not isinstance(msg, dict):
                        logger.warning(f"Skipping non-dict message at index {idx}: {type(msg)}")
                        continue

                    # Create a clean copy with only required fields
                    cleaned_msg = {
                        'role': str(msg.get('role', 'user')),
                        'content': str(msg.get('content', ''))
                    }

                    # Handle data_preview if it exists and is valid
                    if 'data_preview' in msg and msg['data_preview'] is not None:
                        preview = msg['data_preview']
                        if isinstance(preview, dict) and ('columns' in preview or 'data' in preview):
                            try:
                                cleaned_msg['data_preview'] = _sanitize_for_session_state(preview)
                            except Exception as sanitize_err:
                                logger.error(f"Error sanitizing preview at index {idx}: {sanitize_err}")

                    cleaned_msg = _sanitize_for_session_state(cleaned_msg)
                    cleaned_history.append(cleaned_msg)
                except Exception as msg_err:
                    logger.error(f"Error processing message at index {idx}: {msg_err}")
                    continue

            st.session_state.chatbot_history = cleaned_history
    except Exception as e:
        logger.error(f"Error cleaning chat history, resetting: {e}")
        st.session_state.chatbot_history = []

    # Session state variables are now initialized early in main app
    # This ensures all chatbot state persists across interactions

    # ALWAYS display chatbot UI - this section should NEVER be skipped
    st.markdown("---")

    # Check if API token is available and valid
    if not HF_API_TOKEN or HF_API_TOKEN == "your_token_here":
        st.warning("⚠️ HuggingFace API token not configured. Using default token.")
        st.info("For better performance, get your own token from https://huggingface.co/settings/tokens")

    # Initialize data understanding on first load (only if data exists)
    # This runs in background and doesn't block UI rendering
    if not st.session_state.chatbot_initialized and master_df is not None and not master_df.empty:
        try:
            with st.spinner("🔍 Understanding data structure from kunmings.pkl..."):
                orchestrator = ChatbotOrchestrator(HF_API_TOKEN)
                orchestrator.initialize_data_understanding(master_df)
                st.session_state.chatbot_initialized = True
                st.success("✅ Data structure analyzed successfully!")
        except Exception as e:
            logger.error(f"Error initializing data understanding: {e}")
            st.warning("⚠️ Could not analyze data structure")
            # Continue anyway - chatbot can still function for general queries

    # Chat input at top - ALWAYS render this UI element regardless of data state
    st.markdown("---")
    user_input = st.text_input(
        "Ask me anything about your inventory:",
        key="chatbot_input_main",
        placeholder="e.g., Show stocks with quantity < 2"
    )

    logger.info(f"Chat input rendered successfully")

    # Send and Clear buttons
    col1, col2, col3 = st.columns([3, 1, 1])

    with col1:
        send_button = st.button("Send", key="chatbot_send_main", use_container_width=True)

    with col2:
        clear_button = st.button("🗑️ Clear", key="chatbot_clear_main", help="Clear chat history", use_container_width=True)

    with col3:
        close_button = st.button("✖️ Close", key="chatbot_close_main", help="Close chatbot", use_container_width=True)

    # Display chat history section below buttons
    st.markdown("#### Chat History")

    # Chat container with scrollable area
    chat_container = st.container()

    with chat_container:
        # Display conversation history
        for idx, message in enumerate(st.session_state.chatbot_history):
            try:
                role = message.get("role", "user")
                content = message.get("content", "")

                if role == "user":
                    st.markdown(f"**You:** {content}")
                elif role == "assistant":
                    st.markdown(f"**AI:** {content}")

                    # Show thinking section if available (collapsible, like Claude)
                    if "thinking" in message and message["thinking"]:
                        thinking_content = message["thinking"]
                        with st.expander("💭 View AI Thinking Process", expanded=False):
                            st.markdown(f"```\n{thinking_content}\n```")

                # Show data preview if available
                if "data_preview" in message and message["data_preview"] is not None:
                    preview_data = message["data_preview"]

                    if isinstance(preview_data, dict):
                        try:
                            # Convert from {column-name: [values]} format to DataFrame
                            if 'columns' in preview_data:
                                preview_df = pd.DataFrame(preview_data['columns'])
                                row_count = preview_data.get('row_count', len(preview_df))
                            elif 'data' in preview_data:
                                preview_df = pd.DataFrame(preview_data['data'])
                                row_count = preview_data.get('row_count', len(preview_df))
                            else:
                                preview_df = pd.DataFrame(preview_data)
                                row_count = len(preview_df)

                            if not preview_df.empty:
                                # Clean the dataframe
                                for col in preview_df.columns:
                                    try:
                                        col_dtype = preview_df[col].dtype
                                        if col_dtype == 'object':
                                            preview_df[col] = preview_df[col].apply(
                                                lambda x: str(x) if x is not None else None
                                            )
                                        else:
                                            preview_df[col] = preview_df[col].apply(
                                                lambda x: x.item() if hasattr(x, 'item') else (x if x is not None else None)
                                            )
                                    except Exception as col_err:
                                        logger.error(f"Error processing column {col}: {col_err}")
                                        preview_df[col] = preview_df[col].astype(str)

                                display_df = preview_df.head(10).copy()

                                with st.expander(f"📊 Data Preview ({row_count} rows total)"):
                                    safe_key = f"preview_{int(idx)}_{int(row_count)}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
                                    try:
                                        st.dataframe(
                                            display_df,
                                            use_container_width=True,
                                            key=safe_key
                                        )
                                    except Exception as display_err:
                                        logger.error(f"Error displaying dataframe: {display_err}")
                                        st.text(display_df.to_string())
                        except Exception as e:
                            logger.error(f"Could not display data preview at index {idx}: {e}")
                            st.warning(f"Could not display data preview: {str(e)}")

            except Exception as e:
                logger.warning(f"Error displaying message {idx}: {e}")
                continue

    # Close button - disabled since chatbot should always be available in AI Assistant tab
    # User can simply switch tabs instead
    if close_button:
        st.info("💡 Switch to the Dashboard tab to hide the chatbot")

    # Clear chat history
    if clear_button:
        st.session_state.chatbot_history = []
        st.rerun()

    # Process user input
    if send_button and user_input.strip():
        try:
            if not isinstance(st.session_state.chatbot_history, list):
                st.session_state.chatbot_history = []

            # Add user message to history
            user_message = {
                "role": "user",
                "content": str(user_input).strip()
            }

            try:
                user_message = _sanitize_for_session_state(user_message)
            except Exception as sanitize_err:
                logger.error(f"Error sanitizing user message: {sanitize_err}")
                user_message = {
                    "role": "user",
                    "content": str(user_input).strip()
                }

            st.session_state.chatbot_history.append(user_message)

            # Create dedicated thinking display section (outside spinner for real-time updates)
            st.markdown("---")
            st.markdown("### 🧠 AI Thinking Process")
            st.caption("Watch the AI analyze your query and plan its response in real-time")

            from thinking_display import create_thinking_placeholder

            thinking_container = st.container()
            thinking_display, thinking_wrapper = create_thinking_placeholder(
                container=thinking_container,
                title="💭 Real-time Analysis & Planning",
                expanded=True
            )

            st.markdown("---")
            st.markdown("### 💬 AI Response")
            response_placeholder = st.empty()

            # Show processing indicator
            with st.spinner("Thinking:"):
                try:
                    orchestrator = ChatbotOrchestrator(HF_API_TOKEN)
                    orchestrator.enable_streaming(
                        thinking_placeholder=thinking_wrapper,
                        response_placeholder=response_placeholder
                    )

                    response, filtered_df, analysis_info = orchestrator.process_query_enhanced(
                        user_input,
                        master_df,
                        max_qty_df,
                        min_qty_df,
                        max_buy_df,
                        st.session_state.chatbot_history
                    )

                    # Add assistant response to history
                    data_preview_dict = None
                    if filtered_df is not None and not filtered_df.empty:
                        try:
                            data_preview_dict = _convert_dataframe_to_safe_dict(filtered_df)
                        except Exception as convert_err:
                            logger.error(f"Error converting DataFrame to safe dict: {convert_err}")
                            data_preview_dict = None

                    assistant_message = {
                        "role": "assistant",
                        "content": str(response)
                    }

                    if data_preview_dict is not None:
                        assistant_message["data_preview"] = data_preview_dict

                    try:
                        assistant_message = _sanitize_for_session_state(assistant_message)
                        json.dumps(assistant_message)
                    except Exception as sanitize_err:
                        logger.error(f"Error sanitizing assistant message: {sanitize_err}")
                        assistant_message = {
                            "role": "assistant",
                            "content": str(response)
                        }

                    st.session_state.chatbot_history.append(assistant_message)

                    if len(st.session_state.chatbot_history) > MAX_HISTORY_MESSAGES:
                        st.session_state.chatbot_history = st.session_state.chatbot_history[-MAX_HISTORY_MESSAGES:]

                except Exception as e:
                    error_msg = f"Chatbot error: {str(e)}"
                    logger.error(error_msg)

                    error_response = {
                        "role": "assistant",
                        "content": f"I encountered an error: {str(e)}. Please try again or rephrase your question."
                    }

                    error_response = _sanitize_for_session_state(error_response)

                    if isinstance(st.session_state.chatbot_history, list):
                        st.session_state.chatbot_history.append(error_response)
                    else:
                        st.session_state.chatbot_history = [error_response]

        except Exception as outer_error:
            error_str = str(outer_error)
            logger.error(f"Error handling chat message: {error_str}")
            st.error(f"An error occurred processing your message: {error_str}. Please try clearing the chat and trying again.")

        # Rerun to update UI
        st.rerun()

    # Help section
    with st.expander("💡 Example Questions"):
        st.markdown("""
        - Show me stocks with quantity less than 2
        - What are the top 10 items by cost?
        - Analyze trends for Cushion diamonds
        - Which shapes are most profitable?
        - Compare Round vs Princess diamonds
        - What should I focus on buying?
        """)

