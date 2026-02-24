"""
Enhanced Query Analyzer - Identifies Filter Columns and Filter Values from User Query
Distinguishes between column names and values in the data
"""

import logging
import requests
import json
from typing import Dict, List, Any, Tuple, Optional
import pandas as pd

logger = logging.getLogger(__name__)


class IntentClassifier:
    """
    Classifies user query intent to determine if it's a simple filter or requires analysis.
    Examples:
    - "Show me data for Cushion Diamonds" -> FILTER_ONLY
    - "What is the trend of Cushion Diamonds?" -> ANALYSIS
    """

    def __init__(self, api_token: str, api_url: str, model: str):
        self.api_token = api_token
        self.api_url = api_url
        self.model = model
        self.headers = {"Authorization": f"Bearer {api_token}"}

    def classify_intent(self, user_query: str) -> str:
        """
        Classify the intent behind the user query.

        Returns:
            "filter_only" - Simple filter query, just return filtered data
            "analysis" - Requires analysis, use deque sliding window
        """

        system_prompt = f"""You are an intent classifier for a data query system.

Your job is to determine if the user wants:
1. FILTER_ONLY: Just see/view/display filtered data (no analysis needed)
2. ANALYSIS: Perform analysis, find trends, recommendations, comparisons, calculations, insights

User Query: "{user_query}"

FILTER_ONLY keywords/patterns:
- "show me", "display", "give me", "list", "view", "see"
- "data for", "records of", "items with", "all"
- Just asking to filter without any analysis words

ANALYSIS keywords/patterns:
- "analyze", "trend", "compare", "recommendation", "best", "worst"
- "average", "total", "sum", "count", "calculate"
- "why", "how", "what is the pattern", "insight", "correlation"
- "predict", "forecast", "suggest", "optimize"

Examples:
- "Show me data for Cushion Diamonds" -> FILTER_ONLY
- "Display all Round diamonds" -> FILTER_ONLY
- "Give me records with color D" -> FILTER_ONLY
- "What is the price trend for Cushion Diamonds?" -> ANALYSIS
- "Analyze Round diamonds" -> ANALYSIS
- "Compare D color vs E color" -> ANALYSIS
- "Find best price for Princess cut" -> ANALYSIS

Respond with ONLY valid JSON:
{{
    "intent": "filter_only" or "analysis",
    "confidence": 0.0 to 1.0,
    "reasoning": "brief explanation"
}}"""

        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json={
                    "messages": [
                        {"role": "user", "content": system_prompt}
                    ],
                    "model": self.model,
                    "max_tokens": 150,
                    "temperature": 0.1
                },
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                if 'choices' in result and len(result['choices']) > 0:
                    generated_text = result['choices'][0]['message']['content']

                    # Extract JSON
                    start_idx = generated_text.find('{')
                    end_idx = generated_text.rfind('}') + 1
                    if start_idx != -1 and end_idx > start_idx:
                        json_str = generated_text[start_idx:end_idx]
                        parsed = json.loads(json_str)
                        intent = parsed.get('intent', 'analysis')
                        logger.info(f"🎯 Intent Classification: {intent} (confidence: {parsed.get('confidence', 0)})")
                        logger.info(f"   Reasoning: {parsed.get('reasoning', 'N/A')}")
                        return intent

            # Fallback to simple keyword matching
            return self._fallback_intent_classification(user_query)

        except Exception as e:
            logger.error(f"Error in intent classification: {e}")
            return self._fallback_intent_classification(user_query)

    def _fallback_intent_classification(self, user_query: str) -> str:
        """Simple keyword-based fallback for intent classification"""
        query_lower = user_query.lower()

        # Filter-only keywords
        filter_keywords = ['show me', 'display', 'give me', 'list', 'view', 'see', 'get']

        # Analysis keywords
        analysis_keywords = [
            'analyze', 'trend', 'compare', 'recommendation', 'best', 'worst',
            'average', 'total', 'sum', 'count', 'calculate', 'why', 'how',
            'pattern', 'insight', 'predict', 'forecast', 'suggest', 'optimize',
            'correlation', 'relationship', 'impact'
        ]

        # Check for analysis keywords first (higher priority)
        if any(keyword in query_lower for keyword in analysis_keywords):
            logger.info(f"🎯 Intent (fallback): ANALYSIS (found analysis keywords)")
            return "analysis"

        # Check for filter-only keywords
        if any(keyword in query_lower for keyword in filter_keywords):
            logger.info(f"🎯 Intent (fallback): FILTER_ONLY (found filter keywords)")
            return "filter_only"

        # Default to analysis if uncertain
        logger.info(f"🎯 Intent (fallback): ANALYSIS (default)")
        return "analysis"


class SmartQueryUnderstanding:
    """
    Intelligently identifies whether query terms refer to column names or values.
    Example: "Analyze the trend of Cushion Diamond"
    - LLM identifies: "Cushion" is not a column name, but a VALUE in "Shape key" column
    - Result: Filter Column = "Shape key", Filter Value = "Cushion"
    """

    def __init__(self, api_token: str, api_url: str, model: str):
        self.api_token = api_token
        self.api_url = api_url
        self.model = model
        self.headers = {"Authorization": f"Bearer {api_token}"}

    def identify_column_and_value(self, user_query: str, data_schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze user query to identify:
        1. If query mentions a column name directly
        2. If query mentions a VALUE, find which column contains that value
        3. Extract ALL filter columns and filter values (supports multiple filters)

        Args:
            user_query: User's natural language query
            data_schema: Schema information with columns and their sample values

        Returns:
            {
                "filter_column": "column_name" or None (primary filter for backward compatibility),
                "filter_value": "value" or None (primary value for backward compatibility),
                "filters": [{"column": "...", "operator": "==", "value": "..."}] (all filters),
                "operation_type": "filter_and_analyze" | "analyze_all" | "unique_values",
                "analysis_focus": "trend" | "comparison" | "summary" | "count"
            }
        """

        # Build context about columns and their values
        schema_context = self._build_schema_context(data_schema)

        system_prompt = f"""You are a smart query analyzer for a diamond inventory database.
Your job is to understand if the user is referring to a COLUMN NAME or a VALUE in a column.

Database Schema:
{schema_context}

IMPORTANT RULES FOR IDENTIFYING FILTER VALUES (SUPPORTS MULTIPLE FILTERS):

1. COLUMN PRIORITY (CRITICAL):
   - ALWAYS prefer columns ending with "key" or "Key" (e.g., "Shape key", "Color Key") over display columns (e.g., "Shape", "Color")
   - If both "Shape" and "Shape key" contain matching values, ALWAYS use "Shape key"
   - If both "Color" and "Color Key" contain matching values, ALWAYS use "Color Key"
   - This ensures filters are applied to normalized data columns, not display columns

2. DETECT ALL FILTERS in the query (HIGHEST PRIORITY):
   - CAREFULLY check the sample_values for each column for ALL mentioned values
   - Look for EXACT or PARTIAL matches (e.g., "Cushion" matches "Cushion" in sample_values)
   - Look for case-insensitive matches (e.g., "cushion" matches "Cushion")
   - Common diamond values to watch for:
     * Shape key column: "Cushion", "Round", "Princess", "Oval", "Emerald", "Marquise", "Pear", "Heart", "Asscher", "Radiant"
     * Color Key column: "FY", "FVY", "FIY", "FLY", "WXYZ" (Note: user may say "FY Color" meaning Color Key = "FY")
     * Clarity column: "IF", "VVS1", "VVS2", "VS1", "VS2", "SI1", "SI2", "FL", "I1", "I2", "I3"
   - DATE/TIME FILTERS - CRITICAL:
     * Month column: "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"
     * Year column: 2024, 2025 (numeric values)
     * If user says "June 2025", create TWO filters: {{"column": "Month", "value": "June"}}, {{"column": "Year", "value": 2025}}
     * If user says "January 2024", create TWO filters: {{"column": "Month", "value": "January"}}, {{"column": "Year", "value": 2024}}
   - When you find multiple value matches in DIFFERENT columns:
     * Create a filter for EACH (column, value) pair
     * Set filter_column = first column (for backward compatibility)
     * Set filter_value = first value (for backward compatibility)
     * Set filters = [{{"column": "col1", "operator": "==", "value": "val1"}}, {{"column": "col2", "operator": "==", "value": "val2"}}, ...]
     * Set operation_type = "filter_and_analyze"

3. If the query mentions a COLUMN NAME directly (e.g., "Shape key", "Color Key", "Weight"):
   - Set filter_column = the column name
   - Set filter_value = null
   - Set filters = []
   - Set operation_type = "unique_values" (if asking about unique values) or "analyze_all"

4. If no specific column or value is mentioned:
   - Set filter_column = null
   - Set filter_value = null
   - Set filters = []
   - Set operation_type = "analyze_all"

EXAMPLES TO HELP YOU:
- Query: "Analyze trend of Cushion Diamond" → filter_column="Shape key", filter_value="Cushion", filters=[{{"column": "Shape key", "operator": "==", "value": "Cushion"}}]
- Query: "Show me Round diamonds" → filter_column="Shape key", filter_value="Round", filters=[{{"column": "Shape key", "operator": "==", "value": "Round"}}]
- Query: "Show me Cushion diamonds with color D" → filter_column="Shape key", filter_value="Cushion", filters=[{{"column": "Shape key", "operator": "==", "value": "Cushion"}}, {{"column": "Color Key", "operator": "==", "value": "D"}}]
- Query: "Display Princess cut with VS1 clarity" → filters=[{{"column": "Shape key", "operator": "==", "value": "Princess"}}, {{"column": "Clarity", "operator": "==", "value": "VS1"}}]
- Query: "Compare D color vs E color" → filter_column="Color Key", filter_value="D", filters=[{{"column": "Color Key", "operator": "==", "value": "D"}}]
- Query: "Show me data for June 2025" → filters=[{{"column": "Month", "operator": "==", "value": "June"}}, {{"column": "Year", "operator": "==", "value": 2025}}]
- Query: "Show me Data for June 2025 for Cushion Diamond of FY Color" → filters=[{{"column": "Month", "operator": "==", "value": "June"}}, {{"column": "Year", "operator": "==", "value": 2025}}, {{"column": "Shape key", "operator": "==", "value": "Cushion"}}, {{"column": "Color Key", "operator": "==", "value": "FY"}}]

User Query: "{user_query}"

STEP-BY-STEP ANALYSIS:
1. Extract key terms from the query: {user_query}
2. Check each term against sample_values in ALL columns
3. Identify if any term is a VALUE (found in sample_values)
4. If found, extract the column name and exact value
5. For date/time queries, check for month names and years - create separate filters for each

Respond with ONLY valid JSON in this format:
{{
    "filter_column": "column_name or null (first filter for backward compatibility)",
    "filter_value": "exact_value_from_sample or null (first value for backward compatibility)",
    "filters": [
        {{"column": "column_name", "operator": "==", "value": "exact_value"}},
        {{"column": "another_column", "operator": "==", "value": "another_value"}}
    ],
    "operation_type": "filter_and_analyze|analyze_all|unique_values",
    "analysis_focus": "trend|comparison|summary|count|list",
    "reasoning": "Step-by-step: searched for [terms] in sample_values, found [value] in [column], [value2] in [column2], therefore..."
}}"""

        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json={
                    "messages": [
                        {"role": "user", "content": system_prompt}
                    ],
                    "model": self.model,
                    "max_tokens": 300,
                    "temperature": 0.1
                },
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                if 'choices' in result and len(result['choices']) > 0:
                    generated_text = result['choices'][0]['message']['content']

                    # Extract JSON
                    start_idx = generated_text.find('{')
                    end_idx = generated_text.rfind('}') + 1
                    if start_idx != -1 and end_idx > start_idx:
                        json_str = generated_text[start_idx:end_idx]
                        parsed = json.loads(json_str)
                        logger.info(f"✅ Query Understanding: {parsed}")
                        return parsed

            # Fallback
            return self._fallback_understanding(user_query, data_schema)

        except Exception as e:
            logger.error(f"Error in query understanding: {e}")
            return self._fallback_understanding(user_query, data_schema)

    def _build_schema_context(self, data_schema: Dict[str, Any]) -> str:
        """Build readable schema context for LLM"""
        context = "Columns:\n"

        columns_info = data_schema.get('columns_info', {})
        for col_name, col_info in columns_info.items():
            context += f"\n- {col_name} ({col_info.get('dtype', 'unknown')})\n"

            sample_values = col_info.get('sample_values', [])
            if sample_values:
                context += f"  Sample values: {sample_values}\n"

        return context

    def _fallback_understanding(self, user_query: str, data_schema: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback logic using simple keyword matching - supports multiple filters"""
        user_query_lower = user_query.lower()
        columns_info = data_schema.get('columns_info', {})

        # Check if query mentions a column name
        for col_name in columns_info.keys():
            if col_name.lower() in user_query_lower:
                return {
                    "filter_column": None,
                    "filter_value": None,
                    "filters": [],
                    "operation_type": "analyze_all",
                    "analysis_focus": "summary",
                    "reasoning": f"Column '{col_name}' mentioned in query"
                }

        # Check if query mentions multiple values across different columns
        # IMPORTANT: Prioritize normalized columns (ending with "key" or "Key") over display columns
        detected_filters = []
        candidate_filters = []  # Store all potential matches

        for col_name, col_info in columns_info.items():
            sample_values = col_info.get('sample_values', [])
            for value in sample_values:
                # Check if this value is mentioned in the query
                if str(value).lower().strip() in user_query_lower:
                    candidate_filters.append({
                        "column": col_name,
                        "operator": "==",
                        "value": value,
                        "is_key_column": col_name.endswith(" key") or col_name.endswith(" Key") or col_name.endswith("_key") or col_name.endswith("_Key") or col_name == "Month" or col_name == "Year"
                    })
                    break  # Only take one value per column in fallback

        # Prioritize "key" columns over display columns
        # For each base column (e.g., "Shape" vs "Shape key"), prefer the "key" version
        seen_bases = set()
        for candidate in sorted(candidate_filters, key=lambda x: (not x["is_key_column"], x["column"])):
            # Extract base name (e.g., "Shape" from "Shape key" or "Shape")
            base_name = candidate["column"].replace(" key", "").replace(" Key", "").replace("_key", "").replace("_Key", "")

            # If we've already added a filter for this base, skip non-key versions
            if base_name in seen_bases and not candidate["is_key_column"]:
                continue

            detected_filters.append({
                "column": candidate["column"],
                "operator": candidate["operator"],
                "value": candidate["value"]
            })
            seen_bases.add(base_name)

        # If we found filters, return them
        if detected_filters:
            return {
                "filter_column": detected_filters[0]["column"],
                "filter_value": detected_filters[0]["value"],
                "filters": detected_filters,
                "operation_type": "filter_and_analyze",
                "analysis_focus": "summary",
                "reasoning": f"Found {len(detected_filters)} filter(s): {detected_filters} (prioritized key columns)"
            }

        # No specific filter
        return {
            "filter_column": None,
            "filter_value": None,
            "filters": [],
            "operation_type": "analyze_all",
            "analysis_focus": "summary",
            "reasoning": "No specific column or value identified"
        }


class EnhancedQueryAnalyzer:
    """
    Enhanced analyzer that properly distinguishes columns from values
    and classifies query intent
    """

    def __init__(self, api_token: str, api_url: str, model: str):
        self.intent_classifier = IntentClassifier(api_token, api_url, model)
        self.smart_understanding = SmartQueryUnderstanding(api_token, api_url, model)

    def analyze_query(self, user_query: str, data_schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point for query analysis with intent classification

        Returns:
            {
                "intent": "filter_only" or "analysis",
                "filter_column": column name or None,
                "filter_value": value or None,
                "operation_type": type of operation,
                "analysis_focus": what type of analysis to perform,
                "filters": list of filter dicts for backward compatibility
            }
        """

        # Step 1: Classify intent first
        logger.info("🎯 Step 1: Classifying query intent...")
        intent = self.intent_classifier.classify_intent(user_query)

        # Step 2: Use smart understanding to identify filter column and value
        logger.info("🔍 Step 2: Identifying filter column and value...")
        understanding = self.smart_understanding.identify_column_and_value(
            user_query, data_schema
        )

        # Get filters from understanding - use 'filters' field if available, otherwise build from single filter
        filters = understanding.get('filters', [])

        # Backward compatibility: if no filters array but we have filter_column and filter_value, create one
        if not filters and understanding.get('filter_column') and understanding.get('filter_value'):
            filters.append({
                "column": understanding['filter_column'],
                "operator": "==",
                "value": understanding['filter_value']
            })

        result = {
            "intent": intent,  # NEW: Intent classification
            "filter_column": understanding.get('filter_column'),
            "filter_value": understanding.get('filter_value'),
            "operation_type": understanding.get('operation_type', 'analyze_all'),
            "analysis_focus": understanding.get('analysis_focus', 'summary'),
            "filters": filters,  # Now supports multiple filters!
            "reasoning": understanding.get('reasoning', '')
        }

        logger.info(f"📊 Query Analysis Result:")
        logger.info(f"   Intent: {result['intent']}")
        logger.info(f"   Filter Column: {result['filter_column']}")
        logger.info(f"   Filter Value: {result['filter_value']}")
        logger.info(f"   Operation: {result['operation_type']}")
        logger.info(f"   Reasoning: {result['reasoning']}")

        return result
