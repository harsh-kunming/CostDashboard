"""
Deque Sliding Window Processor - Map-Reduce Pattern
Processes large datasets in windows, accumulating results in a deque
"""

import logging
import pandas as pd
import requests
import json
import time
import re
from collections import deque
from typing import Dict, List, Any, Tuple, Optional

logger = logging.getLogger(__name__)


def extract_thinking_and_response(text: str) -> Tuple[Optional[str], str]:
    """
    Extract thinking and response from text containing <thinking> tags

    Args:
        text: Raw text that may contain <thinking></thinking> tags

    Returns:
        (thinking_content, response_content) tuple
        - thinking_content: Text inside thinking tags (None if no tags)
        - response_content: Text outside thinking tags (full text if no tags)
    """
    if not text:
        return None, ""

    # Check for thinking tags
    thinking_pattern = r'<thinking>(.*?)</thinking>'
    match = re.search(thinking_pattern, text, re.DOTALL)

    if match:
        thinking_content = match.group(1).strip()
        # Remove thinking tags and content from the text
        response_content = re.sub(thinking_pattern, '', text, flags=re.DOTALL).strip()
        return thinking_content, response_content
    else:
        # No thinking tags found
        return None, text


class DequeWindowProcessor:
    """
    Implements sliding window processing with deque accumulation (Map-Reduce pattern)

    WORKFLOW:
    1. Filter dataframe based on filter_column and filter_value
    2. Split filtered data into windows of fixed size
    3. MAP PHASE: Process each window through LLM → intermediate result → append to deque
    4. REDUCE PHASE: Combine all deque results through LLM → final comprehensive answer
    """

    def __init__(self, api_token: str, api_url: str, model: str, window_size: int = 100,
                 max_retries: int = 3, retry_delay: float = 2.0):
        self.api_token = api_token
        self.api_url = api_url
        self.model = model
        self.window_size = window_size
        self.headers = {"Authorization": f"Bearer {api_token}"}
        self.max_retries = max_retries
        self.retry_delay = retry_delay  # Initial delay in seconds for exponential backoff

        # Store last thinking content for persistence in chat history
        self.last_thinking_content = None

    def process_with_deque(self, user_query: str, df: pd.DataFrame,
                          filter_column: str = None, filter_value: Any = None,
                          relevant_columns: List[str] = None,
                          stream: bool = False, thinking_placeholder=None, response_placeholder=None) -> Tuple[str, pd.DataFrame, deque]:
        """
        Main processing function using deque sliding window

        Args:
            user_query: User's question
            df: Full dataframe
            filter_column: Column to filter on (None if no filter)
            filter_value: Value to filter for (None if no filter)
            relevant_columns: List of columns to focus on (None = all columns)
            stream: Enable streaming with thinking display
            thinking_placeholder: Streamlit placeholder for thinking display
            response_placeholder: Streamlit placeholder for response display

        Returns:
            (final_answer, filtered_df, results_deque)
        """

        logger.info("🚀 Starting Deque Sliding Window Processing")
        logger.info(f"   Original data: {len(df)} rows, {len(df.columns)} columns")

        # Step 1: Apply filter if needed
        filtered_df = self._apply_filter(df, filter_column, filter_value)
        logger.info(f"   After filtering: {len(filtered_df)} rows")

        if filtered_df.empty:
            return "No data found matching your criteria.", filtered_df, deque()

        # Step 2: Select relevant columns
        data_to_process = self._select_columns(filtered_df, relevant_columns)
        logger.info(f"   Columns to process: {list(data_to_process.columns)}")

        # Step 3: Split into windows
        windows = self._create_windows(data_to_process)
        num_windows = len(windows)
        logger.info(f"   Created {num_windows} windows (size: {self.window_size})")

        # Step 4: MAP PHASE - Process each window
        results_deque = deque()
        logger.info("🗺️  MAP PHASE: Processing windows...")

        # IMPORTANT: During MAP phase, do NOT stream thinking to UI
        # Users should ONLY see the final REDUCE phase output
        for idx, window_df in enumerate(windows, 1):
            logger.info(f"   Processing window {idx}/{num_windows}...")

            # Process window WITHOUT showing intermediate results to user
            window_result = self._process_single_window(
                user_query=user_query,
                window_df=window_df,
                window_number=idx,
                total_windows=num_windows,
                stream=False,  # CRITICAL: Set to False to prevent intermediate outputs
                thinking_placeholder=None  # CRITICAL: No placeholder during MAP phase
            )

            results_deque.append(window_result)
            logger.info(f"   ✅ Window {idx}/{num_windows} completed")

        logger.info(f"✅ MAP PHASE complete. Deque has {len(results_deque)} results")

        # Step 5: REDUCE PHASE - Generate final answer
        logger.info("🔽 REDUCE PHASE: Synthesizing final answer...")

        # REDUCE phase: This is where we stream thinking and response to UI
        # Pass streaming parameters to reduce phase
        final_answer = self._reduce_deque_results(
            user_query=user_query,
            results_deque=results_deque,
            total_rows=len(filtered_df),
            stream=stream,
            thinking_placeholder=thinking_placeholder,
            response_placeholder=response_placeholder
        )

        logger.info("✅ REDUCE PHASE complete")

        return final_answer, filtered_df, results_deque

    def _apply_filter(self, df: pd.DataFrame, filter_column: str, filter_value: Any) -> pd.DataFrame:
        """Apply filter to dataframe"""
        if filter_column is None or filter_value is None:
            return df.copy()

        if filter_column not in df.columns:
            logger.warning(f"Filter column '{filter_column}' not found in dataframe")
            return df.copy()

        try:
            # Case-insensitive matching for string values
            if isinstance(filter_value, str):
                mask = df[filter_column].astype(str).str.lower() == filter_value.lower()
            else:
                mask = df[filter_column] == filter_value

            filtered = df[mask].copy()
            logger.info(f"   Filter applied: {filter_column} == {filter_value}")
            logger.info(f"   Result: {len(filtered)} rows")

            return filtered

        except Exception as e:
            logger.error(f"Error applying filter: {e}")
            return df.copy()

    def _select_columns(self, df: pd.DataFrame, relevant_columns: List[str] = None) -> pd.DataFrame:
        """Select relevant columns from dataframe"""
        if relevant_columns is None:
            return df.copy()

        available_cols = [col for col in relevant_columns if col in df.columns]

        if not available_cols:
            return df.copy()

        return df[available_cols].copy()

    def _create_windows(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        """Split dataframe into windows"""
        windows = []
        total_rows = len(df)

        for start_idx in range(0, total_rows, self.window_size):
            end_idx = min(start_idx + self.window_size, total_rows)
            window_df = df.iloc[start_idx:end_idx].copy()
            windows.append(window_df)

        return windows

    def _process_single_window(self, user_query: str, window_df: pd.DataFrame,
                               window_number: int, total_windows: int,
                               stream: bool = False, thinking_placeholder=None) -> str:
        """
        Process a single window through LLM (MAP operation) with retry logic and thinking support

        Args:
            user_query: User's question
            window_df: Data window to process
            window_number: Current window number
            total_windows: Total number of windows
            stream: Enable streaming with thinking display
            thinking_placeholder: Streamlit placeholder for thinking (if streaming)

        Returns:
            Meaningful intermediate analysis result for this window
        """

        # Convert window to string representation
        window_text = self._dataframe_to_text(window_df, window_number, total_windows)

        system_prompt = f"""You are a data analyst extracting meaningful insights from a data window.

User's Question: "{user_query}"

Window Data (Window {window_number}/{total_windows}):
{window_text}

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
        base_tokens = 300
        complexity_bonus = min(200, len(window_df) // 10)
        query_complexity_bonus = min(100, len(user_query.split()) * 5)
        max_tokens = base_tokens + complexity_bonus + query_complexity_bonus

        logger.info(f"   Window {window_number}: Using {max_tokens} max_tokens for analysis")

        # Retry logic with exponential backoff
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self._make_api_request_with_retry(
                    system_prompt=system_prompt,
                    max_tokens=max_tokens,
                    attempt=attempt,
                    window_number=window_number
                )

                if response.status_code == 200:
                    result = response.json()

                    # Log the response structure for debugging
                    logger.info(f"   Window {window_number}: Response keys: {result.keys()}")

                    if 'choices' in result and len(result['choices']) > 0:
                        choice = result['choices'][0]
                        logger.info(f"   Window {window_number}: Choice keys: {choice.keys()}")

                        # Handle different response formats
                        raw_window_result = None

                        if 'message' in choice and isinstance(choice['message'], dict):
                            # Try multiple possible content fields
                            if 'content' in choice['message']:
                                raw_window_result = choice['message']['content'].strip()
                            elif 'reasoning' in choice['message']:
                                # Some APIs return 'reasoning' field with the actual response
                                raw_window_result = choice['message']['reasoning'].strip()
                                logger.info(f"Window {window_number}: Using 'reasoning' field as content")
                            elif 'text' in choice['message']:
                                raw_window_result = choice['message']['text'].strip()
                            else:
                                logger.error(f"Window {window_number}: No content field found in message. Message keys: {choice['message'].keys()}")
                                logger.error(f"Window {window_number}: Full message: {choice['message']}")
                        elif 'text' in choice:
                            # Some APIs return 'text' instead of 'message.content'
                            raw_window_result = choice['text'].strip()
                        elif 'reasoning' in choice:
                            # Some APIs return 'reasoning' at choice level
                            raw_window_result = choice['reasoning'].strip()
                            logger.info(f"Window {window_number}: Using 'reasoning' field at choice level")
                        else:
                            logger.error(f"Window {window_number}: Unexpected choice structure: {choice}")
                            last_error = f"Window {window_number}: Invalid API response format"
                            if attempt < self.max_retries - 1:
                                continue
                            return last_error

                        if not raw_window_result:
                            logger.error(f"Window {window_number}: Could not extract content from response")
                            last_error = f"Window {window_number}: Empty response content"
                            if attempt < self.max_retries - 1:
                                continue
                            return last_error

                        # Extract thinking and response separately
                        thinking_content, window_result = extract_thinking_and_response(raw_window_result)

                        # Display thinking if streaming and present
                        if stream and thinking_placeholder and thinking_content:
                            # Display thinking in the placeholder
                            if hasattr(thinking_placeholder, 'markdown'):
                                thinking_placeholder.markdown(
                                    f"**🧠 Window {window_number} Thinking:**\n\n```\n{thinking_content}\n```"
                                )
                            logger.info(f"   Window {window_number}: Displayed thinking ({len(thinking_content)} chars)")

                        # Log extraction results
                        if thinking_content:
                            logger.info(f"   Window {window_number}: Extracted thinking ({len(thinking_content)} chars) and response ({len(window_result)} chars)")

                        # Filter out meaningless responses
                        if window_result and len(window_result) > 20:
                            logger.info(f"   ✅ Window {window_number} processed successfully - clean output without thinking tags")
                            return window_result
                        else:
                            logger.warning(f"Window {window_number}: Response too short after removing thinking")
                            return f"Window {window_number}: No meaningful insights found for the query in this data segment."
                    else:
                        logger.warning(f"Window {window_number}: Unexpected response format")

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

                else:  # Other error codes
                    logger.error(f"Window {window_number}: API error {response.status_code} - {response.text[:200]}")
                    return f"Window {window_number}: Could not analyze - API returned status {response.status_code}"

            except requests.exceptions.Timeout:
                logger.error(f"Window {window_number}: Request timeout (attempt {attempt + 1})")
                last_error = f"Window {window_number}: Request timeout"
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))
                    continue

            except Exception as e:
                logger.error(f"Error processing window {window_number} (attempt {attempt + 1}): {e}")
                logger.exception("Full traceback:")
                last_error = f"Window {window_number}: Analysis error - {str(e)}"
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))
                    continue

        # All retries exhausted
        return last_error or f"Window {window_number}: Failed after {self.max_retries} attempts"

    def _make_api_request_with_retry(self, system_prompt: str, max_tokens: int,
                                     attempt: int, window_number: int):
        """Make API request with proper configuration"""
        return requests.post(
            self.api_url,
            headers=self.headers,
            json={
                "messages": [
                    {"role": "system", "content": "You are a data analyst extracting meaningful insights. Show your thinking process in <thinking> tags."},
                    {"role": "user", "content": system_prompt}
                ],
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": 0.1,
                "stream": False  # Window processing doesn't use streaming (only final reduce does)
            },
            timeout=60
        )

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


    def _reduce_deque_results(self, user_query: str, results_deque: deque, total_rows: int,
                              stream: bool = False, thinking_placeholder=None, response_placeholder=None) -> str:
        """
        Combine all window results from deque into final comprehensive answer (REDUCE operation)

        This method synthesizes all meaningful insights from the deque to produce a complete,
        well-structured answer to the user's query.

        Args:
            user_query: Original user query
            results_deque: Deque containing all meaningful window results
            total_rows: Total number of rows processed
            stream: Enable streaming with thinking display (for REDUCE phase only)
            thinking_placeholder: Streamlit placeholder for thinking display
            response_placeholder: Streamlit placeholder for response display

        Returns:
            Final comprehensive answer based on complete deque analysis
        """

        # Combine all window results - only meaningful ones
        accumulated_insights = "\n\n".join([
            f"=== Segment {idx + 1} Analysis ===\n{result}"
            for idx, result in enumerate(results_deque)
        ])

        logger.info(f"🔍 DEBUG: Accumulated insights length: {len(accumulated_insights)} characters")
        logger.info(f"🔍 DEBUG: Number of segments in deque: {len(results_deque)}")
        logger.info(f"🔍 DEBUG: Accumulated insights preview (first 1000 chars):\n{accumulated_insights[:1000]}")

        system_prompt = f"""You are an expert data analyst providing a comprehensive final answer based on complete dataset analysis.

CONTEXT:
- User's Question: "{user_query}"
- Total Data Analyzed: {total_rows} rows across {len(results_deque)} data segments
- All segments have been processed and insights have been accumulated

ACCUMULATED INSIGHTS FROM ALL DATA SEGMENTS:
{accumulated_insights}

YOUR TASK - Provide a COMPREHENSIVE FINAL ANSWER:

1. ANSWER THE QUESTION DIRECTLY:
   - Start with a clear, direct answer to the user's query
   - Use specific numbers and facts from the accumulated insights

2. SYNTHESIZE CROSS-SEGMENT INSIGHTS:
   - Identify patterns and trends that emerge across ALL segments
   - Compare and contrast findings from different segments
   - Calculate aggregate statistics (totals, averages, distributions)

3. PROVIDE COMPLETE ANALYSIS:
   - Key findings with supporting data
   - Notable trends or patterns
   - Significant outliers or anomalies
   - Statistical summaries

4. ACTIONABLE RECOMMENDATIONS (if applicable):
   - Based on the data, suggest specific actions
   - Prioritize recommendations by impact

5. STRUCTURE YOUR RESPONSE:
   - Use clear sections and bullet points
   - Lead with the most important information
   - Be thorough but concise

IMPORTANT: This is the FINAL answer the user will see. It must be:
- Complete and self-contained
- Based on ALL the accumulated insights
- Clear, professional, and actionable
- NOT just a summary of windows, but a synthesized answer to their question"""

        try:
            # Set max_tokens to 2500 for comprehensive final answer
            max_tokens = 2500

            logger.info(f"   REDUCE phase: Using {max_tokens} max_tokens for final synthesis")
            logger.info(f"🔍 DEBUG: System prompt length: {len(system_prompt)} characters")
            logger.info(f"🔍 DEBUG: Estimated tokens (rough): ~{len(system_prompt) // 4} tokens")

            # Enable streaming for REDUCE phase if requested
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json={
                    "messages": [
                        {"role": "system", "content": "You are an expert data analyst who synthesizes insights from large datasets to provide comprehensive, actionable answers."},
                        {"role": "user", "content": system_prompt}
                    ],
                    "model": self.model,
                    "max_tokens": max_tokens,
                    "temperature": 0.4,  # Balanced - creative enough to synthesize, factual enough to be accurate
                    "stream": stream  # Enable streaming in REDUCE phase
                },
                timeout=90,  # Longer timeout for comprehensive analysis
                stream=stream  # Enable streaming in requests
            )

            if response.status_code == 200:
                # Handle streaming response
                if stream and thinking_placeholder and response_placeholder:
                    logger.info("✅ Streaming final response with thinking to UI")
                    full_text = ""
                    thinking_text = ""
                    response_text = ""
                    in_thinking = False
                    thinking_complete = False

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
                                        choice = data['choices'][0]

                                        # Handle different streaming formats
                                        content = ''
                                        if 'delta' in choice and isinstance(choice['delta'], dict):
                                            # Check multiple possible content fields in delta
                                            content = choice['delta'].get('content', '')
                                            if not content:
                                                content = choice['delta'].get('reasoning', '')
                                            if not content:
                                                content = choice['delta'].get('text', '')
                                        elif 'text' in choice:
                                            content = choice.get('text', '')
                                        elif 'reasoning' in choice:
                                            content = choice.get('reasoning', '')
                                        elif 'message' in choice and isinstance(choice['message'], dict):
                                            # Check multiple possible content fields in message
                                            content = choice['message'].get('content', '')
                                            if not content:
                                                content = choice['message'].get('reasoning', '')
                                            if not content:
                                                content = choice['message'].get('text', '')

                                        if content:
                                            full_text += content

                                            # Parse thinking tags in real-time
                                            if '<thinking>' in full_text and not thinking_complete:
                                                in_thinking = True
                                                # Extract thinking content so far
                                                if '</thinking>' in full_text:
                                                    # Thinking is complete
                                                    thinking_complete = True
                                                    in_thinking = False
                                                    thinking_text, response_text = extract_thinking_and_response(full_text)

                                                    # Display complete thinking
                                                    if thinking_text and hasattr(thinking_placeholder, 'markdown'):
                                                        thinking_placeholder.markdown(
                                                            f"**🧠 Final Synthesis Thinking:**\n\n```\n{thinking_text}\n```"
                                                        )
                                                        logger.info(f"   Displayed thinking ({len(thinking_text)} chars)")

                                                    # Display response so far
                                                    if response_text and hasattr(response_placeholder, 'markdown'):
                                                        response_placeholder.markdown(response_text)
                                                else:
                                                    # Thinking still streaming
                                                    parts = full_text.split('<thinking>')
                                                    if len(parts) > 1:
                                                        thinking_text = parts[1]
                                                        if hasattr(thinking_placeholder, 'markdown'):
                                                            thinking_placeholder.markdown(
                                                                f"**🧠 Final Synthesis Thinking:**\n\n```\n{thinking_text}\n```"
                                                            )

                                            elif thinking_complete:
                                                # Thinking already shown, just update response
                                                _, response_text = extract_thinking_and_response(full_text)
                                                if response_text and hasattr(response_placeholder, 'markdown'):
                                                    response_placeholder.markdown(response_text)

                                            elif '<thinking>' not in full_text:
                                                # No thinking tags at all, just stream response
                                                if hasattr(response_placeholder, 'markdown'):
                                                    response_placeholder.markdown(full_text)
                                                response_text = full_text

                                except json.JSONDecodeError:
                                    continue

                    # Extract final response without thinking tags
                    thinking_final, response_final = extract_thinking_and_response(full_text)

                    # Store thinking content for persistence in chat history
                    self.last_thinking_content = thinking_final

                    # Use clean response without thinking
                    final_answer = response_final

                    if final_answer and len(final_answer) > 50:
                        logger.info(f"✅ Streamed comprehensive final answer: {len(final_answer)} characters (clean, no thinking tags)")
                        if thinking_final:
                            logger.info(f"   Thinking was extracted, displayed, and stored ({len(thinking_final)} chars)")
                        return final_answer
                    else:
                        logger.warning(f"⚠️ Streamed answer too short ({len(final_answer) if final_answer else 0} chars)")
                        logger.warning(f"⚠️ Raw response had thinking: {thinking_final is not None}")
                        logger.warning(f"⚠️ Full streamed text: {full_text[:1000]}")

                else:
                    # Non-streaming response
                    result = response.json()
                    logger.info(f"Reduce API response keys: {result.keys()}")

                    if 'choices' in result and len(result['choices']) > 0:
                        choice = result['choices'][0]
                        logger.info(f"Reduce phase choice keys: {choice.keys()}")

                        # Handle different response formats
                        raw_answer = None

                        if 'message' in choice and isinstance(choice['message'], dict):
                            # Try multiple possible content fields
                            if 'content' in choice['message']:
                                raw_answer = choice['message']['content'].strip()
                            elif 'reasoning' in choice['message']:
                                # Some APIs return 'reasoning' field with the actual response
                                raw_answer = choice['message']['reasoning'].strip()
                                logger.info(f"Reduce phase: Using 'reasoning' field as content")
                            elif 'text' in choice['message']:
                                raw_answer = choice['message']['text'].strip()
                            else:
                                logger.error(f"Reduce phase: No content field found in message. Message keys: {choice['message'].keys()}")
                                logger.error(f"Reduce phase: Full message: {choice['message']}")
                                return f"""I encountered an error while synthesizing the analysis - API response format is invalid. Please check your API configuration."""
                        elif 'text' in choice:
                            # Some APIs return 'text' instead of 'message.content'
                            raw_answer = choice['text'].strip()
                        elif 'reasoning' in choice:
                            # Some APIs return 'reasoning' at choice level
                            raw_answer = choice['reasoning'].strip()
                            logger.info(f"Reduce phase: Using 'reasoning' field at choice level")
                        else:
                            logger.error(f"Reduce phase: Unexpected choice structure: {choice}")
                            return f"""I encountered an error while synthesizing the analysis - unexpected API response format."""

                        if not raw_answer:
                            logger.error("Reduce phase: Could not extract content from response")
                            return f"""I encountered an error while synthesizing the analysis - empty API response."""

                        logger.info(f"🔍 DEBUG: Raw API response length: {len(raw_answer)} characters")
                        logger.info(f"🔍 DEBUG: Raw response preview (first 500 chars): {raw_answer[:500]}")

                        # Extract thinking and response separately
                        thinking_content, final_answer = extract_thinking_and_response(raw_answer)

                        # Store thinking content for persistence in chat history
                        self.last_thinking_content = thinking_content

                        # Log extraction results
                        if thinking_content:
                            logger.info(f"🔍 DEBUG: Extracted and stored thinking ({len(thinking_content)} chars) and response ({len(final_answer)} chars)")
                            logger.info(f"🔍 DEBUG: Thinking preview: {thinking_content[:200]}...")
                            logger.info(f"🔍 DEBUG: Response preview: {final_answer[:200]}...")
                        else:
                            logger.info(f"🔍 DEBUG: No thinking tags found in response")

                        # Validate that we got a substantial answer
                        if final_answer and len(final_answer) > 50:
                            logger.info(f"✅ Generated comprehensive final answer: {len(final_answer)} characters (clean, no thinking tags)")
                            return final_answer
                        else:
                            logger.warning(f"⚠️ Final answer too short ({len(final_answer) if final_answer else 0} chars)")
                            if thinking_content:
                                logger.warning(f"⚠️ Thinking was extracted ({len(thinking_content)} chars) but response is too short")
                            logger.warning(f"⚠️ Raw answer was: {raw_answer[:1000]}")
                    else:
                        logger.warning(f"Unexpected API response format in reduce phase: {result}")

                # Fallback: generate a basic summary without showing segment details
                logger.warning("Final answer too short, generating basic summary")
                return f"""I analyzed {total_rows} rows of data across {len(results_deque)} segments. However, I wasn't able to generate a comprehensive synthesis. Please try asking your question in a different way or be more specific about what insights you're looking for."""

            else:
                logger.error(f"Reduce API error: {response.status_code} - {response.text}")
                return f"""I encountered an API error (status {response.status_code}) while synthesizing the analysis of {total_rows} rows. Please try again in a moment."""

        except Exception as e:
            logger.error(f"Error in reduce phase: {e}")
            logger.exception("Full traceback:")
            return f"""I encountered an error while synthesizing the final analysis: {str(e)}. Please try asking your question again or rephrase it."""

    def _dataframe_to_text(self, df: pd.DataFrame, window_number: int, total_windows: int) -> str:
        """Convert dataframe window to readable text"""
        text = f"Window {window_number}/{total_windows} - Rows: {len(df)}\n\n"

        # Show first few rows
        text += df.head(20).to_string(index=False)

        # Add statistics
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            text += "\n\n=== Statistics ===\n"
            text += df[numeric_cols].describe().to_string()

        return text
