"""
Thinking Display Component for Streamlit
Handles LLM thinking display similar to Claude's streaming thinking feature
"""

import streamlit as st
import re
from typing import Optional, Tuple


class ThinkingDisplay:
    """
    Manages the display of LLM thinking process in a collapsible expander.

    Features:
    - Streaming support: Updates thinking in real-time as it arrives
    - Separate display: Shows thinking in an expander, keeps output clean
    - Auto-extraction: Removes thinking tags from final output
    """

    def __init__(self, container=None, title: str = "🧠 AI Thinking Process", expanded: bool = True):
        """
        Initialize thinking display component

        Args:
            container: Streamlit container to render in (default: creates new)
            title: Title for the expander
            expanded: Whether expander starts expanded
        """
        self.container = container if container is not None else st.empty()
        self.title = title
        self.expanded = expanded
        self.thinking_content = ""
        self.has_thinking = False

    def update(self, thinking_text: str):
        """
        Update the thinking display with new content

        Args:
            thinking_text: The thinking content to display
        """
        if not thinking_text or not thinking_text.strip():
            return

        self.thinking_content = thinking_text.strip()
        self.has_thinking = True

        # Render with styled container
        with self.container.container():
            # Use info box for better visibility
            st.info(self.thinking_content, icon="💭")

    def clear(self):
        """Clear the thinking display"""
        self.thinking_content = ""
        self.has_thinking = False
        self.container.empty()

    @staticmethod
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

    @staticmethod
    def stream_with_thinking_extraction(text_stream, thinking_display: 'ThinkingDisplay',
                                       response_placeholder) -> str:
        """
        Process a streaming text response, extracting and displaying thinking separately

        Args:
            text_stream: Iterator yielding text chunks
            thinking_display: ThinkingDisplay instance to show thinking
            response_placeholder: Streamlit placeholder for response

        Returns:
            Complete response text (without thinking tags)
        """
        full_text = ""
        thinking_text = ""
        response_text = ""
        in_thinking = False
        thinking_complete = False

        for chunk in text_stream:
            if not chunk:
                continue

            full_text += chunk

            # Parse thinking tags in real-time
            if '<thinking>' in full_text and not thinking_complete:
                in_thinking = True
                # Extract thinking content so far
                if '</thinking>' in full_text:
                    # Thinking is complete
                    thinking_complete = True
                    in_thinking = False
                    thinking_text, response_text = ThinkingDisplay.extract_thinking_and_response(full_text)

                    # Display complete thinking
                    if thinking_text:
                        thinking_display.update(thinking_text)

                    # Display response so far
                    if response_text:
                        response_placeholder.markdown(response_text)
                else:
                    # Thinking still streaming
                    parts = full_text.split('<thinking>')
                    if len(parts) > 1:
                        thinking_text = parts[1]
                        thinking_display.update(thinking_text)

            elif thinking_complete:
                # Thinking already shown, just update response
                _, response_text = ThinkingDisplay.extract_thinking_and_response(full_text)
                if response_text:
                    response_placeholder.markdown(response_text)

            elif '<thinking>' not in full_text:
                # No thinking tags at all, just stream response
                response_placeholder.markdown(full_text)
                response_text = full_text

        # Return final response without thinking
        _, final_response = ThinkingDisplay.extract_thinking_and_response(full_text)
        return final_response


class StreamlitThinkingWrapper:
    """
    Wrapper for Streamlit placeholders to work with deque_window_processor

    This provides a simple .markdown() interface that the processor expects,
    while internally managing the thinking display
    """

    def __init__(self, thinking_display: ThinkingDisplay):
        """
        Initialize wrapper

        Args:
            thinking_display: ThinkingDisplay instance to use
        """
        self.thinking_display = thinking_display

    def markdown(self, content: str):
        """
        Display content - extracts thinking if present

        Args:
            content: Content to display (may contain thinking tags or markdown formatting)
        """
        # Handle formatted markdown from deque_processor
        # Format: "**🧠 AI Thinking Process:**\n\n```\n{thinking_text}\n```"

        if "```" in content:
            # Parse markdown code block
            parts = content.split("```")
            if len(parts) >= 2:
                # Extract text from code block
                thinking_text = parts[1].strip()
                # Remove language identifier if present (e.g., "```python")
                if '\n' in thinking_text:
                    lines = thinking_text.split('\n')
                    if lines[0].strip() and not lines[0].strip().startswith('*'):
                        thinking_text = '\n'.join(lines[1:])
                self.thinking_display.update(thinking_text)
                return

        # Check for thinking tags
        thinking_text, _ = ThinkingDisplay.extract_thinking_and_response(content)
        if thinking_text:
            self.thinking_display.update(thinking_text)
        else:
            # No thinking tags, just display content
            self.thinking_display.update(content)


def create_thinking_placeholder(container=None, title: str = "🧠 AI Thinking Process",
                                expanded: bool = True) -> Tuple[ThinkingDisplay, StreamlitThinkingWrapper]:
    """
    Create a thinking display and wrapper for use in Streamlit apps

    Args:
        container: Streamlit container to render in
        title: Title for the thinking expander
        expanded: Whether expander starts expanded

    Returns:
        (ThinkingDisplay, StreamlitThinkingWrapper) tuple
    """
    thinking_display = ThinkingDisplay(container, title, expanded)
    wrapper = StreamlitThinkingWrapper(thinking_display)
    return thinking_display, wrapper
