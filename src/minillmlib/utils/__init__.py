"""Utils module for MiniLLMLib."""

from .json_utils import to_dict, extract_json_from_completion
from .message_utils import (
    format_prompt,
    merge_contiguous_messages,
    NodeCompletionParameters
)

__all__ = [
    'to_dict',
    'extract_json_from_completion',
    'format_prompt',
    'merge_contiguous_messages',
    'NodeCompletionParameters'
]
