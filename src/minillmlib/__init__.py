"""MiniLLMLib - A library for interacting with various LLM providers."""

from .core.chat_node import ChatNode
from .models.generator_info import (
    GeneratorInfo,
    GeneratorCompletionParameters,
    HUGGINGFACE_ACTIVATED,
    anthropic,
    openai,
    openrouter
)
from .utils.json_utils import to_dict, extract_json_from_completion
from .utils.message_utils import (
    format_prompt, 
    merge_contiguous_messages,
    NodeCompletionParameters
)

__all__ = [
    'ChatNode',
    'GeneratorInfo',
    'GeneratorCompletionParameters',
    'HUGGINGFACE_ACTIVATED',
    'to_dict',
    'extract_json_from_completion',
    'format_prompt',
    'merge_contiguous_messages',
    'NodeCompletionParameters',
    'anthropic',
    'openai',
    'openrouter',
]