"""MiniLLMLib - A library for interacting with various LLM providers."""

from .core.chat_node import ChatNode
from .models.generator_info import (
    GeneratorInfo,
    GeneratorCompletionParameters,
    HUGGINGFACE_ACTIVATED,
    pretty_messages
)
from .utils.json_utils import to_dict, extract_json_from_completion
from .utils.message_utils import (
    format_prompt, 
    merge_contiguous_messages,
    NodeCompletionParameters,
    AudioData,
    process_audio_for_completion,
    base64_to_wav,
    base64_to_temp_audio_file
)

from .models.model_zoo import (
    anthropic,
    openai,
    openrouter,
    openai_audio
)

from .utils.logging_utils import get_logger, configure_logger

__all__ = [
    'ChatNode',
    'GeneratorInfo',
    'GeneratorCompletionParameters',
    'HUGGINGFACE_ACTIVATED',
    'pretty_messages',
    'to_dict',
    'extract_json_from_completion',
    'format_prompt',
    'merge_contiguous_messages',
    'NodeCompletionParameters',
    'anthropic',
    'openai',
    'openai_audio',
    'openrouter',
    'AudioData',
    'process_audio_for_completion',
    'base64_to_wav',
    'base64_to_temp_audio_file',
    'get_logger',
    'configure_logger'
]