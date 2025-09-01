# Extending MiniLLMLib

## Adding New Models or Providers

### Using Model Zoo

```python
import minillmlib as mll

# Use pre-configured models
gi = mll.openai["gpt-4o"]
# or
gi = mll.anthropic["claude-3.5-sonnet-20241022"]
# or
gi = mll.openrouter["anthropic/claude-3.5-sonnet"]
```

### Custom Provider Configuration

```python
# Custom OpenAI-compatible API
custom_gi = mll.GeneratorInfo(
    model="custom-model-name",
    _format="url",
    api_url="https://your-api.com/v1/chat/completions",
    api_key="your-api-key",
    completion_parameters=mll.GeneratorCompletionParameters(
        temperature=0.7,
        max_tokens=2048
    )
)

# Local HuggingFace model
hf_gi = mll.GeneratorInfo(
    model="microsoft/DialoGPT-medium",
    _format="hf",
    completion_parameters=mll.GeneratorCompletionParameters(
        max_tokens=512,
        temperature=0.8
    )
)
```

## Multimodal Extensions

### Image Analysis

```python
import minillmlib as mll
import os

def analyze_image(image_path: str, description_prompt: str = "Describe this image in detail"):
    """Analyze an image using vision-capable models."""
    
    # Create image analysis node
    node = mll.ChatNode(
        content=description_prompt,
        image_data=mll.ImageData(images=[image_path])  # Only URLs supported currently
    )
    
    # Use vision-capable model (URL format only)
    gi = mll.GeneratorInfo(
        model="anthropic/claude-3.5-sonnet",
        _format="url",
        api_url="https://openrouter.ai/api/v1/chat/completions",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        completion_parameters=mll.GeneratorCompletionParameters(
            temperature=0.3,
            max_tokens=1024
        )
    )
    
    # Complete analysis with error handling
    result = node.complete_one(mll.NodeCompletionParameters(
        gi=gi,
        retry=2,
        crash_on_empty_response=True
    ))
    
    return result.content if result else None

# Usage with image URLs only
description = analyze_image(
    "https://example.com/image.jpg",
    "What objects and activities do you see in this image?"
)
print(description)
```

### Audio Processing

```python
import minillmlib as mll

def process_audio_conversation(audio_files: list[str]):
    """Process multiple audio files into a conversation."""
    
    # Process and chunk audio files
    processed = mll.process_audio_for_completion(
        file_paths=audio_files,
        target_sample_rate=24000,
        enable_chunking=True,
        max_chunk_size=10 * 1024 * 1024  # 10MB chunks
    )
    
    # Create audio data container
    audio_data = mll.AudioData(
        audio_raw=processed["chunks"][0]  # Use first chunk
    )
    
    # Create conversation node
    node = mll.ChatNode(
        content="Please transcribe and summarize this audio",
        audio_data=audio_data
    )
    
    # Use audio-capable model
    gi = mll.openai_audio["gpt-4o-audio-preview"]
    
    result = node.complete_one(gi)
    return result.content
```

## Custom Message Processing

### Advanced Prompt Formatting

```python
import minillmlib as mll
from minillmlib.utils import format_prompt

def create_structured_prompt(template: str, **kwargs):
    """Create formatted prompts with validation."""
    
    # Format with validation
    try:
        formatted = format_prompt(template, **kwargs)
        return formatted
    except KeyError as e:
        print(f"Missing template variable: {e}")
        return None

# Usage
template = "Analyze {topic} from a {perspective} viewpoint, focusing on {aspects}."
formatted = create_structured_prompt(
    template,
    topic="machine learning",
    perspective="business",
    aspects="ROI and implementation challenges"
)
```

### Message Merging and Processing

```python
import minillmlib as mll
from minillmlib.utils import merge_contiguous_messages

def process_conversation_thread(messages: list[dict]):
    """Process and optimize conversation messages."""
    
    # Merge contiguous messages from same role
    merged = merge_contiguous_messages(
        messages, 
        merge_contiguous="user"  # Only merge user messages
    )
    
    return merged
```

## Error Handling Extensions

```python
import minillmlib as mll
import asyncio
from typing import Optional

async def robust_completion(prompt: str, gi: mll.GeneratorInfo, max_retries: int = 3) -> Optional[str]:
    """Completion with comprehensive error handling."""
    
    node = mll.ChatNode(content=prompt)
    
    completion_params = mll.NodeCompletionParameters(
        gi=gi,
        retry=max_retries,
        exp_back_off=True,
        back_off_time=1.0,
        max_back_off=30,
        crash_on_refusal=False,
        crash_on_empty_response=False
    )
    
    try:
        result = await node.complete_async(completion_params)
        return result.content if result else None
    except Exception as e:
        print(f"Completion failed after {max_retries} retries: {e}")
        return None
```

## Contributing

- Fork the repository and create feature branches
- Add comprehensive tests for new functionality
- Follow existing code style and type hints
- Update documentation for new features
- See [contributing.md](contributing.md) for detailed guidelines
