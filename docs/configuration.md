# Configuration

## Installation

### Basic Installation

```bash
pip install minillmlib
```

### With Optional Dependencies

```bash
# For HuggingFace local models
pip install minillmlib[huggingface]

# For audio processing (required for audio features)
pip install minillmlib[audio]

# For all features
pip install minillmlib[all]
```

### System Dependencies

For audio processing, install FFmpeg:

```bash
# Ubuntu/Debian
sudo apt update && sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

## API Keys & Environment Variables

Set provider API keys as environment variables for security:

```bash
# Core providers
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENROUTER_API_KEY="sk-or-..."
export MISTRAL_API_KEY="..."

# Optional providers
export TOGETHER_API_KEY="..."
export GROQ_API_KEY="..."
```

### Using .env Files

```python
# .env file
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
OPENROUTER_API_KEY=sk-or-...
```

```python
# Load in Python
import minillmlib as mll
from dotenv import load_dotenv
import os

load_dotenv()

gi = mll.GeneratorInfo(
    model="gpt-4o",
    _format="openai",
    api_key=os.getenv("OPENAI_API_KEY")
)
```

## Provider Configuration

### OpenAI

```python
import minillmlib as mll
import os

# Direct configuration
gi = mll.GeneratorInfo(
    model="gpt-4o",
    _format="openai",
    api_key=os.getenv("OPENAI_API_KEY"),
    completion_parameters=mll.GeneratorCompletionParameters(
        temperature=0.7,
        max_tokens=2048,
        top_p=0.9
    )
)

# Or use model zoo
gi = mll.openai["gpt-4o"]
```

### Anthropic

```python
# Direct configuration
gi = mll.GeneratorInfo(
    model="claude-3.5-sonnet-20241022",
    _format="anthropic",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    completion_parameters=mll.GeneratorCompletionParameters(
        temperature=0.7,
        max_tokens=4096
    )
)

# Model zoo
gi = mll.anthropic["claude-3.5-sonnet-20241022"]
```

### OpenRouter

```python
# Access to 100+ models
gi = mll.GeneratorInfo(
    model="anthropic/claude-3.5-sonnet",
    _format="url",
    api_url="https://openrouter.ai/api/v1/chat/completions",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

# Model zoo
gi = mll.openrouter["anthropic/claude-3.5-sonnet"]
```

### HuggingFace (Local)

```python
# Requires: pip install minillmlib[huggingface]
gi = mll.GeneratorInfo(
    model="microsoft/DialoGPT-medium",
    _format="hf",
    completion_parameters=mll.GeneratorCompletionParameters(
        max_tokens=512,
        temperature=0.8,
        do_sample=True
    )
)
```

## Advanced Configuration

### Completion Parameters

```python
import minillmlib as mll

# Comprehensive parameter configuration
params = mll.GeneratorCompletionParameters(
    # Generation settings
    temperature=0.7,
    max_tokens=2048,
    top_p=0.9,
    top_k=50,
    frequency_penalty=0.1,
    presence_penalty=0.1,
    
    # Audio settings (OpenAI)
    voice="alloy",  # alloy, echo, fable, onyx, nova, shimmer
    audio_output_folder="./audio_outputs",
    
    # HuggingFace settings
    do_sample=True,
    num_beams=1,
    early_stopping=False
)

gi = mll.GeneratorInfo(
    model="gpt-4o",
    _format="openai",
    api_key=os.getenv("OPENAI_API_KEY"),
    completion_parameters=params
)
```

### Node Completion Parameters

```python
# Advanced completion control
completion_params = mll.NodeCompletionParameters(
    gi=gi,
    
    # Response handling
    add_child=True,  # Add response as child node
    parse_json=True,  # Auto-parse JSON responses -> still return a str, but it will be loadable with json.loads()
    merge_contiguous="user",  # Merge consecutive messages
    
    # Error handling
    retry=3,
    exp_back_off=True,
    back_off_time=1.0,
    max_back_off=30,
    crash_on_refusal=False,
    crash_on_empty_response=False,
    
    # Multiple completions
    n=1  # Number of completions to generate
)

result = chat.complete_one(completion_params)
```

## Multimodal Configuration

### Audio Processing

```python
# Configure audio processing
audio_params = {
    "target_sample_rate": 24000,
    "enforce_mono": True,
    "bit_depth": 16,
    "enable_chunking": True,
    "max_chunk_size": 10 * 1024 * 1024,  # 10MB
    "enforce_format": "wav"
}

# Process audio files
processed = mll.process_audio_for_completion(
    file_paths=["audio1.mp3", "audio2.wav"],
    **audio_params
)
```

### Image Processing

```python
# Image data currently supports URLs only
image_data = mll.ImageData(images=[
    "https://example.com/image.png",  # URL (supported)
    # Note: Local files and base64 not yet supported for images
])

# Images are automatically processed for API compatibility
processed_images = image_data.get_processed_images()
```

## Usage Tracking & Monitoring

```python
# Enable usage tracking (optional)
gi = mll.GeneratorInfo(
    model="gpt-4o",
    _format="openai",
    api_key=os.getenv("OPENAI_API_KEY"),
    usage_db="mongodb://localhost:27017/llm_usage"  # MongoDB connection
)

# Usage data is automatically tracked for cost monitoring
```

## Logging Configuration

```python
import minillmlib as mll

# Configure logging
logger = mll.get_logger()
mll.configure_logger(level="DEBUG")  # DEBUG, INFO, WARNING, ERROR

# Use in your application
logger.info("Starting LLM completion...")
```

