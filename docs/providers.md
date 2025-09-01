# Supported Providers & Models

MiniLLMLib supports a wide range of LLM providers with full multimodal capabilities. Switch between providers seamlessly using the same API.

## Provider Matrix

| Provider    | Sync | Async | Images | Audio | Local | API Key/Env | Notes                       |
|-------------|------|-------|--------|-------|-------|-------------|-----------------------------|
| OpenAI      | ✅   | ✅    | ❌     | ✅    | ❌    | OPENAI_API_KEY | GPT-4o, audio generation   |
| Anthropic   | ✅   | ✅    | ❌     | ❌    | ❌    | ANTHROPIC_API_KEY | Claude 3.5 Sonnet         |
| OpenRouter  | ✅   | ✅    | ✅     | ❌    | ❌    | OPENROUTER_API_KEY | 100+ models, vision support |
| HuggingFace | ✅   | ❌    | ❌     | ❌    | ✅    | (local)        | Install `[huggingface]`   |
| Custom URL  | ✅   | ✅    | ✅     | ❌    | ❌    | api_key param  | OpenAI-compatible APIs    |

## Model Zoo

See `minillmlib.models.model_zoo` for pre-configured models and how to add your own.

---

### Example: Switching Providers

```python
import minillmlib as mll

# Pre-configured models from model zoo
gi_openai = mll.openai["gpt-4o"]
gi_anthropic = mll.anthropic["claude-3.5-sonnet-20241022"]
gi_openrouter = mll.openrouter["anthropic/claude-3.5-sonnet"]

# Use any provider the same way
chat = mll.ChatNode(content="Hello, world!")
response = chat.complete_one(gi_openai)
print(response.content)
```

## Multimodal Capabilities

### Vision Models

```python
import minillmlib as mll

# Image analysis with URL format providers only
image_node = mll.ChatNode(
    content="What do you see in this image?",
    image_data=mll.ImageData(images=["https://example.com/image.jpg"])
)

# OpenRouter (supports vision models)
gi_router = mll.GeneratorInfo(
    model="google/gemini-pro-vision",
    _format="url",
    api_url="https://openrouter.ai/api/v1/chat/completions",
    api_key="your-openrouter-key"
)

# MistralAI Pixtral
gi_mistral = mll.GeneratorInfo(
    model="pixtral-12b-2409",
    _format="url",
    api_url="https://api.mistral.ai/v1/chat/completions",
    api_key="your-mistral-key"
)

# Custom OpenAI-compatible API with vision
gi_custom = mll.GeneratorInfo(
    model="custom-vision-model",
    _format="url",
    api_url="https://your-api.com/v1/chat/completions",
    api_key="your-api-key"
)

# Image analysis works with URL format providers
result = image_node.complete_one(gi_router)
print(result.content)
```

### Audio Generation (OpenAI)

```python
import minillmlib as mll

# Audio-capable model
audio_gi = mll.openai_audio["gpt-4o-audio-preview"]

# Create conversation
chat = mll.ChatNode(content="Please respond with audio")
result = chat.complete_one(audio_gi)

# Result can be AudioData or text
if isinstance(result.content, mll.AudioData):
    print(f"Audio file saved: {result.content.audio_paths[0]}")
else:
    print(f"Text response: {result.content}")
```

---

## Advanced Configuration

```python
import minillmlib as mll

# Custom completion parameters
advanced_gi = mll.GeneratorInfo(
    model="gpt-4o",
    _format="openai",
    api_key="your-key",
    completion_parameters=mll.GeneratorCompletionParameters(
        temperature=0.7,
        max_tokens=2048,
        top_p=0.9,
        frequency_penalty=0.1,
        presence_penalty=0.1,
        voice="alloy"  # For audio models
    )
)

# Advanced completion with error handling
completion_params = mll.NodeCompletionParameters(
    gi=advanced_gi,
    retry=3,
    exp_back_off=True,
    crash_on_refusal=False,
    parse_json=True,  # Auto-parse JSON responses
    merge_contiguous="user"  # Merge consecutive user messages
)

result = chat.complete_one(completion_params)
```

See [Configuration](configuration.md) for API keys and [Usage](usage.md) for detailed examples.
