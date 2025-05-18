# Supported Providers & Models

MiniLLMLib supports a wide range of LLM providers and models out-of-the-box. You can easily switch between them or add your own.

## Provider Matrix

| Provider    | Sync | Async | Audio | Local | API Key/Env | Notes                       |
|-------------|------|-------|-------|-------|-------------|-----------------------------|
| OpenAI      | ✅   | ✅    | ✅    | ❌    | OPENAI_API_KEY | Most models, incl. audio   |
| Anthropic   | ✅   | ✅    | ❌    | ❌    | ANTHROPIC_API_KEY | Claude family           |
| MistralAI   | ✅   | ❌    | ❌    | ❌    | MISTRAL_API_KEY   |                           |
| HuggingFace | ✅   | ❌    | ❌    | ✅    | (local)           | Install `[huggingface]`   |
| Custom URL  | ✅   | ✅    | ❌    | ❌    | api_key param      | OpenAI-compatible APIs    |

## Model Zoo

See `minillmlib.models.model_zoo` for pre-configured models and how to add your own.

---

### Example: Switching Providers

```python
from minillmlib.models.model_zoo import openai, anthropic

gi = openai["gpt-4"]
# or
gi = anthropic["claude-3-opus-20240229"]
```

---

See [Configuration](configuration.md) for setting API keys and environment variables.
