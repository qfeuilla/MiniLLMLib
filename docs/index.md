# MiniLLMLib

Production-ready Python library for sophisticated LLM workflows and conversation management.

---

**MiniLLMLib** is designed for building complex, real-world LLM applications with advanced conversation management, robust error handling, and production-grade features.

## Key Features
- 🌐 **OpenRouter-first**: Optimized for OpenRouter API with 100+ models
- 🧠 **Conversation Trees**: Build complex branching dialogues and conversation flows
- 📁 **JSON Prompt Management**: Load, merge, and template prompts from files
- 🔄 **Dynamic Templating**: Runtime variable injection with `format_kwargs`
- 💰 **Cost Tracking**: Built-in usage monitoring and cost management
- ⚡ **Production Ready**: Comprehensive error handling, retries, async support
- 🎯 **Advanced Completion**: JSON parsing, validation, structured outputs
- 🎤 **Multimodal Support**: Audio input (WAV/MP3) and image processing

## Real-World Use Cases
- Multi-agent conversation systems
- Complex prompt engineering workflows
- Production chatbots with cost tracking
- AI evaluation and testing frameworks
- Dynamic prompt templating systems

---

## Quick Start

```python
import minillmlib as mll
import os

# Option 1: Direct OpenAI
gi_openai = mll.GeneratorInfo(
    model="gpt-4o",
    _format="openai",
    api_key=os.getenv("OPENAI_API_KEY")
)

# Option 2: Anthropic Claude
gi_anthropic = mll.GeneratorInfo(
    model="claude-3.5-sonnet-20241022",
    _format="anthropic",
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

# Option 3: OpenRouter (access to 100+ models)
gi_openrouter = mll.GeneratorInfo(
    model="anthropic/claude-3.5-sonnet",
    _format="url",
    api_url="https://openrouter.ai/api/v1/chat/completions",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

# Use any provider the same way
chat = mll.ChatNode(content="Explain quantum computing simply", role="user")
response = chat.complete_one(gi_openai)  # or gi_anthropic, gi_openrouter
print(response.content)

# Load prompt templates (works with any provider)
prompt = mll.ChatNode.from_thread("my_prompt.json")
prompt.update_format_kwargs(topic="quantum computing", audience="beginners")
result = prompt.complete_one(gi_anthropic)
```

## Documentation
- [**Usage Guide**](usage.md) - Real-world patterns and examples
- [**Prompt Management**](prompt-management.md) - JSON templates and workflows
- [**Providers**](providers.md) - Supported models and capabilities
- [**Configuration**](configuration.md) - Setup and advanced options
- [**Extending**](extending.md) - Custom providers and multimodal usage

---

[GitHub](https://github.com/qfeuilla/MiniLLMLib) · [PyPI](https://pypi.org/project/minillmlib/)

```{toctree}
:maxdepth: 2

usage.md
providers.md
configuration.md
contributing.md
extending.md
troubleshooting.md
```
