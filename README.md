# MiniLLMLib

A Python library for interacting with various LLM providers (OpenAI, Anthropic, Mistral, Together, HuggingFace, through URL).

Author: Quentin Feuillade--Montixi

## Installation

### From Source
```bash
git clone https://github.com/yourusername/MiniLLMLib.git
cd MiniLLMLib
pip install -e .  # Install in editable mode
```

## Usage

```python
import minillmlib as lel

# Create a generator info
gi = lel.GeneratorInfo(
    model="gpt-4",
    _format="openai",
    api_key="your-api-key"  # Or use environment variable OPENAI_API_KEY
)

# Create a chat node
chat = lel.ChatNode(content="Hello!", role="user")

# Complete with defaults
response = chat.complete_one(lel.CompletionParameters(gi=gi))

# Or use async version
response = await chat.complete_one_async(lel.CompletionParameters(gi=gi))

# Print the response
print(response.content)
```

## Features

- Support for multiple LLM providers:
  - OpenAI
  - Anthropic
  - Mistral
  - Together
  - HuggingFace (local models)
  - URL (e.g. OpenRouter) -> Must be in "OpenAI" request format
- Async support
- Conversation threading
- Flexible parameter configuration
- Type hints and dataclass-based API

## Development

Run tests:
```bash
pytest tests/
```