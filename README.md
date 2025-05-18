# MiniLLMLib

A Python library for interacting with various LLM providers (OpenAI, Anthropic, Mistral, HuggingFace, through URL).

Author: Quentin Feuillade--Montixi

## Installation

### From Source
```bash
git clone https://github.com/qfeuilla/MiniLLMLib.git
cd MiniLLMLib
pip install -e .  # Install in editable mode
```

## Usage

```python
import minillmlib as mll

# Create a GeneratorInfo for your model/provider
import os

gi = mll.GeneratorInfo(
    model="gpt-4",
    _format="openai",
    api_key=os.getenv("OPENAI_API_KEY")  # Recommended: use env var for secrets
)

# Create a chat node (conversation root)
chat = mll.ChatNode(content="Hello!", role="user")

# Synchronous completion
response = chat.complete_one(gi)
print(response.content)

# Or asynchronous version
# response = await chat.complete_one_async(gi)

```

## Features

- Unified interface for major LLM providers:
  - OpenAI, Anthropic, Mistral, HuggingFace (local), custom URL (e.g. OpenRouter)
- Thread (linear) and loom (tree/branching) conversation modes
- Synchronous & asynchronous API
- Audio completions (OpenAI audio models, beta)
- Flexible parameter/config management via `GeneratorInfo` and `GeneratorCompletionParameters`
- Save/load conversation trees
- Extensible: add new models/providers easily

## Documentation

- See [docs/usage.md](docs/usage.md) for advanced usage, parameter tables, and branching/loom semantics.
- See [docs/providers.md](docs/providers.md) for supported models and configuration tips.
- See [docs/troubleshooting.md](docs/troubleshooting.md) for common issues and debugging.

## Configuration

- Set API keys as environment variables for security (see [docs/configuration.md](docs/configuration.md)).

## Development & Contribution

- Run tests with:
  ```bash
  pytest tests/
  ```
- See [docs/contributing.md](docs/contributing.md) for contribution guidelines.

---

For more, see the full documentation in the `docs/` folder or open an issue on GitHub if you need help.
