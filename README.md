# MiniLLMLib

[![GitHub stars](https://img.shields.io/github/stars/qfeuilla/MiniLLMLib?style=social)](https://github.com/qfeuilla/MiniLLMLib/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/qfeuilla/MiniLLMLib?style=social)](https://github.com/qfeuilla/MiniLLMLib/network/members)
[![GitHub issues](https://img.shields.io/github/issues/qfeuilla/MiniLLMLib)](https://github.com/qfeuilla/MiniLLMLib/issues)
[![GitHub last commit](https://img.shields.io/github/last-commit/qfeuilla/MiniLLMLib)](https://github.com/qfeuilla/MiniLLMLib/commits/main)

[![PyPI version](https://img.shields.io/pypi/v/minillmlib.svg)](https://pypi.org/project/minillmlib/)
[![Docs](https://readthedocs.org/projects/minillmlib/badge/?version=latest)](https://minillmlib.readthedocs.io/en/latest/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/pypi/pyversions/minillmlib.svg)](https://pypi.org/project/minillmlib/)

---

## Installation

```bash
pip install minillmlib
# For HuggingFace/local models: (Beta - not well tested)
pip install minillmlib[huggingface]
```

---

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

- See the [Usage Guide](https://minillmlib.readthedocs.io/en/latest/usage.html) for advanced usage, parameter tables, and branching/loom semantics.
- See the [Provider Matrix](https://minillmlib.readthedocs.io/en/latest/providers.html) for supported models and configuration tips.
- See [Troubleshooting](https://minillmlib.readthedocs.io/en/latest/troubleshooting.html) for common issues and debugging.

## Configuration

- Set API keys as environment variables for security (see the [Configuration Guide](https://minillmlib.readthedocs.io/en/latest/configuration.html)).

## Development & Contribution

- Run tests with:
  ```bash
  pytest tests/
  ```
- See [Contributing](https://minillmlib.readthedocs.io/en/latest/contributing.html) for contribution guidelines.

---

For more, see the full documentation at [minillmlib.readthedocs.io](https://minillmlib.readthedocs.io/) or open an issue on [GitHub](https://github.com/qfeuilla/MiniLLMLib) if you need help.
