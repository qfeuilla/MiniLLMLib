# Configuration

## API Keys & Environment Variables

Set provider API keys as environment variables for security and convenience:

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=...
export MISTRAL_API_KEY=...
export TOGETHER_API_KEY=...

export OPENROUTER_API_KEY=...
# '-> This will need to be passed as api_key=os.getenv("OPENROUTER_API_KEY")
```

You can also use a `.env` file (see `python-dotenv`):

```
OPENAI_API_KEY=sk-...
```

## Custom Parameters

- All completion/generation parameters can be set via `GeneratorCompletionParameters`.
- Use `NodeCompletionParameters` for advanced control (retries, JSON parsing, etc).
see [usage.md](usage.md) for details.

## Model Zoo

See [providers.md](providers.md) for available models and [usage.md](usage.md) to learn how to configure your own.

