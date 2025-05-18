# Extending MiniLLMLib

## Adding New Models or Providers

- Add a new entry to the relevant dictionary in `minillmlib.models.model_zoo`.
- Use `GeneratorInfo` for custom configuration.
- For new providers, implement completion logic in `ChatNode` as needed.

## Custom Message Processing
- Use/extend utilities in `minillmlib.utils.message_utils`.

## Contributing
- See [contributing.md](contributing.md).
