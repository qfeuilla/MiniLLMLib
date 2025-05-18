# Troubleshooting

## Common Issues
- **Missing API key:** Set the required environment variable.
- **Provider errors:** Check model name, API key, and provider status.
- **Audio errors:** Ensure `pydub` and ffmpeg are installed for audio features.
- **HuggingFace errors:** Install `[huggingface]` extras and required models.

- **Timeouts:** Currently the async url chat completion has timeouts and parallelism as following:
```python
timeout = httpx.Timeout(
    connect=10.0, 
    read=300.0,
    write=10.0,
    pool=10.0
)
limits = httpx.Limits(
    max_keepalive_connections=20, 
    max_connections=20, 
    keepalive_expiry=30
)
client = httpx.AsyncClient(verify=False, timeout=timeout, limits=limits)
```
This is still a feature in development. This will be added as arguments to the `NodeCompletionParameters` class asap.

---

## Debugging
- Use `minillmlib.utils.logging_utils.get_logger()` to enable debug output.
- Check API responses and error messages.

## Getting Help
- Open an issue on [GitHub](https://github.com/qfeuilla/MiniLLMLib/issues)
