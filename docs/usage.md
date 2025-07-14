# Quickstart

Get started with MiniLLMLib in just a few lines:

```python
import minillmlib as mll

gi = mll.GeneratorInfo(model="gpt-4", _format="openai", api_key="sk-...")
chat = mll.ChatNode(content="Hello!", role="user")
response = chat.complete_one(mll.NodeCompletionParameters(gi=gi))
print(response.content)
```

> **Tip:** For most users, this is all you need. See below for details and advanced usage.

---

# Installation

```bash
# (Recommended) Install from PyPI
pip install minillmlib

# Or install from source
git clone https://github.com/qfeuilla/MiniLLMLib.git
cd MiniLLMLib
pip install -e .

# For HuggingFace/local models:
pip install minillmlib[huggingface]
```

---

# Core Concepts

## 1. ChatNode: Building Conversations

A `ChatNode` represents a single message in a conversation, with a `role` (like `"user"`, `"assistant"`, or `"system"`) and a `content` (the text or payload of the message).

**Creating a Simple Message:**
```python
chat = mll.ChatNode(content="Hello, how are you?", role="user")
```

**Building a Conversation Tree:**
```python
root = mll.ChatNode(content="What's the weather like today?", role="user")
assistant = mll.ChatNode(content="It's sunny and 25°C.", role="assistant")
root.add_child(assistant)
followup = mll.ChatNode(content="Will it rain tomorrow?", role="user")
assistant.add_child(followup)
```

**Visualizing the Structure:**
```
root (user): What's the weather like today?
└── assistant: It's sunny and 25°C.
    └── user: Will it rain tomorrow?
```

**Saving and Loading Threads:**
```python
followup.save_thread("weather_thread.json")
thread = mll.ChatNode.from_thread(path="weather_thread.json")
print(thread.content)  # Will it rain tomorrow?
```
> Saving a thread always saves the path from the current node up to the root. Loom saving is supported, but loom loading is **not yet implemented**.

## 2. GeneratorInfo: Model and Provider Configuration

`GeneratorInfo` holds all configuration needed to interact with a model/provider:
- Model name (e.g. "gpt-4", "claude-3-opus-20240229")
- Provider format (e.g. "openai", "anthropic", "hf", "url")
- API keys and URLs
- Default generation parameters

**Example:**
```python
gi = mll.GeneratorInfo(
    model="gpt-4",
    _format="openai",
    api_key="sk-..."  # Or use environment variable OPENAI_API_KEY
)
```

**Passing Generation Parameters:**

You can pass generation parameters in two ways:

1. **Via `GeneratorInfo`**: Set default generation parameters by passing a `completion_parameters` argument when creating a `GeneratorInfo` object. These will be used for all completions unless overridden.
```python
gi = mll.GeneratorInfo(
    model="gpt-4",
    _format="openai",
    api_key="sk-...",
    completion_parameters=mll.GeneratorCompletionParameters(
        temperature=0.2,
        max_tokens=256
    )
)
```

2. **Per-completion**: Override default parameters by passing a `GeneratorCompletionParameters` object to `complete_one` or `complete_one_async`.
```python
custom_params = mll.GeneratorCompletionParameters(
    temperature=0.1,
    max_tokens=128
)
params = mll.NodeCompletionParameters(gi=gi, generation_parameters=custom_params)
response = chat.complete_one(params)
```

**Conflict Resolution:** If a parameter is set in both `GeneratorInfo` and `GeneratorCompletionParameters`, the per-completion value takes precedence.

## 3. Branching/Loom Conversations

A loom conversation is a tree-like structure where each node can have multiple children. This allows for branching conversations and more complex dialogue flows.

**ASCII Diagram:**
```
root (user): What's the weather like today?
├── assistant: It's sunny and 25°C.
│   ├── user: Will it rain tomorrow?
│   └── user: What's the forecast for the weekend?
└── assistant: It's cloudy and 20°C.
    └── user: Will it rain tomorrow?
```

**Completion Semantics:** When completing a node, the context is set to the node's parents. The completion grows the branch by adding a new child node.

## 4. Merging Contiguous Messages

`merge_contiguous` is a compatibility feature that merges consecutive messages with the same role. This is useful when working with certain models or providers that expect a specific input format.

**Options:**
- `'all'`: Merge all consecutive messages with the same role, regardless of role.
- `'user'`: Only merge consecutive user messages.
- `'assistant'`: Only merge consecutive assistant messages.
- `'system'`: Only merge consecutive system messages.
- `'base'`: Only merge base messages (rare, for advanced workflows).
- `None`: No merging; every message is passed as-is.

**Example:**
```python
params = mll.NodeCompletionParameters(
    gi=gi,
    merge_contiguous="user"
)
```

---

# Basic Usage

## Completion Methods Overview

MiniLLMLib provides several ways to generate completions from a `ChatNode`:

| Method                   | Sync/Async | Returns                | Description                                    |
|-------------------------|------------|------------------------|------------------------------------------------|
| `complete_one(gi)`      | Sync       | Single ChatNode        | One response (most common)                     |
| `complete(gi)`          | Sync       | List[ChatNode] or ChatNode (if `n` is 1)         | All possible completions (if provider supports) |
| `complete_one_async(gi)`| Async      | Single ChatNode        | Async single response                          |
| `complete_async(gi)`    | Async      | List[ChatNode] or ChatNode (if `n` is 1)        | Async all completions                          |

---

## Collapsing (Truncating or Summarizing) Chat Threads

MiniLLMLib provides an advanced method for managing long chat threads: `ChatNode.collapse_thread`. This asynchronous method lets you keep only the most relevant parts of a conversation, either by truncating (removing the middle) or summarizing it with an LLM.

### Method Signature
```python
collapsed = await chat.collapse_thread(keep_last_n, keep_n, gi=None)
```
- `keep_last_n` (int): Number of most recent nodes to keep (from the current node backward).
- `keep_n` (int): Total number of nodes to keep in the thread (including the start and end).
- `gi` (GeneratorInfo | None): If provided, summarizes the truncated section using the model; if None, inserts a prominent truncation marker node.

### Modes
- **Truncation (gi=None):**
  - Middle nodes are replaced by a visually prominent marker node (role="assistant") indicating that a section was truncated.
  - Example:
    ```python
    collapsed = await chat.collapse_thread(keep_last_n=3, keep_n=6, gi=None)
    ```
- **Summarization (gi=GeneratorInfo):**
  - Middle nodes are summarized using the provided generator. A summary node (role="assistant") is inserted, prefixed with a clear message (e.g., "Conversation truncated. Summary of the truncated conversation:").
  - Example:
    ```python
    collapsed = await chat.collapse_thread(keep_last_n=3, keep_n=6, gi=gi)
    ```

### Behavior and Edge Cases
- If the thread is shorter than `keep_n` or `keep_last_n`, no truncation occurs.
- If `keep_n` < 2, only the last node is kept.
- If `keep_last_n` = 0, only the start nodes are kept (up to `keep_n`).
- The method always returns the last node of the collapsed thread.
- The summary/truncation node never inherits metadata or formatting from the truncated section.

### Best Practices
- Use truncation for efficiency or when summarization is not needed.
- Use summarization to preserve context for the model when removing large sections.
- Always use `await` as this method is asynchronous (especially when summarizing).

See tests for more advanced usage and edge case handling.


- For simple use, you can pass a `GeneratorInfo` (`gi`) directly. Use `NodeCompletionParameters` only for advanced options.
- If you want to override defaults or use advanced features, wrap with `NodeCompletionParameters`:
  ```python
  response = chat.complete_one(mll.NodeCompletionParameters(gi=gi, ...))
  ```

## Synchronous Example
```python
response = chat.complete_one(gi)
```

## Asynchronous Example
```python
response = await chat.complete_one_async(gi)
```
> **Note:** Async is supported for all providers **except** HuggingFace/local models.

---

## Advanced Completion Options

You can control completion behavior using `NodeCompletionParameters`. Here are the most important options:

| Parameter                | Type      | Default   | Description                                                                 |
|--------------------------|-----------|-----------|-----------------------------------------------------------------------------|
| `gi`                     | GeneratorInfo | —     | The model/provider configuration                                            |
| `generation_parameters`  | GeneratorCompletionParameters | — | Per-call generation settings (overrides gi defaults)                        |
| `merge_contiguous`       | str/None  | "all"    | Merge consecutive messages with same role (see above)                       |
| `parse_json`             | bool      | False     | Parse/repair model output as JSON                                           |
| `crash_on_refusal`       | bool      | False     | Raise error & retry if model doesn't return JSON (when `parse_json=True`)   |
| `crash_on_empty_response`| bool      | False     | Raise error & retry if model returns empty output                           |
| `retry`                  | int       | 4         | Number of retry attempts on error                                           |
| `exp_back_off`           | bool      | False     | Use exponential backoff between retries                                     |
| `back_off_time`          | int/float | 1         | Initial wait time (seconds) for backoff                                     |
| `max_back_off`           | int/float | 15        | Max wait time (seconds) for backoff                                         |
| `force_prepend`          | str/None  | None      | Force some text to be prepended to the completion before the assistant answer. For example if you want the LLM to start their answer with "Score: " and then continue, you can set `force_prepend="Score: "`. |
| `add_child`              | bool      | False     | Whether to attach the completion as a child node to the current node                            |
| `n`                      | int       | 1         | Number of completions to generate. If > 1, returns a list of completions with `complete(_async)` (`complete_one(_async)` will ignore this parameter). |

**Special Features:**
- **Strict JSON Output:** Use `parse_json=True` and `crash_on_refusal=True` to enforce JSON output and trigger retries/backoff if the model refuses.
- **Crash on Empty:** Use `crash_on_empty_response=True` to ensure you always get a non-empty response.

**Example:**
```python
params = mll.NodeCompletionParameters(
    gi=gi,
    parse_json=True,
    crash_on_refusal=True,
    retry=5,
    exp_back_off=True,
    back_off_time=2,
    max_back_off=30
)
response = chat.complete_one(params)
```

---

## Branching and Loom Semantics

- Each completion grows a branch: calling `.complete_one()` adds a new assistant node as a child.
- Only the path from the current node to the root is used as context for completions (siblings/other branches are ignored).
- Looms enable branching conversations—each node can have multiple children, representing different possible continuations.

---

## Advanced GeneratorInfo Parameters

The `GeneratorInfo` class configures all model/provider and runtime options for completions. Most users only need to set `model`, `_format`, and `api_key`, but advanced users can fine-tune many behaviors.

| Parameter                    | Type      | Default     | Description |
|------------------------------|-----------|-------------|-------------|
| `model`                      | str/None  | None        | Model name (e.g. 'gpt-4', 'claude-3-opus-20240229') |
| `is_chat`                    | bool      | True        | Whether the model uses chat format |
| `completion_parameters`      | GeneratorCompletionParameters | — | Default generation parameters (temperature, max_tokens, etc) |
| `api_url`                    | str/None  | None        | Custom API endpoint (for custom providers) |
| `api_key`                    | str/None  | None        | API key for the provider |
| `_format`                    | str       | "url"      | Provider format: 'openai', 'openai-audio', 'anthropic', 'url', 'mistralai', 'hf', 'prettify' |
| `force_merge`                | bool      | False       | Force merging of contiguous messages |
| `enforce_json_compatible_prompt` | bool  | False       | Ensure prompt is JSON-compatible (rarely needed) |
| `no_system`                  | bool      | False       | Treat all 'system' messages as 'user' |
| `deactivate_temperature`     | bool      | False       | Ignore temperature in completions |
| `deactivate_max_tokens`      | bool      | False       | Ignore max_tokens in completions |
| `price_table`                | tuple     | (0.0, 0.0)  | Cost per input/output token (for tracking) |
| `is_uncensored`              | bool      | False       | Mark model as uncensored (for filtering/auditing) |
| `translation_table`          | dict      | `{}`        | Custom role translations (see 'mll.pretty_messages' configuration) |
| `usage_tracking_type`        | str/None  | None        | Usage tracking backend (currently only "openrouter" supported with openrouter url backend) |
| `usage_db`                   | Collection/None | None | MongoDB collection for usage tracking |
| `usage_id_key`               | str/None  | None        | Key for identifying usage records |
| `usage_id_value`             | str/None  | None        | Value for identifying usage records |
| `usage_key`                  | str/None  | None        | The key to use to update the price in the usage record (this should already exist and be set to 0) |
| `hf_model_kwargs`            | dict      | `{}`        | Extra kwargs for HuggingFace model loading |
| `hf_process_kwargs`          | dict      | `{}`        | Extra kwargs for HuggingFace processor |
| `hf_auto_model`              | object    | None        | Loaded HuggingFace model (internal) |
| `hf_processor`               | object    | None        | Loaded HuggingFace processor (internal) |
| `hf_tokenizer`               | object    | None        | Loaded HuggingFace tokenizer (internal) |
| `hf_device`                  | str       | "cuda:0"   | Device for HuggingFace models |
| `hf_expected_eoc`            | str       | "<|end|>\n"| Expected end-of-completion token for HF models |

> **Note:**
> If your provider does not support `temperature` and/or `max_tokens`, and throws an error when these are sent (for example, some OpenAI reasoning models), you can set `deactivate_max_tokens=True` and/or `deactivate_temperature=True` in your `GeneratorInfo`. This will prevent these parameters from being sent to the provider. See the following code pattern:
>
> ```python
> GeneratorInfo(
>     model="o1-preview",
>     _format="openai",
>     deactivate_max_tokens=True,
>     # ...
> )
> ```
> This is used in the model zoo (see `model_zoo.py`, e.g. `deactivate_max_tokens="o1" in model_name`).

**Usage Tracking:**
- To enable usage tracking, set `usage_tracking_type="openrouter"` and provide a valid `usage_db` (a MongoDB collection). Other backends are not yet supported.
- Use the `usage_id_key`, `usage_id_value`, and `usage_key` fields to customize how usage is tracked.

See :
```python
assistant_builder = mll.GeneratorInfo(
    model="deepseek/deepseek-chat-v3-0324",
    _format="url",
    api_url=f"https://openrouter.ai/api/v1/chat/completions",
    api_key=OPENROUTER_API_KEY,
    completion_parameters=mll.GeneratorCompletionParameters(
        provider={
            "data_collection": "deny",
            "sort": "throughput",
        },
        usage={
            "include": True
        }
    ),
    usage_tracking_type="openrouter", # Only openrouter is supported for now
    usage_id_key="_id", # This will filter object on _id
    usage_id_value=database_objects["_id"], # The value of the _id to find the object to update in the db
    usage_key="cost" # The key to use to update the price in the object record. It should be set to 0 in your db for each object, and mll will update this value
    usage_db=database # database must be a MongoDB collection
)
```

---

# Advanced Usage

## Parametric Prompts with `update_format_kwargs`

MiniLLMLib supports dynamic, programmatic prompting by allowing you to update variables (placeholders) in prompt templates using `update_format_kwargs` on a `ChatNode`.

This is especially powerful when loading prompt trees from files and injecting runtime variables for each completion.

**Example:**
Suppose you have a prompt file with placeholders like `{instructions}` and `{to_transform}`:

```json
{
    "role": "user",
    "content": "Transform this paragraph: {to_transform}\nInstructions: {instructions}"
}
```

You can load this as a node and update the variables dynamically:

```python
node = mll.ChatNode.from_thread(["prompts/pre_prompt.json"])
node.update_format_kwargs(propagate=True, instructions="Rewrite in passive voice", to_transform="The cat chased the mouse.")

response = node.complete_one(gi)
print(response.content)  # The prompt will have the variables filled in
```

- The `propagate=True` option updates all parent nodes up to the root, ensuring all relevant placeholders are filled.
- This enables powerful, reusable prompt trees for advanced programmatic workflows.

---

# Advanced/Beta Features

## Audio Completions (Beta)

> **Beta:** Audio support is experimental and only for OpenAI's audio model. Not for general LLM completions. See this:
```python
to_complete = mll.ChatNode(
    role="system",
    content="You are a helpful assistant that always talk to the user using voice."
)
user = to_complete.add_child(
    mll.ChatNode(
        role="user",
        content="Hello! Please explain why the sky is blue like if I was a child."
    )
)

answer = user.complete_one(
    mll.NodeCompletionParameters(
        gi=mll.openai_audio["gpt-4o-audio-preview"],
    )
)

print(answer.audio_data.audio_paths[0])
print(answer.audio_data.audio_ids) # If the id is available, it will use the voice to keep completing the text, if it is not, it will fall back to using the transcript
```

## HuggingFace/Local Model Usage (Beta)

> **Beta:** Local HuggingFace model support is experimental. Only synchronous completions are supported.
```python
import minillmlib as mll
gi = mll.GeneratorInfo(model="TheBloke/Llama-2-7B-Chat-GPTQ", _format="hf")
chat = mll.ChatNode(content="Hi!", role="user")
response = chat.complete_one(mll.NodeCompletionParameters(gi=gi))
```

---

# Utilities

## Prompt Formatting
```python
from minillmlib.utils.message_utils import format_prompt
prompt = format_prompt("Hello, {name}!", name="Alice")
```

## Logging & Debugging
```python
from minillmlib.utils.logging_utils import get_logger
logger = get_logger()
logger.info("Debug message")
```

---

## GeneratorCompletionParameters Reference

The `GeneratorCompletionParameters` dataclass lets you specify default or per-call generation settings for completions. Any unknown kwargs will be passed directly to the provider API.

| Parameter            | Type    | Default   | Description |
|----------------------|---------|-----------|-------------|
| `temperature`        | float   | 0.8       | Sampling temperature for generation (higher = more random) |
| `max_tokens`         | int     | 512       | Maximum number of tokens to generate |
| `voice`              | str     | "alloy"  | Audio voice (OpenAI audio models only; options: alloy, ash, ballad, coral, echo, sage, shimmer) |
| `audio_output_folder`| str/None| None      | Where to save audio output files (audio completions) |
| `kwargs`             | dict    | `{}`      | Any extra provider/model-specific parameters (e.g., `top_p`, `frequency_penalty`, etc.) |

**Custom kwargs:**
- Any additional keyword arguments not matching a known field will be stored in `kwargs` and passed to the underlying API. This allows you to use new or provider-specific parameters without waiting for library updates.

**Example:**
```python
params = mll.GeneratorCompletionParameters(
    temperature=0.7,
    max_tokens=256,
    top_p=0.95,  # Passed via kwargs
    frequency_penalty=0.1  # Passed via kwargs
)
```

---

## Next Steps
- [Provider Matrix](providers.md)
- [Configuration](configuration.md)
- [Frontend/CLI](frontend.md)