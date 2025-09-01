# Prompt Management

MiniLLMLib's prompt management system allows you to store prompts as JSON files and dynamically fill in variables at runtime.

## JSON Prompt Structure

Prompts are stored as JSON files with this structure:

```json
{
  "required_kwargs": {
    "topic": null,
    "difficulty": null,
    "examples": null
  },
  "prompts": [
    {
      "role": "system",
      "content": "You are a helpful teacher explaining complex topics."
    },
    {
      "role": "user", 
      "content": "Explain {topic} at a {difficulty} level. Include {examples}."
    }
  ]
}
```

## Loading and Templating

```python
import minillmlib as mll

# Load prompt from file
prompt = mll.ChatNode.from_thread("teacher_prompt.json")

# Update template variables
prompt.update_format_kwargs(
    topic="machine learning",
    difficulty="beginner",
    examples="practical applications like recommendation systems"
)

# The prompt content is now dynamically filled
result = prompt.complete_one(gi)
```

## Multi-Stage Prompt Workflows

You can build complex workflows by chaining prompts:

```python
# Load initial analysis prompt
analysis_prompt = mll.ChatNode.from_thread("analysis_start.json")

# Add conversation history
analysis_prompt = analysis_prompt.merge(existing_conversation)

# Add final instructions
final_step = mll.ChatNode.from_thread("analysis_end.json")
analysis_prompt = analysis_prompt.add_child(final_step)

# Fill in variables
analysis_prompt.update_format_kwargs(
    data_source="user feedback",
    analysis_type="sentiment analysis",
    output_format="structured summary"
)

# Execute the workflow
result = await analysis_prompt.complete_async(gi)
```

## Advanced Templating Patterns

### Conditional Content
```python
# Build content conditionally
content_parts = []
if include_examples:
    content_parts.append("Here are some examples:")
if include_warnings:
    content_parts.append("Important warnings to consider:")

final_content = "\n".join(content_parts)
prompt.update_format_kwargs(additional_info=final_content)
```

### Dynamic Prompt Assembly
```python
# Build a complex analysis workflow
base_prompt = mll.ChatNode.from_thread("analysis_base.json")
base_prompt = base_prompt.merge(user_conversation)
final_prompt = base_prompt.add_child(
    mll.ChatNode.from_thread("analysis_conclusion.json")
)

# Fill in all variables
final_prompt.update_format_kwargs(
    analysis_focus="user satisfaction",
    methodology="qualitative analysis",
    output_style="executive summary"
)
```

## Template Variable Propagation

Use `propagate=True` to update variables across the entire conversation tree:

```python
# Updates all parent nodes up to root
prompt.update_format_kwargs(
    propagate=True,
    global_context="important context",
    session_id="user_123"
)
```

## Error Handling for Missing Variables

```python
try:
    formatted_content = prompt.get_messages(gi=gi)
except KeyError as e:
    logger.error(f"Missing template variable: {e}")
    # Handle missing variables gracefully
```

## Best Practices

1. **Store prompts as JSON files** - Version control and easy editing
2. **Use descriptive variable names** - `{user_request}` not `{req}`
3. **Define all variables in required_kwargs** - Documents expected inputs
4. **Use propagate=True sparingly** - Only when variables need global scope
5. **Validate templates** - Check for missing variables before completion
6. **Organize by workflow** - Group related prompts in directories

## Example: Complete Evaluation Workflow

```python
async def evaluate_interaction(request, response, criteria):
    # Load evaluation prompt
    evaluator = mll.ChatNode.from_thread("prompts/tap_judge_evaluator.json")
    
    # Template with interaction data
    evaluator.update_format_kwargs(
        goal=criteria.goal,
        target_str=criteria.target,
        attack_prompt=request,
        target_response=response
    )
    
    # Get evaluation with error handling
    try:
        rating = await evaluator.complete_async(
            mll.NodeCompletionParameters(
                gi=evaluation_gi,
                temperature=0.1,  # Deterministic evaluation
                max_tokens=50,
                crash_on_refusal=True,
                retry=3
            )
        )
        
        # Parse structured output
        match = re.search(r"Rating: \[\[(\d+)\]\]", rating.content)
        if not match:
            raise ValueError(f"Could not parse rating: {rating.content}")
            
        return int(match.group(1))
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise
```
