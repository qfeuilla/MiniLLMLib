"""Message processing utilities for MiniLLMLib."""
from dataclasses import dataclass
from typing import Dict, List, Optional

from ..models.generator_info import GeneratorCompletionParameters, GeneratorInfo


# NOTE: This is separated because more features could be added to this like tool use
def format_prompt(
    prompt: str, 
    **kwargs
) -> str:
    """Format a prompt with given kwargs."""
    return prompt.format(**kwargs)


def merge_contiguous_messages(
    messages: List[Dict[str, str]], 
    merge_contiguous: Optional[str] = None
) -> List[Dict[str, str]]:
    """Merge contiguous messages with the same role or all if merge_contiguous is all."""

    if merge_contiguous is None:
        return messages

    if merge_contiguous not in [
        "all",
        "user",
        "assistant",
        "system",
        "base",
    ]:
        raise ValueError(
            "merge_contiguous must be one of None, 'all', 'user', 'assistant', 'system', 'base'"
        )

    result = []
    previous_role = None
    system_ended = False
    
    for message in messages:
        if (
            "role" not in message or
            "content" not in message or
            (content := message["content"]) is None or
            not isinstance(content, str) or
            (role := message["role"]) not in ["system", "user", "assistant", "base"]
        ):
            raise ValueError(
                "Message content must be a string and role must be one of 'system', 'user', 'assistant' or 'base'. Message: " + str(message)
            )

        # If the current message is a "system" but it is not at the beginning, then make it a user
        if role == "system":
            if system_ended:
                role = "user"
        elif not system_ended:
            system_ended = True

        if (role == merge_contiguous or merge_contiguous == "all") and \
            len(result) > 0 and \
            previous_role == role:
                result[-1]["content"] += "\n" + content
        else:
            result.append({"role": role, "content": content})

        previous_role = role

    return result

def hf_process_messages(
    gi: GeneratorInfo, 
    messages: List[Dict[str, str]], 
    force_prepend: str | None = None, 
    padding: bool = False
) -> str:
        prompt = gi.hf_tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=force_prepend is None
        )

        if prompt.endswith(gi.hf_tokenizer.eos_token):
            prompt = prompt.rstrip(gi.hf_tokenizer.eos_token)

        if force_prepend is not None and prompt.endswith(gi.hf_expected_eoc):
            prompt = prompt[:-len(gi.hf_expected_eoc)]

        try:
            inputs = gi.hf_processor(
                prompt, 
                return_tensors="pt", 
                padding=padding
            ).to(gi.hf_device)
        except:
            inputs = gi.hf_tokenizer(
                prompt, 
                return_tensors="pt", 
                padding=padding
            ).to(gi.hf_device)
        
        return inputs

@dataclass
class NodeCompletionParameters:
    gi: GeneratorInfo
    parameters: Optional[GeneratorCompletionParameters] = None
    add_child: bool = False
    parse_json: bool = False
    crash_on_refusal: bool = False
    merge_contiguous: str = "all"
    retry: int = 4 
    force_prepend: Optional[str] = None
    exp_back_off: bool = False
    backoff_time: float = 5
    max_back_off: int = 15
    crash_on_empty_response: bool = False