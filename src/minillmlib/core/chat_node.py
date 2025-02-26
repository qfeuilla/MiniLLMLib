"""Core ChatNode implementation for MiniLLMLib."""
from __future__ import annotations

import asyncio
import json
import os
import time
import warnings
from copy import deepcopy
from typing import Any, Dict, List, Optional

import aiohttp
import requests
from anthropic import Anthropic, AsyncAnthropic
from anthropic.types.message import Message
from colorama import Fore
from mistralai import Mistral
from openai import AsyncOpenAI, OpenAI
from openai.types.chat import ChatCompletion
from together import AsyncTogether, Together

from ..models.generator_info import (HUGGINGFACE_ACTIVATED,
                                     GeneratorCompletionParameters,
                                     GeneratorInfo, pretty_messages, torch)
from ..utils.json_utils import extract_json_from_completion, to_dict
from ..utils.message_utils import (NodeCompletionParameters, format_prompt,
                                   get_payload, hf_process_messages,
                                   merge_contiguous_messages)

warnings.filterwarnings("ignore", message=".*verification is strongly advised.*")

try:
    anthropic_api = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", None))
except:
    anthropic_api = None
try:
    anthropic_async_api = AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", None))
except:
    anthropic_async_api = None


try:
    openai_api = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", None))
except:
    openai_api = None
try:
    openai_async_api = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY", None))
except:
    openai_async_api = None


try:
    mistralai_api = Mistral(api_key=os.environ.get("MISTRAL_API_KEY", None))
except:
    mistralai_api = None


try:
    together_api = Together(api_key=os.environ.get("TOGETHER_API_KEY", None))
except:
    together_api = None
try:
    together_async_api = AsyncTogether(api_key=os.environ.get("TOGETHER_API_KEY", None))
except:
    together_async_api = None

# Create a single shared session object at the module level
_AIOHTTP_SESSION = None

async def get_aiohttp_session():
    global _AIOHTTP_SESSION
    if _AIOHTTP_SESSION is None or _AIOHTTP_SESSION.closed:
        connector = aiohttp.TCPConnector(limit=100, limit_per_host=100, ssl=False)
        timeout = aiohttp.ClientTimeout(total=60)
        _AIOHTTP_SESSION = aiohttp.ClientSession(connector=connector, timeout=timeout)
    return _AIOHTTP_SESSION


# NOTE: This object can be used in two shapes:
# - A Thread (in case there is only one child per node) --> I use it mostly like this. Loom will be for when we'll do the cyborg tool
# - A Loom (in case there are multiple children per node)
class ChatNode:    
    def __init__(self, 
        content: str, 
        role: str = "user", 
        format_kwargs: Optional[Dict[str, Any]] = None
    ):
        """Initialize a ChatNode.
        
        Args:
            content: The content of the message
            role: Role of the message sender (user/assistant/system/base)
            format_kwargs: Optional formatting arguments for the content
        """

        if role not in [
            "user",
            "assistant",
            "system",
            "base",
        ]: 
            raise ValueError("role must be one of 'user', 'assistant', 'system' or 'base'")

        self.role = role
        self.content = content

        self.children: List[ChatNode] = []
        self._parent: Optional[ChatNode] = None
        
        self.metadata: Dict[str, Any] = {}

        # The self.format_kwargs are loaded only if they can be retrieved in the content.
        self.format_kwargs: Dict[str, str] = {
            k: v for k, v in (format_kwargs or {}).items() if f"{{{k}}}" in self.content
        }

    def is_root(self) -> bool:
        """Check if this node is the root of the tree."""
        return self._parent is None

    def get_root(self) -> ChatNode:
        """Get the root node of the tree."""
        return self if self.is_root() else self._parent.get_root()

    def add_child(self, 
        child: ChatNode, 
        illegitimate: bool = False
    ) -> None:
        """Add a child node to this node.
        
        Args:
            child: The child node to add
            illegitimate: If True, only the child knows it's parent but the parent doesn't get the child added
        """
        child._parent = self

        if not illegitimate:
            self.children.append(child)

            # For Thread mode, save the format_kwargs of the child to the root of the tree (only the root's format_kwargs are saved in thread mode)
            root = self.get_root()
            for key in child.format_kwargs:
                if key not in root.format_kwargs:
                    root.format_kwargs[key] = child.format_kwargs[key]
                else:
                    print(f"{Fore.YELLOW}Caution:{Fore.RESET} format_kwargs key {key} already exists in the root of the tree. After loading, only one will be used.")

        return child

    def get_messages(self, 
        gi: GeneratorInfo = pretty_messages, 
        merge_contiguous: Optional[str] = "all"
    ) -> List[Dict[str, str]] | str:
        """Get all messages in the conversation path to this node."""
        if merge_contiguous not in [
            None,
            "all",
            "user",
            "assistant",
            "system",
            "base",
        ]:
            raise ValueError(
                "merge_contiguous must be one of None, 'all', 'user', 'assistant', 'system', 'base'"
            )

        messages : List[Dict[str, str]] = []
        current = self
        while current is not None:
            if gi.is_chat and current.role == "base":
                raise ValueError("The role 'base' is not allowed in chat models")
            
            try:
                content = format_prompt(current.content, **current.format_kwargs)
            except Exception as e:
                print(f"{Fore.YELLOW}Error while formatting the content (formatting skipped){Fore.RESET}: {current.content}: {e}")
                content = current.content

            messages.append({"role": current.role, "content": content})
            current = current._parent

        messages.reverse()

        # If no_system, turn all system messages to user
        if gi.no_system:
            for msg in messages:
                if msg["role"] == "system":
                    msg["role"] = "user"

        messages = merge_contiguous_messages(
            messages=messages,
            merge_contiguous="all" if gi.force_merge else merge_contiguous,
        )

        # NOTE: This makes sure that the content is a string that can be JSON decoded without issues (useful in case the LLM API url is not well coded, but it could affect the prompt so it should not be on by default)
        if gi.enforce_json_compatible_prompt:
            messages = [
                {
                    "role": message["role"],
                    "content": json.dumps({"prompt": message["content"]})[12:-2],
                } for message in messages
            ]

        # Choose the correct formatting depending on the gi _format
        # Opt 1: Do Nothing
        if gi._format in ["openai", "anthropic", "url", "mistralai", "hf", "together"]:
            pass
        # Opt 2: Use the translation table
        elif gi._format in ["prettify"]:
            # Check that the translation table is complete
            if not all(
                role in gi.translation_table for role in ["user", "assistant", "system", "base"]
            ):
                raise ValueError(
                    f"The translation table must be complete for the model {gi._format}, with all the keys ['user', 'assistant', 'system', 'base']"
                )
            
            messages = "".join(
                f"{gi.translation_table[msg['role']]}{msg['content']}"
                for msg in messages
            )
        else:
            raise NotImplementedError(f"{gi._format} not supported. It must be one of ['openai', 'anthropic', 'url', 'mistralai', 'hf', 'together', 'prettify'] to support get_messages")
    
        return messages
    
    def get_child(self,
        _map: List[int] = None
    ) -> ChatNode:
        """Get the child of this node using a map."""
        node = self
        i = 0
        while len(node.children) > 0:
            node = node.children[-1 if _map is None else _map[i]]
            i += 1
            if _map is not None and i >= len(_map):
                break
        return node
    
    def get_last_child(self) -> ChatNode:
        """Get the last child of this node taking the last child each time."""
        return self.get_child()
    
    def merge(self,
        other: ChatNode
    ) -> ChatNode:
        """Merge the root of the other tree to the current node."""
        self.add_child(other.get_root())
        return other
    
    def update_format_kwargs(self,
        propagate: bool = True,
        **kwargs
    ) -> None:
        """Update the format_kwargs in the current node and in all the parents until the root if propagate is True."""
        if self._parent is not None and propagate:
            self._parent.update_format_kwargs(propagate=propagate, **kwargs)

        for k in kwargs:
            # Only update the parameter if it's in the format_kwargs
            if k in self.format_kwargs:
                self.format_kwargs[k] = kwargs[k]
    
    def save_thread(self,
        path: str
    ) -> None:
        node_list = []
        node = self
        while node:
            node_list.append(node)
            node = node._parent
        node_list.reverse()

        json.dump({
            "required_kwargs": self.get_root().format_kwargs,
            "prompts": [
                {"role": node.role, "content": node.content} for node in node_list
            ],
        }, open(path, "w+"), indent=4)
    
    def save_loom(self,
        path: str
    ) -> None:
        json.dump(to_dict(self.get_root()), open(path, "w+"), indent=4)
    
    def __chat_complete_openai(self,
        gi: GeneratorInfo,
        messages: List[Dict[str, str]],
        api: OpenAI,
        *_1, **_2
    ) -> str:
        response: ChatCompletion = api.chat.completions.create(**get_payload(gi, messages))
        return response.choices[0].message.content

    async def __chat_complete_openai_async(self,
        gi: GeneratorInfo,
        messages: List[Dict[str, str]],
        api: AsyncOpenAI,
        *_1, **_2
    ) -> str:
        response: ChatCompletion = await api.chat.completions.create(**get_payload(gi, messages))
        return response.choices[0].message.content

    def __chat_complete_anthropic(self,
        gi: GeneratorInfo,
        messages: List[Dict[str, str]],
        api: Anthropic,
        *_1, **_2
    ) -> str:
        system_prompt = ""
        while messages[0]["role"] == "system":
            system_prompt += messages[0]["content"] + "\n"
            messages = messages[1:]
        system_prompt = system_prompt[:-1]

        payload = get_payload(gi, messages)

        response: Message = api.messages.create(
            system=system_prompt if system_prompt else None,
            **payload
        )

        return response.content[0].text

    async def __chat_complete_anthropic_async(self,
        gi: GeneratorInfo,
        messages: List[Dict[str, str]],
        api: AsyncAnthropic,
        *_1, **_2
    ) -> str:
        system_prompt = ""
        while messages[0]["role"] == "system":
            system_prompt += messages[0]["content"] + "\n"
            messages = messages[1:]
        system_prompt = system_prompt[:-1]

        payload = get_payload(gi, messages)

        response: Message = await api.messages.create(
            system=system_prompt if system_prompt else None,
            **payload
        )

        return response.content[0].text


    def __chat_complete_mistralai(self,
        gi: GeneratorInfo,
        messages: List[Dict[str, str]],
        api: Mistral,
        *_1, **_2
    ) -> str:
        response = api.chat.complete(**get_payload(gi, messages))
        return response.choices[0].message.content
    
    async def __chat_complete_mistralai_async(self,
        gi: GeneratorInfo,
        messages: List[Dict[str, str]],
        api: Mistral,
        *_1, **_2
    ) -> str:
        response = await api.chat.complete_async(**get_payload(gi, messages))
        return response.choices[0].message.content

    def __chat_complete_together(self,
        gi: GeneratorInfo,
        messages: List[Dict[str, str]],
        api: Together,
        *_1, **_2
    ) -> str:
        response = api.chat.completions.create(**get_payload(gi, messages))
        return response.choices[0].message.content

    async def __chat_complete_together_async(self,
        gi: GeneratorInfo,
        messages: List[Dict[str, str]],
        api: AsyncTogether,
        *_1, **_2
    ) -> str:
        response = await api.chat.completions.create(**get_payload(gi, messages))
        return response.choices[0].message.content
    
    def __chat_complete_url(self,
        gi: GeneratorInfo,
        messages: List[Dict[str, str]],
        api_url: str,
        *_1, **_2
    ) -> str:
        
        headers = {"Authorization": f"Bearer {gi.api_key}"}
        response = requests.post(
            api_url,
            headers=headers,
            json={**get_payload(gi, messages)},
            verify=False
        )
        try:
            response = response.json()
        except Exception as e:
            raise Exception(f"{Fore.RED}Error: {Fore.RESET} {e} \n Response: {response}")

        if "choices" not in response:
            raise Exception(f"{Fore.RED}Error: {Fore.RESET} {response}")

        return response["choices"][0]["message"]["content"]
    
    async def __chat_complete_url_async(self,
        gi: GeneratorInfo,
        messages: List[Dict[str, str]],
        api_url: str,
        *_1, **_2
    ) -> str:
        headers = {"Authorization": f"Bearer {gi.api_key}"}

        session = await get_aiohttp_session()

        async with session.post(
            api_url,
            headers=headers,
            json={**get_payload(gi, messages)},
            raise_for_status=True
        ) as response:
            try:
                response = await response.json()
            except Exception as e:
                raise Exception(f"{Fore.RED}Error: {Fore.RESET} {e} \n Response: {response}")

        if "choices" not in response:
            raise Exception(f"{Fore.RED}Error: {Fore.RESET} {response}")

        return response["choices"][0]["message"]["content"]

    def __chat_complete_hf(self,
        gi: GeneratorInfo,
        messages: List[Dict[str, str]],
        _1,
        force_prepend: str,
        *_2, **_3
    ) -> str:
        if gi.hf_auto_model is None or gi.hf_processor is None:
            gi.build_hf_model()
        
        inputs = hf_process_messages(
            gi=gi, 
            messages=messages, 
            force_prepend=force_prepend
        )

        generate_ids = gi.hf_auto_model.generate(
            **inputs, 
            max_new_tokens=gi.completion_parameters.max_tokens,
            eos_token_id=gi.hf_tokenizer.eos_token_id, 
            temperature=gi.completion_parameters.temperature,
            **gi.completion_parameters.kwargs,
        )

        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]

        decoded = gi.hf_tokenizer.batch_decode(
            generate_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]

        if force_prepend is not None:
            decoded = " " + decoded

        return decoded

    async def general_complete_one(self, 
        completion_params: NodeCompletionParameters | GeneratorInfo,
        use_async: bool = False
    ) -> ChatNode:
        if isinstance(completion_params, GeneratorInfo):
            return self.complete_one(NodeCompletionParameters(gi=completion_params))

        gi: GeneratorInfo
        generation_parameters: Optional[GeneratorCompletionParameters]
        add_child: bool
        parse_json: bool
        crash_on_refusal: bool
        merge_contiguous: bool
        retry: int
        force_prepend: Optional[str]
        exp_back_off: bool 
        back_off_time: float
        max_back_off: int
        crash_on_empty_response: bool
        gi, generation_parameters, add_child, parse_json, crash_on_refusal, merge_contiguous, retry, force_prepend, exp_back_off, back_off_time, max_back_off, crash_on_empty_response = completion_params.__dict__.values()

        if gi.is_chat == False:
            raise NotImplementedError(f"Non chat completion is not supported for now")
        
        if gi._format not in ["openai", "anthropic", "url", "mistralai", "hf", "together"]:
            raise NotImplementedError(f"{gi._format} not supported for chat completion")
        
        # deepcopy the gi to avoid modifying the original
        # TODO: This might not be the best idea for local LLMs, I need to find a way to handle them properly without having to clone the weights
        gi = deepcopy(gi)

        # NOTE (design choice): the kwargs here are added to the ones in the gi.completions_parameters, and overwrite them if they are already present
        if generation_parameters is not None:
            for k, v in gi.completion_parameters.kwargs.items():
                if k not in generation_parameters.kwargs:
                    generation_parameters.kwargs[k] = v
            
            gi.completion_parameters = generation_parameters

        back_off = True
        retry = max(retry + 1, 1)

        complete_from = self
        if force_prepend is not None:
            complete_from = self.add_child(
                ChatNode(content=force_prepend, role="assistant"), illegitimate=True
            )

        messages = complete_from.get_messages(
            gi=gi,
            merge_contiguous=merge_contiguous
        )

        api = None
        if gi.api_key is not None:
            api = {
                "openai": [OpenAI(api_key=gi.api_key), AsyncOpenAI(api_key=gi.api_key)][use_async],
                "anthropic": [Anthropic(api_key=gi.api_key), AsyncAnthropic(api_key=gi.api_key)][use_async],
                "mistralai": Mistral(api_key=gi.api_key),
                "together": [Together(api_key=gi.api_key), AsyncTogether(api_key=gi.api_key)][use_async],
                "url": gi.api_url,
                "hf": None
            }[gi._format]
        else:
            api = {
                "openai": [openai_api, openai_async_api][use_async],
                "anthropic": [anthropic_api, anthropic_async_api][use_async],
                "mistralai": mistralai_api,
                "together": [together_api, together_async_api][use_async],
                "url": gi.api_url,
                "hf": None
            }[gi._format]

        complete = {
            "hf": self.__chat_complete_hf,
        }
        if not use_async:
            complete.update({
                "openai": self.__chat_complete_openai,
                "anthropic": self.__chat_complete_anthropic,
                "mistralai": self.__chat_complete_mistralai,
                "together": self.__chat_complete_together,
                "url": self.__chat_complete_url
            })
        else:
            complete.update({
                "openai": self.__chat_complete_openai_async,
                "anthropic": self.__chat_complete_anthropic_async,
                "mistralai": self.__chat_complete_mistralai_async,
                "together": self.__chat_complete_together_async,
                "url": self.__chat_complete_url_async
            })

        complete = complete[gi._format]
    
        child = None
        content = ""
        while retry:
            try:
                if use_async and gi._format != "hf":
                    content = await complete(gi, messages, api, force_prepend)
                else:
                    content = complete(gi, messages, api, force_prepend)

                if crash_on_empty_response and content.strip() == "":
                    raise Exception(f"No content returned by the model: {content}. The request was: {messages} and the model used was: {gi.model}")

                to_prepend = force_prepend or ""
                if not content.startswith(to_prepend):
                    content = to_prepend + content
                
                if parse_json:
                    # Prevent back_off if parsing is failing
                    back_off = False
                    parsed_content = extract_json_from_completion(content)

                    if parsed_content in ['""', "{}", ''] and crash_on_refusal:
                        raise Exception(f"No JSON found in the response: {content}. The request was: {messages} and the model used was: {gi.model}")
                    back_off = True
                    content = parsed_content
                child = ChatNode(content=content, role="assistant")
                break
            except Exception as e:
                print(f"{Fore.RED}Error: {Fore.RESET}{e}")
                print("AI content: ", content)
                if retry <= 1:
                    raise e
                retry -= 1
                if back_off:
                    time.sleep(back_off_time)
                    if exp_back_off:
                        back_off_time = min(back_off_time * 2, max_back_off)
                else:
                    back_off = True
        
        if add_child:
            self.add_child(child)
        return child

    def complete_one(self,
        completion_params: NodeCompletionParameters | GeneratorInfo,
    ) -> ChatNode:
        return asyncio.run(self.general_complete_one(completion_params, use_async=False))

    def complete(self,
        completion_params: NodeCompletionParameters | GeneratorInfo
    ) -> ChatNode | List[ChatNode]:
        if isinstance(completion_params, GeneratorInfo):
            return self.complete(NodeCompletionParameters(gi=completion_params))

        children = [
            self.complete_one(completion_params) 
            for _ in range(
                completion_params.generation_parameters.n 
                if completion_params.generation_parameters is not None 
                else completion_params.gi.completion_parameters.n
            )
        ]

        return children if len(children) > 1 else children[0]
    
    async def complete_one_async(self,
        completion_params: NodeCompletionParameters | GeneratorInfo
    ) -> ChatNode:
        return await self.general_complete_one(
            completion_params=completion_params,
            use_async=True
        )
    
    async def complete_async(self,
        completion_params: NodeCompletionParameters | GeneratorInfo
    ) -> ChatNode | List[ChatNode]:
        if isinstance(completion_params, GeneratorInfo):
            return await self.complete_async(NodeCompletionParameters(gi=completion_params))

        tasks = [
            self.complete_one_async(completion_params)
            for _ in range(
                completion_params.generation_parameters.n 
                if completion_params.generation_parameters is not None 
                else completion_params.gi.completion_parameters.n
            )
        ]

        children = await asyncio.gather(*tasks)

        return children if len(children) > 1 else children[0]
    
    if HUGGINGFACE_ACTIVATED:
        @torch.no_grad()
        def hf_compute_logits(self, 
            gi: GeneratorInfo
        ) -> torch.Tensor:
            assert gi._format == "hf", "hf_compute_logits should only be called for hf format"
            
            message_len = gi.hf_tokenizer(self.content, return_tensors="pt", padding=True).input_ids.shape[1] - 1

            messages = self.get_messages(gi=gi)
            input_ids = self.hf_process_messages(gi=gi, messages=messages, padding=True).input_ids
            
            outputs = gi.hf_auto_model(input_ids=input_ids)
            probs = torch.log_softmax(outputs.logits, dim=-1).detach()

            # collect the probability of the generated token -- probability at index 0 corresponds to the token at index 1
            probs = probs[:, :-1, :]
            input_ids = input_ids[:, 1:]
            gen_probs = torch.gather(probs, 2, input_ids[:, :, None]).squeeze(-1)

            return gen_probs[:, -message_len:].cpu().detach()

        @torch.no_grad()
        def hf_compute_logits_average(self, 
            gi: GeneratorInfo, 
            quantile: float = 1, 
            repeat_penalty: bool = True, 
            repeat_treshold: int = 8, 
            n_start_penalize_repeat: int = 3
        ) -> torch.Tensor:
            logits = self.hf_compute_logits(gi=gi)[0]
            prompt_tks = gi.hf_tokenizer(self.content, return_tensors="pt").input_ids[0, 1:]

            if repeat_penalty:
                last_token_values = {}
                for i, (token, logit) in enumerate(zip(prompt_tks, logits)):
                    token = token.item()
                    if token not in last_token_values:
                        last_token_values[token] = (i, logit, 0)
                    else:
                        last_i, worst_logit, n_repeat = last_token_values[token]
                        if last_i + repeat_treshold >= i:
                            if n_repeat >= n_start_penalize_repeat:
                                logits[i] = worst_logit
                            last_token_values[token] = (i, min(worst_logit, logit), n_repeat + 1)
                        else:
                            last_token_values[token] = (i, logit, 0)

            logits_quantile = torch.topk(logits, k=int(quantile * logits.shape[-1]), largest=False, dim=-1).values

            return (torch.sum(logits_quantile, dim=-1) / logits_quantile.shape[-1]).item()
    else:
        def hf_compute_logits(self, **kwargs):
            raise NotImplementedError("hf_compute_logits should only be called for hf format, and if HUGGINGFACE_ACTIVATED is True (transformers and torch must installed)")

        def hf_compute_logits_average(self, **kwargs):
            raise NotImplementedError("hf_compute_logits_average should only be called for hf format, and if HUGGINGFACE_ACTIVATED is True (transformers and torch must installed)")
    
    @classmethod
    def from_thread(cls, 
        path: str | List[str]
    ) -> ChatNode:
        """Load a thread from a JSON file or multiple JSON files.
        
        Args:
            path: Path to a JSON file or list of paths to JSON files.
                 If a list is provided, the threads will be merged in order.
        
        Returns:
            The root node of the loaded thread
        """
        if isinstance(path, list):
            if not path:
                raise ValueError("Empty list of paths provided")
            
            # Load the first thread
            parent = cls.from_thread(path[0])
            
            # Merge subsequent threads
            for thread_path in path[1:]:
                parent = parent.merge(cls.from_thread(thread_path))
            
            return parent
        
        # Single path logic
        data = json.load(open(path, "r"))
        if "required_kwargs" not in data:
            data["required_kwargs"] = {}

        prompts = data["prompts"]
        parent = None
        for prompt in prompts:
            assert (
                "role" in prompt and "content" in prompt
            ), "Each prompt must have a role and a content"
            assert prompt["role"] in [
                "user",
                "assistant",
                "system",
            ], "role must be one of 'user', 'assistant', 'system'"

            current_node = ChatNode(
                content=prompt["content"],
                role=prompt["role"],
                format_kwargs=data["required_kwargs"],
            )

            if parent is None:
                parent = current_node
            else:
                parent = parent.add_child(current_node)

        return parent
    
    # TODO: Add a "from_loom" method to load complex graph