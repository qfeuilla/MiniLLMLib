"""Core ChatNode implementation for MiniLLMLib."""
# pylint: disable=too-many-lines
# pylint: disable=protected-access

from __future__ import annotations

import asyncio
import copy
import json
import os
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

import httpx
import requests
from anthropic import Anthropic, AsyncAnthropic
from anthropic.types.message import Message
from mistralai import Mistral
from openai import AsyncOpenAI, OpenAI
from openai.types.chat import ChatCompletion

from ..models.generator_info import (HUGGINGFACE_ACTIVATED, GeneratorInfo,
                                     pretty_messages, torch)
from ..utils.json_utils import extract_json_from_completion, to_dict
from ..utils.logging_utils import get_logger
from ..utils.message_utils import (AudioData, NodeCompletionParameters,
                                   base64_to_wav, format_prompt, get_payload,
                                   hf_process_messages,
                                   merge_contiguous_messages,
                                   process_audio_for_completion,
                                   validate_json_response)

warnings.filterwarnings("ignore", message=".*verification is strongly advised.*")

logger = get_logger()


def _initialize_api(api_class, env_key: str, async_api_class=None):
    try:
        api_key = os.environ.get(env_key)
        sync_client = api_class(api_key=api_key) if api_key else None
        async_client = (
            async_api_class(api_key=api_key) if api_key and async_api_class else None
        )
        return sync_client, async_client
    except Exception as e: # pylint: disable=broad-except
        logger.warning(
            {
                "message": "Failed to initialize API",
                "api": api_class.__name__,
                "error": str(e),
            }
        )
        return None, None


# Initialize API clients
anthropic_api, anthropic_async_api = _initialize_api(
    Anthropic, "ANTHROPIC_API_KEY", AsyncAnthropic
)
openai_api, openai_async_api = _initialize_api(OpenAI, "OPENAI_API_KEY", AsyncOpenAI)
mistralai_api, _ = _initialize_api(Mistral, "MISTRAL_API_KEY")


# NOTE: This object can be used in two shapes:
# - A Thread (in case there is only one child per node) --> I use it mostly like this.
#   Loom will be for when we'll do the cyborg tool
# - A Loom (in case there are multiple children per node)
class ChatNode:
    """A node in a chat tree."""

    def __init__(self,
        content: Optional[str] = None,
        role: str = "user",
        audio_data: Optional[AudioData] = None,
        format_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """Initialize a ChatNode.

        Args:
            content: Optional textual content of the message
            role: Role of the message sender (user/assistant/system/base)
            audio_data: Optional audio data
            format_kwargs: Optional formatting arguments for the content
        """
        # NOTE (design choice): For now audio and content will be stored in separate nodes
        if (content is None) == (audio_data is None):
            raise ValueError(
                "content xor audio_data must be provided, audio and content "
                "must be sent in separate nodes in the correct order"
            )

        if role not in [
            "user",
            "assistant",
            "system",
            "base",
        ]:
            raise ValueError(
                "role must be one of 'user', 'assistant', 'system' or 'base'"
            )

        self.role = role
        self.content = content

        self.children: List[ChatNode] = []
        self._parent: Optional[ChatNode] = None

        self.metadata: Dict[str, Any] = {}

        # The self.format_kwargs are loaded only if they can be retrieved in the content.
        self.format_kwargs: Dict[str, str] = {
            k: v for k, v in (format_kwargs or {}).items() if f"{{{k}}}" in self.content
        }

        self.audio_data: Optional[AudioData] = audio_data

    def is_root(self) -> bool:
        """Check if this node is the root of the tree."""
        return self._parent is None

    def get_root(self) -> ChatNode:
        """Get the root node of the tree."""
        return self if self.is_root() else self._parent.get_root()

    def add_child(self, child: ChatNode, illegitimate: bool = False) -> ChatNode:
        """Add a child node to this node.

        Args:
            child: The child node to add
            illegitimate: If True, only the child knows it's parent but
                          the parent doesn't get the child added
        """
        child._parent = self

        if not illegitimate:
            self.children.append(child)

            # For Thread mode, save the format_kwargs of the child to the root of the tree
            # (only the root's format_kwargs are saved in thread mode)
            root = self.get_root()
            for key in child.format_kwargs:
                if key not in root.format_kwargs:
                    root.format_kwargs[key] = child.format_kwargs[key]
                else:
                    logger.warning(
                        {
                            "message": "Format kwargs key collision",
                            "key": key,
                            "detail": (
                                "Key already exists in the root of the tree. "
                                "After loading, only one will be used."
                            ),
                        }
                    )

        return child

    def detach(self) -> ChatNode:
        """Detach this node from its parent."""
        self._parent.children.remove(self)
        self._parent = None
        return self

    def get_messages(self,
        gi: GeneratorInfo = pretty_messages,
        merge_contiguous: Optional[str] = "all",
    ) -> List[Dict[str, Any]] | str:
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

        messages: List[Dict[str, str]] = []
        current = self
        while current is not None:
            if gi.is_chat and current.role == "base":
                raise ValueError("The role 'base' is not allowed in chat models")

            content = None
            if current.content is not None:
                try:
                    content = format_prompt(current.content, **current.format_kwargs)
                except Exception as e: # pylint: disable=broad-except
                    logger.debug(
                        {
                            "message": "Error formatting content",
                            "content": current.content,
                            "error": str(e),
                            "detail": "Formatting skipped",
                        }
                    )
                    content = current.content
            else:
                # Check if audio paths are valid
                for audio_path in current.audio_data.audio_paths:
                    if not os.path.exists(audio_path) and current.role != "assistant":
                        # It's ok if the audio paths are not found for assistant messages
                        raise FileNotFoundError(f"Audio file not found: {audio_path}")

            messages.append(
                {
                    "role": current.role,
                    "content": content,
                    "audio_data": current.audio_data,
                }
            )
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

        # Choose the correct formatting depending on the gi _format
        # Opt 1: Remove the audio from the messages
        if gi._format in ["openai", "anthropic", "url", "mistralai", "hf"]:
            new_messages = []
            for message in messages:
                content = ""

                if message["audio_data"] is not None:
                    for _id, data in message["audio_data"].audio_ids.items():
                        # If there is a transcript, use it for the content
                        if data["transcript"] is not None:
                            content += "\n" + data["transcript"]
                        else:
                            content += (
                                f"\n*The {message['role']} attempted to provide an audio here, "
                                "but there is not transcription available*"
                            )

                    # TODO: Maybe use a transcription tool on the file
                    if (
                        message["role"] != "assistant"
                        and len(message["audio_data"].audio_paths) > 0
                    ):
                        content += (
                            f"\n*The {message['role']} provided "
                            f"{len(message['audio_data'].audio_paths)} audio, "
                            "but there is not transcriptions available*"
                        )
                else:
                    content = message["content"]

                new_messages.append({"role": message["role"], "content": content})

            messages = new_messages

        # Opt 2: Process text and audio for openai
        elif gi._format == "openai-audio":
            new_messages = []
            for message in messages:
                if message["audio_data"] is not None:
                    if message["role"] == "assistant":
                        for _id, data in message["audio_data"].audio_ids.items():
                            # If the id is not expired, use it, else use the transcript
                            if data["expires_at"] > time.time():
                                new_messages.append(
                                    {"role": "assistant", "audio": {"id": _id}}
                                )
                            else:
                                new_messages.append(
                                    {
                                        "role": "assistant",
                                        "content": "\n[Audio transcription]\n"
                                        + data["transcript"],
                                    }
                                )
                    else:
                        audio_data = process_audio_for_completion(
                            file_paths=message["audio_data"].audio_paths
                        )
                        for chunk in audio_data["chunks"]:
                            new_messages.append(
                                {
                                    "role": message["role"],
                                    "content": [
                                        {
                                            "type": "input_audio",
                                            "input_audio": {
                                                "data": chunk,
                                                "format": "wav",
                                            },
                                        }
                                    ],
                                }
                            )
                else:
                    new_messages.append(
                        {"role": message["role"], "content": message["content"]}
                    )

            messages = new_messages

        # Opt 3: Use the translation table
        elif gi._format in ["prettify"]:
            # Check that the translation table is complete
            if not all(
                role in gi.translation_table
                for role in ["user", "assistant", "system", "base"]
            ):
                raise ValueError(
                    "The translation table must be complete for the model "
                    f"{gi._format}, "
                    "with all the keys ['user', 'assistant', 'system', 'base']"
                )

            messages = "".join(
                f"{gi.translation_table[msg['role']]}{msg['content']}"
                for msg in messages
            )
        else:
            raise NotImplementedError(
                f"{gi._format} not supported. It must be one of "
                "['openai', 'anthropic', 'url', 'mistralai', 'hf', 'prettify']"
            )

        # NOTE: This makes sure that the content is a string that can be JSON decoded without issues
        #       (useful in case the LLM API url is not well coded, but it could affect the prompt
        #       so it should not be on by default)

        if gi.enforce_json_compatible_prompt:
            if gi._format == "openai-audio":
                raise ValueError(
                    "enforce_json_compatible_prompt is not supported for openai-audio"
                )
            messages = [
                {
                    "role": message["role"],
                    "content": json.dumps({"prompt": message["content"]})[12:-2],
                }
                for message in messages
            ]

        return messages

    def get_child(self, _map: List[int] = None) -> ChatNode:
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

    def merge(self, other: ChatNode) -> ChatNode:
        """Merge the root of the other tree to the current node."""
        self.add_child(other.get_root())
        return other

    def update_format_kwargs(self, propagate: bool = True, **kwargs) -> None:
        """
        Update the format_kwargs in the current node and 
        in all the parents until the root if propagate is True.
        """
        if self._parent is not None and propagate:
            self._parent.update_format_kwargs(propagate=propagate, **kwargs)

        for k, v in kwargs.items():
            # Only update the parameter if it's in the format_kwargs
            if k in self.format_kwargs:
                self.format_kwargs[k] = v

    def save_thread(self, path: str) -> None:
        """Save the node to a thread file."""
        node_list = []
        node = self
        while node:
            node_list.append(node)
            node = node._parent
        node_list.reverse()

        json.dump(
            {
                "required_kwargs": self.get_root().format_kwargs,
                "prompts": [
                    {
                        "role": node.role,
                        "content": node.content,
                        "audio_data": (
                            {
                                "audio_paths": node.audio_data.audio_paths,
                                "audio_ids": node.audio_data.audio_ids,
                            }
                            if node.audio_data is not None
                            else None
                        ),
                    }
                    for node in node_list
                ],
            },
            open(path, "w+", encoding="utf-8"),
            indent=4,
        )

    def save_loom(self, path: str) -> None:
        """Save the node to a loom file."""
        json.dump(to_dict(self.get_root()), open(path, "w+", encoding="utf-8"), indent=4)

    def _prepare_completion(self,
        completion_params: NodeCompletionParameters | GeneratorInfo,
        use_async: bool = False,
    ) -> Tuple[
        GeneratorInfo, Dict[str, Any], List[Dict[str, str]], Any, Optional[ChatNode]
    ]:
        """
        Prepare parameters for completion - shared code between sync and async implementations.
        Returns: (gi, params_dict, messages, api, complete_from)
        """
        if isinstance(completion_params, GeneratorInfo):
            completion_params = NodeCompletionParameters(gi=completion_params)

        # Extract parameters
        gi = completion_params.gi
        generation_parameters = completion_params.generation_parameters
        force_prepend = completion_params.force_prepend
        merge_contiguous = completion_params.merge_contiguous

        # Validation
        if not gi.is_chat:
            raise NotImplementedError("Non chat completion is not supported for now")

        if gi._format not in [
            "openai",
            "openai-audio",
            "anthropic",
            "url",
            "mistralai",
            "hf",
        ]:
            raise NotImplementedError(f"{gi._format} not supported for chat completion")

        # deepcopy the gi to avoid modifying the original
        # TODO: This might not be the best idea for local LLMs,
        #       I need to find a way to handle them properly without having to clone the weights
        gi = gi.deepcopy()

        # Apply generation parameters if provided
        # NOTE (design choice): the kwargs here are added to the ones in the
        #      gi.completion_parameters, and overwrite them if they are already set
        if generation_parameters is not None:
            for k, v in gi.completion_parameters.kwargs.items():
                if k not in generation_parameters.kwargs:
                    generation_parameters.kwargs[k] = v

            gi.completion_parameters = generation_parameters

        # Set up the complete_from node
        complete_from = self
        if force_prepend is not None:
            complete_from = self.add_child(
                ChatNode(content=force_prepend, role="assistant"), illegitimate=True
            )

        # Get messages
        messages = complete_from.get_messages(gi=gi, merge_contiguous=merge_contiguous)

        # Get the appropriate API client
        api = None
        if gi.api_key is not None:
            api = {
                "openai": [OpenAI(api_key=gi.api_key), AsyncOpenAI(api_key=gi.api_key)][
                    use_async
                ],
                "openai-audio": [
                    OpenAI(api_key=gi.api_key),
                    AsyncOpenAI(api_key=gi.api_key),
                ][use_async],
                "anthropic": [
                    Anthropic(api_key=gi.api_key),
                    AsyncAnthropic(api_key=gi.api_key),
                ][use_async],
                "mistralai": Mistral(api_key=gi.api_key),
                "url": gi.api_url,
                "hf": None,
            }[gi._format]
        else:
            api = {
                "openai": [openai_api, openai_async_api][use_async],
                "openai-audio": [openai_api, openai_async_api][use_async],
                "anthropic": [anthropic_api, anthropic_async_api][use_async],
                "mistralai": mistralai_api,
                "url": gi.api_url,
                "hf": None,
            }[gi._format]

        # Put all parameters in a dict to return (except the ones handled differently)
        params_dict = {
            "add_child": completion_params.add_child,
            "parse_json": completion_params.parse_json,
            "crash_on_refusal": completion_params.crash_on_refusal,
            "crash_on_empty_response": completion_params.crash_on_empty_response,
            "retry": max(completion_params.retry + 1, 1),
            "exp_back_off": completion_params.exp_back_off,
            "back_off_time": completion_params.back_off_time,
            "max_back_off": completion_params.max_back_off,
            "force_prepend": force_prepend,
        }

        return gi, params_dict, messages, api

    def _process_completion_result(
        self,
        content: str | AudioData,
        params: Dict[str, Any],
        messages: List[Dict[str, Any]],
        gi: GeneratorInfo,
        retry_state: Dict[str, Any],
    ) -> ChatNode:
        """
        Process the completion result - shared code between sync and async implementations.
        Modifies retry_state to control backoff behavior.
        """
        force_prepend = params["force_prepend"]
        parse_json = params["parse_json"]
        crash_on_refusal = params["crash_on_refusal"]
        crash_on_empty_response = params["crash_on_empty_response"]
        add_child = params["add_child"]

        clean_messages_for_debug = [
            (
                message
                if "audio_data" not in message or message["audio_data"] is None
                else {"role": message["role"], "content": "|some audio|"}
            )
            for message in messages
        ]

        if crash_on_empty_response and (
            (isinstance(content, str) and content.strip() == "")
            or (isinstance(content, AudioData) and len(content.audio_ids) == 0)
        ):

            raise Exception(  # pylint: disable=broad-exception-raised
                f"No content returned by the model. "
                f"The request was: {clean_messages_for_debug} "
                f"and the model used was: {gi.model}"
            )

        if isinstance(content, str):
            to_prepend = force_prepend.format() if force_prepend is not None else ""
            if not content.startswith(to_prepend):
                content = to_prepend + content

            if parse_json:
                # Prevent back_off if parsing is failing
                retry_state["back_off"] = False

                if crash_on_refusal and ("{" not in content and "[" not in content):
                    raise Exception(  # pylint: disable=broad-exception-raised
                        f"No JSON found in the response: {content}. "
                        f"The request was: {clean_messages_for_debug} "
                        f"and the model used was: {gi.model}"
                    )

                parsed_content = extract_json_from_completion(content)
                if parsed_content in ['""', "{}", ""] and crash_on_refusal:
                    raise Exception(  # pylint: disable=broad-exception-raised
                        f"No JSON found in the response: {content}. "
                        f"The request was: {clean_messages_for_debug} "
                        f"and the model used was: {gi.model}"
                    )

                # Re-enable back_off after successful parsing
                retry_state["back_off"] = True
                content = parsed_content

            child = ChatNode(content=content, role="assistant")
        elif isinstance(content, AudioData):
            to_prepend = force_prepend.format() if force_prepend is not None else ""

            child = ChatNode(audio_data=content, role="assistant")

            if len(to_prepend):
                prepend = ChatNode(content=to_prepend, role="assistant")
                prepend.add_child(child)
        else:
            raise ValueError(f"Unsupported content type: {type(content)}")

        if add_child:
            self.add_child(child)

        return child

    def complete_one(self,
        completion_params: NodeCompletionParameters | GeneratorInfo,
    ) -> ChatNode:
        """
        Synchronous version that uses the correct completion method based on the model format.
        This is entirely synchronous and avoids any async/event loop operations.
        """
        # Prepare all the parameters
        gi, params, messages, api = self._prepare_completion(
            completion_params, use_async=False
        )

        # Get the appropriate synchronous completion method
        complete_method = {
            "openai": self.__chat_complete_openai,
            "openai-audio": self.__chat_complete_openai_audio,
            "anthropic": self.__chat_complete_anthropic,
            "mistralai": self.__chat_complete_mistralai,
            "url": self.__chat_complete_url,
            "hf": self.__chat_complete_hf,
        }[gi._format]

        # Loop for retries
        retry = params["retry"]
        back_off_time = params["back_off_time"]

        content = ""
        retry_state = {"back_off": True}
        while retry:
            try:
                content = complete_method(gi, messages, api, params["force_prepend"])
                # Process and return the result
                child = self._process_completion_result(
                    content, params, messages, gi, retry_state
                )
                return child
            except Exception as e: # pylint: disable=broad-exception-caught
                logger.debug(
                    {
                        "error_type": e.__class__.__name__,
                        "error_message": str(e),
                        "error_args": getattr(e, "args", []),
                        "content": content,
                    }
                )

                if retry <= 1:
                    raise e
                retry -= 1
                if retry_state["back_off"]:
                    time.sleep(back_off_time)
                    if params["exp_back_off"]:
                        back_off_time = min(back_off_time * 2, params["max_back_off"])
                else:
                    retry_state["back_off"] = True

        # This should never happen because we always return or raise in the loop
        raise Exception("Unexpected error in complete_one") # pylint: disable=broad-exception-raised

    async def complete_one_async(self,
        completion_params: NodeCompletionParameters | GeneratorInfo
    ) -> ChatNode:
        """
        Fully asynchronous version that uses the correct async completion method.
        """
        # Prepare all the parameters
        gi, params, messages, api = self._prepare_completion(
            completion_params, use_async=True
        )

        # Check for HuggingFace - which doesn't support async
        if gi._format == "hf":
            raise NotImplementedError(
                "Async completion is not supported for HuggingFace models. "
                "Please use the synchronous 'complete_one' method instead."
            )

        # Get the appropriate async completion method
        complete_method = {
            "openai": self.__chat_complete_openai_async,
            "openai-audio": self.__chat_complete_openai_audio_async,
            "anthropic": self.__chat_complete_anthropic_async,
            "mistralai": self.__chat_complete_mistralai_async,
            "url": self.__chat_complete_url_async,
        }[gi._format]

        # Loop for retries
        retry = params["retry"]
        back_off_time = params["back_off_time"]

        content = ""
        retry_state = {"back_off": True}
        while retry:
            try:
                content = await complete_method(
                    gi, messages, api, params["force_prepend"]
                )
                # Process and return the result
                child = self._process_completion_result(
                    content, params, messages, gi, retry_state
                )
                return child
            except Exception as e: # pylint: disable=broad-except
                logger.debug(
                    {
                        "error_type": e.__class__.__name__,
                        "error_message": str(e),
                        "error_args": getattr(e, "args", []),
                        "content": content,
                    }
                )

                if retry <= 1:
                    raise e
                retry -= 1
                if retry_state["back_off"]:
                    await asyncio.sleep(back_off_time)
                    if params["exp_back_off"]:
                        back_off_time = min(back_off_time * 2, params["max_back_off"])
                else:
                    retry_state["back_off"] = True

        # This should never happen because we always return or raise in the loop
        raise Exception("Unexpected error in complete_one_async") # pylint: disable=broad-exception-raised

    def complete(self,
        completion_params: NodeCompletionParameters | GeneratorInfo
    ) -> ChatNode | List[ChatNode]:
        """Synchronous version of complete."""
        if isinstance(completion_params, GeneratorInfo):
            return self.complete(NodeCompletionParameters(gi=completion_params))

        # Generate n completions using the synchronous complete_one method
        children = [
            self.complete_one(completion_params) for _ in range(completion_params.n)
        ]

        return children if len(children) > 1 else children[0]

    async def complete_async(self,
        completion_params: NodeCompletionParameters | GeneratorInfo
    ) -> ChatNode | List[ChatNode]:
        """Asynchronous version of complete."""
        if isinstance(completion_params, GeneratorInfo):
            return await self.complete_async(
                NodeCompletionParameters(gi=completion_params)
            )

        # Generate n completions using the async complete_one_async method
        tasks = [
            self.complete_one_async(completion_params)
            for _ in range(completion_params.n)
        ]

        children = await asyncio.gather(*tasks)
        return children if len(children) > 1 else children[0]

    def handle_cost(self, gi: GeneratorInfo, cost: float):
        """Handle cost tracking for openrouter."""
        if gi.usage_tracking_type == "openrouter" and None not in [
            gi.usage_db,
            gi.usage_id_key,
            gi.usage_id_value,
            gi.usage_key,
        ]:
            gi.usage_db.update_one(
                {gi.usage_id_key: gi.usage_id_value},
                {"$inc": {gi.usage_key: cost}},
            )

    def __chat_complete_openai(self,
        gi: GeneratorInfo, messages: List[Dict[str, str]], api: OpenAI, *_1, **_2
    ) -> str:
        response: ChatCompletion = api.chat.completions.create(
            **get_payload(gi, messages)
        )
        return response.choices[0].message.content

    async def __chat_complete_openai_async(self,
        gi: GeneratorInfo,
        messages: List[Dict[str, str]],
        api: AsyncOpenAI,
        *_1,
        **_2,
    ) -> str:
        response: ChatCompletion = await api.chat.completions.create(
            **get_payload(gi, messages)
        )
        return response.choices[0].message.content

    def __chat_complete_openai_audio(self,
        gi: GeneratorInfo, messages: List[Dict[str, Any]], api: OpenAI, *_1, **_2
    ) -> AudioData | str:
        response: ChatCompletion = api.chat.completions.create(
            **get_payload(gi, messages),
            modalities=["text", "audio"],
            audio={"format": "pcm16", "voice": gi.completion_parameters.voice},
        )
        if response.choices[0].message.audio is None:
            return response.choices[0].message.content
        else:
            audio = response.choices[0].message.audio
            # First process the audio data and generate a temp file
            audio_file = base64_to_wav(
                base64_data=audio.data,
                output_folder=gi.completion_parameters.audio_output_folder,
            )

            return AudioData(
                audio_paths=[audio_file],
                audio_ids={
                    audio.id: {
                        "transcript": audio.transcript,
                        "expires_at": audio.expires_at,
                    }
                },
                audio_raw=audio.data,
            )

    async def __chat_complete_openai_audio_async(self,
        gi: GeneratorInfo,
        messages: List[Dict[str, Any]],
        api: AsyncOpenAI,
        *_1,
        **_2,
    ) -> AudioData | str:
        response: ChatCompletion = await api.chat.completions.create(
            **get_payload(gi, messages),
            modalities=["text", "audio"],
            audio={"format": "pcm16", "voice": gi.completion_parameters.voice},
        )
        if response.choices[0].message.audio is None:
            return response.choices[0].message.content
        else:
            audio = response.choices[0].message.audio
            # First process the audio data and generate a temp file
            audio_file = base64_to_wav(
                base64_data=audio.data,
                output_folder=gi.completion_parameters.audio_output_folder,
            )

            return AudioData(
                audio_paths=[audio_file],
                audio_ids={
                    audio.id: {
                        "transcript": audio.transcript,
                        "expires_at": audio.expires_at,
                    }
                },
                audio_raw=audio.data,
            )

    def __chat_complete_anthropic(self,
        gi: GeneratorInfo,
        messages: List[Dict[str, str]],
        api: Anthropic,
        *_1,
        **_2,
    ) -> str:
        system_prompt = ""
        while messages[0]["role"] == "system":
            system_prompt += messages[0]["content"] + "\n"
            messages = messages[1:]
        system_prompt = system_prompt[:-1]

        payload = get_payload(gi, messages)

        response: Message = api.messages.create(
            system=system_prompt if system_prompt else None, **payload
        )

        return response.content[0].text

    async def __chat_complete_anthropic_async(self,
        gi: GeneratorInfo,
        messages: List[Dict[str, str]],
        api: AsyncAnthropic,
        *_1,
        **_2,
    ) -> str:
        system_prompt = ""
        while messages[0]["role"] == "system":
            system_prompt += messages[0]["content"] + "\n"
            messages = messages[1:]
        system_prompt = system_prompt[:-1]

        payload = get_payload(gi, messages)

        response: Message = await api.messages.create(
            system=system_prompt if system_prompt else None, **payload
        )

        return response.content[0].text

    def __chat_complete_mistralai(self,
        gi: GeneratorInfo, messages: List[Dict[str, str]], api: Mistral, *_1, **_2
    ) -> str:
        response = api.chat.complete(**get_payload(gi, messages))
        return response.choices[0].message.content

    async def __chat_complete_mistralai_async(self,
        gi: GeneratorInfo, messages: List[Dict[str, str]], api: Mistral, *_1, **_2
    ) -> str:
        response = await api.chat.complete_async(**get_payload(gi, messages))
        return response.choices[0].message.content

    def __chat_complete_url(self,
        gi: GeneratorInfo, messages: List[Dict[str, str]], api_url: str, *_1, **_2
    ) -> str:

        headers = {"Authorization": f"Bearer {gi.api_key}"}
        response = requests.post(
            api_url, headers=headers, json={**get_payload(gi, messages)}, verify=False, timeout=300
        )
        response_json = response.json()

        if gi.usage_tracking_type is not None:
            self.handle_cost(gi, response_json["usage"]["cost"])

        return validate_json_response(response_json)

    async def __chat_complete_url_async(self,
        gi: GeneratorInfo, messages: List[Dict[str, str]], api_url: str, *_1, **_2
    ) -> str:

        timeout = httpx.Timeout(connect=10.0, read=300.0, write=10.0, pool=10.0)
        limits = httpx.Limits(
            max_keepalive_connections=20, max_connections=20, keepalive_expiry=30
        )

        headers = {"Authorization": f"Bearer {gi.api_key}"}

        response = None
        try:
            async with httpx.AsyncClient(
                verify=False, timeout=timeout, limits=limits
            ) as client:
                response = await client.post(
                    api_url,
                    headers=headers,
                    json={**get_payload(gi, messages)},
                )
                response_json = response.json()
        except httpx.ConnectTimeout as e:
            logger.debug("ConnectTimeout: %s", e)
            raise
        except httpx.ReadTimeout as e:
            logger.debug("ReadTimeout: %s", e)
            raise
        except httpx.ConnectError as e:
            logger.debug("ConnectError: %s", e)
            raise
        except httpx.ReadError as e:
            logger.debug("ReadError: %s", e)
            raise
        except Exception as e:
            error_details = {
                "status_code": response.status_code if response else None,
                "response_text": response.text if response else None,
                "response_headers": dict(response.headers) if response else None,
                "url": api_url,
            }
            logger.error(error_details)
            raise e

        if gi.usage_tracking_type is not None:
            self.handle_cost(gi, response_json["usage"]["cost"])

        return validate_json_response(response_json)

    def __chat_complete_hf(self,
        gi: GeneratorInfo,
        messages: List[Dict[str, str]],
        _1,
        force_prepend: str,
        *_2,
        **_3,
    ) -> str:
        if gi.hf_auto_model is None or gi.hf_processor is None:
            gi.build_hf_model()

        inputs = hf_process_messages(
            gi=gi, messages=messages, force_prepend=force_prepend
        )

        generate_ids = gi.hf_auto_model.generate(
            **inputs,
            max_new_tokens=gi.completion_parameters.max_tokens,
            eos_token_id=gi.hf_tokenizer.eos_token_id,
            temperature=gi.completion_parameters.temperature,
            **gi.completion_parameters.kwargs,
        )

        generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]

        decoded = gi.hf_tokenizer.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        if force_prepend is not None:
            decoded = " " + decoded

        return decoded

    if HUGGINGFACE_ACTIVATED:

        @torch.no_grad()
        def hf_compute_logits(self, gi: GeneratorInfo) -> torch.Tensor:
            """ Compute the logits for the current token. """
            assert (
                gi._format == "hf"
            ), "hf_compute_logits should only be called for hf format"

            message_len = (
                gi.hf_tokenizer(
                    self.content, return_tensors="pt", padding=True
                ).input_ids.shape[1]
                - 1
            )

            messages = self.get_messages(gi=gi)
            input_ids = hf_process_messages(
                gi=gi, messages=messages, padding=True
            ).input_ids

            outputs = gi.hf_auto_model(input_ids=input_ids)
            probs = torch.log_softmax(outputs.logits, dim=-1).detach()

            # collect the probability of the generated token
            # -- probability at index 0 corresponds to the token at index 1
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
            n_start_penalize_repeat: int = 3,
        ) -> torch.Tensor:
            """ Compute the average of the logits for the current token. """
            logits = self.hf_compute_logits(gi=gi)[0]
            prompt_tks = gi.hf_tokenizer(self.content, return_tensors="pt").input_ids[
                0, 1:
            ]

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
                            last_token_values[token] = (
                                i,
                                min(worst_logit, logit),
                                n_repeat + 1,
                            )
                        else:
                            last_token_values[token] = (i, logit, 0)

            logits_quantile = torch.topk(
                logits, k=int(quantile * logits.shape[-1]), largest=False, dim=-1
            ).values

            return (
                torch.sum(logits_quantile, dim=-1) / logits_quantile.shape[-1]
            ).item()

    else:

        def hf_compute_logits(self, gi: GeneratorInfo):
            """ Compute the logits for the current token. """
            raise NotImplementedError(
                "hf_compute_logits should only be called for hf format, and if "
                "HUGGINGFACE_ACTIVATED is True (transformers and torch must installed)"
            )

        def hf_compute_logits_average(self, gi: GeneratorInfo, **kwargs):
            """ Compute the average of the logits for the current token. """
            raise NotImplementedError(
                "hf_compute_logits_average should only be called for hf format, and if "
                "HUGGINGFACE_ACTIVATED is True (transformers and torch must installed)"
            )

    async def collapse_thread(self,
        keep_last_n: int,
        keep_n: int,
        gi: Optional[GeneratorInfo] = None,
    ) -> Tuple[ChatNode, Optional[ChatNode]]:
        """
        Collapse the thread, keeping the first (keep_n - keep_last_n) nodes, the last keep_last_n nodes,
        and summarizing or marking the collapsed part as needed. Returns the last node of the collapsed thread.
        """
        
        # 1. Collect the thread from root to self
        thread: list[ChatNode] = []
        node = self
        while node is not None:
            thread.append(node)
            node = node._parent
        thread.reverse()  # root to leaf (self)
        
        N = len(thread)

        if keep_n <= keep_last_n:
            keep_first = 0
            keep_last = min(keep_last_n, N)
        else:
            keep_first = min(keep_n - keep_last_n, N)
            keep_last = min(keep_last_n, N - keep_first)

        # Indices:
        # [0:keep_first] (kept)
        # [keep_first:N-keep_last] (collapsed)
        # [N-keep_last:N] (kept)
        first_nodes = thread[:keep_first]
        last_nodes = thread[N-keep_last:] if keep_last > 0 else []
        truncated_nodes = thread[keep_first:N-keep_last] if N-keep_last > keep_first else []

        # Deepcopy all nodes to avoid modifying originals
        first_nodes_cp = [copy.deepcopy(n) for n in first_nodes]
        last_nodes_cp = [copy.deepcopy(n) for n in last_nodes]

        # Build the new thread
        new_thread = []
        new_thread.extend(first_nodes_cp)

        # Insert summary or truncate marker if there are truncated nodes
        truncate_marker_node = None
        if truncated_nodes:
            if gi is None:
                # Insert a marker node
                truncate_marker_node = ChatNode(
                    content="""
==================== **truncated** ====================

[This part of the conversation was **truncated** for length.]

===================================================
""",
                    role="assistant"
                )
            else:
                # Summarize the truncated nodes using gi
                # Prepare the truncated part as a thread
                truncated_msgs = [
                    {"role": n.role, "content": n.content} for n in truncated_nodes
                ]
                truncated_thread = ChatNode.from_thread(messages=truncated_msgs)

                original_thread = ChatNode.from_thread(messages=self.get_messages(gi=gi))

                # Compose the summarization prompt as described by the user
                prompt_intro = (
                    "Conversation truncated. Here is a conversation between a user and an assistant.\n"
                    "==========\n"
                )
                prompt_outro = (
                    "\n==========\n\nCurrent end of the conversation. "
                    "Your task is to write a summary of the following part of the conversation. "
                    "The rest will keep existing, this part will be replaced by your summary. "
                    "Remember to keep the same language as the text you have to summarize. "
                    "Here is the part to summarize:\n==========\n"
                )
                prompt_json = (
                    "\n==========\nEnd of the part to summarize.\n\nNow answer in this JSON format: { 'brainstorming': '[short brainstorming about how to summarize efficiently]', 'summary': '[your summary]'}"
                )
                # Build the summarizer thread
                summarizer = ChatNode(prompt_intro, role="system")
                summarizer = summarizer.merge(original_thread)
                summarizer = summarizer.add_child(ChatNode(prompt_outro, role="assistant"))
                summarizer = summarizer.merge(truncated_thread)
                summarizer = summarizer.add_child(ChatNode(prompt_json, role="assistant"))

                summary_comp = await summarizer.complete_async(NodeCompletionParameters(
                    gi=gi,
                    parse_json=True
                ))
                summary_str = json.loads(summary_comp.content)["summary"]

                truncate_marker_node = ChatNode(
                    content="""
==================== **truncated** ====================

[This part of the conversation was **truncated** for length.]

-------------------- **summary** ----------------------

""" + summary_str + """

===================================================
""",
                    role="assistant"
                )
            new_thread.append(truncate_marker_node)

        new_thread.extend(last_nodes_cp)

        # Re-link the thread (parent/child)
        for i in range(1, len(new_thread)):
            new_thread[i-1].children = [new_thread[i]]
            new_thread[i]._parent = new_thread[i-1]

        if new_thread:
            new_thread[0]._parent = None
            new_thread[-1].children = []

        # Return the last node (deepcopy of self if it was kept, or the last node in last_nodes_cp)
        if last_nodes_cp:
            return last_nodes_cp[-1], truncate_marker_node
        elif new_thread:
            return new_thread[-1], truncate_marker_node
        else:
            # fallback: return a deepcopy of self
            return copy.deepcopy(self), truncate_marker_node

    @classmethod
    def from_thread(cls,
        path: str | List[str] | None = None,
        messages: List[Dict[str, str]] | None = None,
    ) -> ChatNode:
        """Load a thread from a JSON file or multiple JSON files, or directly from messages.

        Args:
            path: Path to a JSON file or list of paths to JSON files.
                 If a list is provided, the threads will be merged in order.
            messages: List of message dictionaries in the format:
                     [{"role": "user", "content": "message"}, ...].
                     This is compatible with the output of get_messages().

        Returns:
            The root node of the loaded thread

        Raises:
            ValueError: If both path and messages are None, or if path is an empty list
        """
        if path is None and messages is None:
            raise ValueError("Either path or messages must be provided")

        if messages is not None:
            parent = None
            for msg in messages:
                assert (
                    "role" in msg and "content" in msg
                ), "Each message must have a role and a content"
                assert msg["role"] in [
                    "user",
                    "assistant",
                    "system",
                ], "role must be one of 'user', 'assistant', 'system'"

                current_node = ChatNode(
                    content=msg["content"],
                    role=msg["role"],
                )

                if parent is None:
                    parent = current_node
                else:
                    parent = parent.add_child(current_node)
            return parent

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
        data = json.load(open(path, "r", encoding="utf-8"))
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
                audio_data=(
                    AudioData(
                        audio_paths=(
                            prompt["audio_data"]["audio_paths"]
                            if "audio_paths" in prompt["audio_data"]
                            else []
                        ),
                        audio_ids=(
                            prompt["audio_data"]["audio_ids"]
                            if "audio_ids" in prompt["audio_data"]
                            else {}
                        ),
                    )
                    if "audio_data" in prompt and prompt["audio_data"] is not None
                    else None
                ),
                format_kwargs=data["required_kwargs"],
            )

            if parent is None:
                parent = current_node
            else:
                parent = parent.add_child(current_node)

        return parent

    # TODO: Add a "from_loom" method to load complex graph
