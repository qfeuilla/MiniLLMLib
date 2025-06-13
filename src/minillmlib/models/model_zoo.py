"""
Model zoo for MiniLLMLib.
"""

import os

from dotenv import load_dotenv

from .generator_info import GeneratorCompletionParameters, GeneratorInfo

load_dotenv()

anthropic = {}
for model_name in [
    "claude-2.0",
    "claude-2.1",
    "claude-3-haiku-20240307",
    "claude-3-sonnet-20240229",
    "claude-3-opus-20240229",
    "claude-3-5-haiku-20241022",
    "claude-3-5-sonnet-20241022",
    "claude-3-5-sonnet-20240620",
    "claude-3-7-sonnet-20250219"
]:
    anthropic[model_name] = GeneratorInfo(
        model=model_name,
        _format="anthropic",
    )

openai = {}
for model_name in [
    "gpt-4o",
    "gpt-4o-2024-11-20",
    "chatgpt-4o-latest",
    "gpt-4o-mini",
    "gpt-4o-mini-2024-07-18",
    "o1",
    "o1-2024-12-17",
    "o1-mini",
    "o1-mini-2024-09-12",
    "o1-preview",
    "o1-preview-2024-09-12",
    "gpt-4-turbo",
    "gpt-4-turbo-2024-04-09",
    "gpt-4-turbo-preview",
    "gpt-4-0125-preview",
    "gpt-4",
    "gpt-4-0613",
    "gpt-4-0314",
    "gpt-3.5-turbo-0125",
    "gpt-3.5-turbo"
]:

    completion_params = {}
    if "o1" in model_name:
        completion_params["max_completion_tokens"] = 42000
    openai[model_name] = GeneratorInfo(
        model=model_name,
        _format="openai",
        deactivate_max_tokens="o1" in model_name,
        completion_parameters=GeneratorCompletionParameters(
            **completion_params
        )
    )

openai_audio = {
    "gpt-4o-audio-preview": GeneratorInfo(
        model="gpt-4o-audio-preview",
        _format="openai-audio",
        completion_parameters=GeneratorCompletionParameters(
            audio_output_folder="./audio/",
            voice="alloy",
        ),
        deactivate_max_tokens=True
    )
}

openrouter = {}
# Removing SambaNova because it cost a lot
for model_name, uri, providers, exclude_providers in [
    ("hermes-405b", "nousresearch/hermes-3-llama-3.1-405b", None, ["SambaNova"]),
    ("deepseek-v3", "deepseek/deepseek-chat", None, ["SambaNova"]),
    ("deepseek-v3-0324", "deepseek/deepseek-chat-v3-0324", None, ["SambaNova"]),
    ("mistral-7b-instruct", "mistralai/mistral-7b-instruct", None, ["SambaNova"]),
    ("claude-3.7-sonnet", "anthropic/claude-3.7-sonnet", None, ["SambaNova"])]:

    provider_settings = { "data_collection": "deny"}

    if providers is not None:
        provider_settings["order"] = providers
    else:
        provider_settings["sort"] = "throughput"

    if exclude_providers is not None:
        provider_settings["ignore"] = exclude_providers

    openrouter[model_name] = GeneratorInfo(
        model=uri,
        _format="url",
        api_url="https://openrouter.ai/api/v1/chat/completions",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        completion_parameters=GeneratorCompletionParameters(
            provider=provider_settings
        )
    )

bet_leaderboard_v1 = {}
for model_name, uri in [
    ("yi-large", "01-ai/yi-large"),
    ("grok-2-1212", "x-ai/grok-2-1212"),
    ("gpt-4o-mini-2024-07-18", "openai/gpt-4o-mini-2024-07-18"),
    ("gpt-4o-2024-11-20", "openai/gpt-4o-2024-11-20"),
    ("gpt-4o-2024-08-06", "openai/gpt-4o-2024-08-06"),
    ("pixtral-large-2411", "mistralai/pixtral-large-2411"),
    ("mixtral-8x22b-instruct", "mistralai/mixtral-8x22b-instruct"),
    ("mixtral-8x7b-instruct", "mistralai/mixtral-8x7b-instruct"),
    ("mistral-nemo", "mistralai/mistral-nemo"),
    ("ministral-8b", "mistralai/ministral-8b"),
    ("phi-4", "microsoft/phi-4"),
    ("llama-3.3-70b-instruct", "meta-llama/llama-3.3-70b-instruct"),
    ("llama-3.2-90b-vision-instruct", "meta-llama/llama-3.2-90b-vision-instruct"),
    ("llama-3.2-11b-vision-instruct", "meta-llama/llama-3.2-11b-vision-instruct"),
    ("llama-3.2-1b-instruct", "meta-llama/llama-3.2-1b-instruct"),
    ("llama-3.1-405b-instruct", "meta-llama/llama-3.1-405b-instruct"),
    ("llama-3.1-70b-instruct", "meta-llama/llama-3.1-70b-instruct"),
    ("llama-3.1-8b-instruct", "meta-llama/llama-3.1-8b-instruct"),
    ("gemma-2-27b-it", "google/gemma-2-27b-it"),
    ("gemini-pro-1.5", "google/gemini-pro-1.5"),
    ("deepseek-v3", "deepseek/deepseek-chat"),
    ("deepseek-r1", "deepseek/deepseek-r1"),
    ("claude-3.5-sonnet-20240620", "anthropic/claude-3.5-sonnet-20240620"),
    ("claude-3.5-sonnet-20241022", "anthropic/claude-3.5-sonnet"),
    ("claude-3.5-haiku-20241022", "anthropic/claude-3.5-haiku-20241022"),
    ("claude-3-sonnet", "anthropic/claude-3-sonnet"),
    ("claude-3-haiku", "anthropic/claude-3-haiku"),
    ("claude-3-opus", "anthropic/claude-3-opus"),
    ("nova-pro-v1", "amazon/nova-pro-v1"),
    ("nova-lite-v1", "amazon/nova-lite-v1"),
    ("nova-micro-v1", "amazon/nova-micro-v1"),
    ("qwen-plus", "qwen/qwen-plus"),
    ("qwen-max", "qwen/qwen-max"),
    ("qwen-2.5-72b-instruct", "qwen/qwen-2.5-72b-instruct"),
    ("qwen-2.5-7b-instruct", "qwen/qwen-2.5-7b-instruct")
]:
    bet_leaderboard_v1[model_name] = GeneratorInfo(
        model=uri,
        _format="url",
        api_url="https://openrouter.ai/api/v1/chat/completions",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        completion_parameters=GeneratorCompletionParameters(
            provider={
                "data_collection": "deny",
                "sort": "throughput"
            },
            max_tokens=512 if model_name != "deepseek/deepseek-r1" else 42000,
        )
    )

for model_name in ["o3-mini-2025-01-14", "o1-2024-12-17"]:
    bet_leaderboard_v1[model_name] = GeneratorInfo(
        model=uri,
        _format="openai",
        api_key=os.getenv("OPENAI_API_KEY"),
        completion_parameters=GeneratorCompletionParameters(
            max_completion_tokens=42000,
        ),
        deactivate_max_tokens=True
    )
