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
    openai[model_name] = GeneratorInfo(
        model=model_name,
        _format="openai",
        deactivate_default_params="o1" in model_name
    )

openrouter = {}
for model_name, uri in [
    ("hermes-405b", "nousresearch/hermes-3-llama-3.1-405b"),
    ("deepseek-v3", "deepseek/deepseek-chat")
]:
    openrouter[model_name] = GeneratorInfo(
        model=uri,
        _format="url",
        api_url=f"https://openrouter.ai/api/v1/chat/completions",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        completion_parameters=GeneratorCompletionParameters(
            kwargs={
                "provider": {
                    "data_collection": "deny",
                    "sort": "throughput"
                }
            }
        )
    )