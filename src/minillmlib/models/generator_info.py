"""Generator information and model configurations for MiniLLMLib."""
from __future__ import annotations

from dataclasses import MISSING, dataclass, field, fields
from typing import Any, Dict, Literal, Optional, Tuple
from copy import deepcopy
from pymongo.collection import Collection

HUGGINGFACE_ACTIVATED = False
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
    HUGGINGFACE_ACTIVATED = True
except:
    torch = None
    pass

@dataclass(kw_only=True)
class GeneratorCompletionParameters:
    # Text
    temperature: float = 0.8
    max_tokens: int = 512
    
    # Audio
    voice: Literal['alloy', 'ash', 'ballad', 'coral', 'echo', 'sage', 'shimmer'] = "alloy"
    audio_output_folder: Optional[str] = None

    kwargs: Dict[str, Any] = field(default_factory=dict)

    def __init__(self, **kwargs):
        """Initialize with arbitrary kwargs."""
        known_fields = {
            f.name for f in fields(self) if f.name != 'kwargs'
        }
        field_values = {
            k: v for k, v in kwargs.items() if k in known_fields
        }
        other_kwargs = {
            k: v for k, v in kwargs.items() if k not in known_fields
        }
        
        # Set known fields
        for field_name, field_value in field_values.items():
            setattr(self, field_name, field_value)
        
        # Set default values for unspecified fields
        for field in fields(self):
            if field.name not in field_values and field.name != 'kwargs':
                setattr(
                    self, field.name, 
                    field.default_factory() 
                    if field.default_factory != MISSING 
                    else field.default
                )
        
        # Store remaining kwargs
        self.kwargs = other_kwargs

@dataclass
class GeneratorInfo:
    """Configuration class for different LLM generators."""
    # Basic information
    model: Optional[str]
    is_chat: bool = True
    
    # Generation Parameters
    completion_parameters: GeneratorCompletionParameters = field(default_factory=GeneratorCompletionParameters)
    
    # API
    api_url: Optional[str] = None
    api_key: Optional[str] = None
    _format: Literal["openai", "openai-audio", "anthropic", "url", "mistralai", "hf", "prettify"] = "url"
    force_merge: bool = False
    enforce_json_compatible_prompt: bool = False
    no_system: bool = False
    deactivate_temperature: bool = False
    deactivate_max_tokens: bool = False
    
    # Additional information
    price_table: Tuple[float, float] = (0.0, 0.0)
    is_uncensored: bool = False
    translation_table: Dict[str, str] = field(default_factory=dict) # Used to translate the role titles sent to the chatbot. See pretty_messages
    
    # Usage tracking related
    usage_tracking_type: str | None = None
    usage_db: Collection | None = None
    usage_id_key: str | None = None
    usage_id_value: str | None = None
    usage_key: str | None = None

    # HuggingFace
    hf_model_kwargs: Dict[str, Any] = field(default_factory=dict)
    hf_process_kwargs: Dict[str, Any] = field(default_factory=dict)
    hf_auto_model = None
    hf_processor = None
    hf_tokenizer = None
    hf_device: str = "cuda:0"
    hf_expected_eoc: str = "<|end|>\n"

    def build_hf_model(self):
        """Build HuggingFace model, processor and tokenizer."""
        if not HUGGINGFACE_ACTIVATED:
            raise ValueError(f"HuggingFace not activated, you can't initialize the model {self.model} without installing transformers and torch")
        
        if self.hf_auto_model is None:
            self.hf_auto_model = AutoModelForCausalLM.from_pretrained(
                self.model, 
                torch_dtype="auto", 
                device_map="auto", 
                **self.hf_model_kwargs
            )
        if self.hf_processor is None:
            try:
                self.hf_processor = AutoProcessor.from_pretrained(
                    self.model, 
                    **self.hf_process_kwargs
                )
                if self.hf_processor.tokenizer.pad_token_id is None:
                    self.hf_processor.tokenizer.pad_token_id = self.hf_processor.tokenizer.eos_token_id
            except:
                self.hf_processor = None
        if self.hf_tokenizer is None:
            self.hf_tokenizer = AutoTokenizer.from_pretrained(self.model)
            if self.hf_tokenizer.pad_token_id is None:
                self.hf_tokenizer.pad_token_id = self.hf_tokenizer.eos_token_id

    def __eq__(self, 
        value: 'GeneratorInfo'
    ) -> bool:
        """Check equality between two GeneratorInfo instances."""
        return (
            self.model == value.model
            and self.is_chat == value.is_chat
            and self.api_url == value.api_url
            and self.api_key == value.api_key
            and self._format == value._format
            and self.no_system == value.no_system
            and self.translation_table == value.translation_table
            and self.deactivate_temperature == value.deactivate_temperature
            and self.deactivate_max_tokens == value.deactivate_max_tokens
            and self.force_merge == value.force_merge
            and self.price_table == value.price_table
            and self.is_uncensored == value.is_uncensored
        )

    def __hash__(self) -> int:
        """Generate hash for GeneratorInfo instance."""
        return hash((
            self.model,
            self.is_chat,
            self.additional_kwargs,
            self.temperature,
            self.max_tokens,
            self.api_url,
            self._format,
            self.no_system,
            self.force_merge,
            self.api_key,
            self.is_uncensored,
            self.deactivate_max_tokens,
            self.deactivate_temperature,
            self.enforce_json_compatible_prompt
        ))
    
    def deepcopy(self) -> GeneratorInfo:
        
        usage_db = self.usage_db
        self.usage_db = None
        
        new_instance = deepcopy(self)

        self.usage_db = usage_db
        new_instance.usage_db = usage_db
        
        return new_instance

pretty_messages = GeneratorInfo(
    model=None,
    _format="prettify",
    translation_table={
        "assistant": "\n\nASSISTANT: ",
        "user": "\n\nUSER: ",
        "system": "SYSTEM: ",
        "base": " ",
    },
)