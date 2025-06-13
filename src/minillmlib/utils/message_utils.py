"""Message processing utilities for MiniLLMLib."""
from __future__ import annotations

import base64
import io
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Union

from pydub import AudioSegment

from ..models.generator_info import (GeneratorCompletionParameters,
                                     GeneratorInfo)


@dataclass
class AudioData:
    """ Audio data container. """

    audio_paths: List[str] = field(default_factory=list)
    audio_ids: Dict[str, Dict[str, str | int]] = field(default_factory=dict)
    # '-> ids with their transcription and expiration time

    audio_raw: str = ""

    def merge(self,
        other: AudioData
    ) -> AudioData:
        """Merge two AudioData instances."""

        new_audio = AudioData()
        new_audio.audio_paths = self.audio_paths + other.audio_paths
        new_audio.audio_ids = self.audio_ids | other.audio_ids

        current_audio_raw = base64.b64decode(self.audio_raw) if self.audio_raw else b""
        other_audio_raw = base64.b64decode(other.audio_raw) if other.audio_raw else b""
        new_audio.audio_raw = base64.b64encode(current_audio_raw + other_audio_raw).decode('utf-8')

        return new_audio

# NOTE: This is separated because more features could be added to this like tool use
def format_prompt(
    prompt: str,
    **kwargs
) -> str:
    """Format a prompt with given kwargs."""
    return prompt.format(**kwargs)


def merge_contiguous_messages(
    messages: List[Dict[str, Any]],
    merge_contiguous: Optional[str] = None
) -> List[Dict[str, Any]]:
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
        # Validate message structure
        if not isinstance(message, dict):
            raise ValueError("Each message must be a dictionary")

        if "role" not in message:
            raise ValueError("Message must have a 'role' field")

        role = message["role"]
        if role not in ["system", "user", "assistant", "base"]:
            raise ValueError("Role must be one of 'system', 'user', 'assistant' or 'base'")

        content = message.get("content")
        audio_data = message.get("audio_data")

        if content is None and audio_data is None:
            raise ValueError("Message must have either 'content' or 'audio_data'")

        if content is not None and not isinstance(content, str):
            raise ValueError("Message content must be a string")

        if audio_data is not None and not isinstance(audio_data, AudioData):
            raise ValueError("Message audio_data must be an AudioData instance")

        # If the current message is a "system" but it is not at the beginning, then make it a user
        if role == "system":
            if system_ended:
                role = "user"
        elif not system_ended:
            system_ended = True

        # NOTE (design choice): It won't merge audio and text nodes
        if (role == merge_contiguous or merge_contiguous == "all") and \
            len(result) > 0 and previous_role == role:
            if (content is not None) and (result[-1]["content"] is not None):
                result[-1]["content"] += "\n" + content
            elif (result[-1]["audio_data"] is not None) and (audio_data is not None):
                result[-1]["audio_data"] = result[-1]["audio_data"].merge(audio_data)
            else:
                result.append({"role": role, "content": content, "audio_data": audio_data})
        else:
            result.append({"role": role, "content": content, "audio_data": audio_data})

        previous_role = role

    return result

def hf_process_messages(
    gi: GeneratorInfo,
    messages: List[Dict[str, str]],
    force_prepend: str | None = None,
    padding: bool = False
) -> str:
    """Process messages for a Hugging Face model."""

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
    except Exception: # pylint: disable=broad-except
        inputs = gi.hf_tokenizer(
            prompt,
            return_tensors="pt",
            padding=padding
        ).to(gi.hf_device)

    return inputs

def get_payload(gi: GeneratorInfo, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Get payload for a completion request."""
    payload = {
        "model": gi.model,
        "messages": messages,
        **gi.completion_parameters.kwargs
    }
    if not gi.deactivate_temperature:
        payload["temperature"] = gi.completion_parameters.temperature
    if not gi.deactivate_max_tokens:
        payload["max_tokens"] = gi.completion_parameters.max_tokens

    return payload

def process_audio_for_completion(
    file_paths: List[str],
    target_sample_rate: int = 24000,
    enforce_mono: bool = True,
    bit_depth: Literal[32, 16, 8] = 16,
    enable_chunking: bool = True,
    max_chunk_size: int = 10 * 1024 * 1024,  # 10 MiB in bytes
    enforce_format: Literal["wav", "mp3", None] = "wav"
) -> Dict[str, Union[List[str], float, int]]:
    """
    Process multiple audio files, merge them, and optionally chunk them for completion
    
    Args:
        file_paths: List of paths to audio files to process (any format supported by FFmpeg)
        target_sample_rate: Target sample rate for processed audio
        enforce_mono: Whether to convert all audio to mono
        bit_depth: Bit depth for encoding (8, 16, or 32)
        enable_chunking: Whether to split the audio into chunks
        max_chunk_size: Maximum size of each chunk in bytes (before base64 encoding)
        enforce_format: Whether to enforce a specific format for the output
        
    Returns:
        Dict containing base64-encoded chunks and metadata
    """
    if not file_paths:
        raise ValueError("No file paths provided")

    # Calculate bytes per sample based on bit depth
    bytes_per_sample = bit_depth // 8

    # Calculate frames per chunk if chunking is enabled
    frames_per_chunk = max_chunk_size // bytes_per_sample if enable_chunking else None

    # Initialize merged audio
    merged_audio = None

    # Process and merge audio files
    for file_path in file_paths:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        try:
            # Load audio file with pydub
            audio : AudioSegment = AudioSegment.from_file(file_path)

            # Convert to mono if requested
            if enforce_mono and audio.channels > 1:
                audio = audio.set_channels(1)

            # Resample to target rate
            audio = audio.set_frame_rate(target_sample_rate)

            # Set bit depth
            audio = audio.set_sample_width(bytes_per_sample)

            # Append to merged audio
            if merged_audio is None:
                merged_audio = audio
            else:
                merged_audio += audio

        except Exception as e:
            raise ValueError(f"Failed to process file {file_path}: {str(e)}") from e

    if merged_audio is None:
        raise ValueError("No audio data was successfully processed")

    # Get raw PCM data
    raw_data = merged_audio.raw_data

    # Initialize list for base64 chunks
    base64_chunks = []

    if enable_chunking and frames_per_chunk:
        # Calculate total number of frames
        total_frames = len(raw_data) // bytes_per_sample

        # Split into chunks at frame boundaries
        for i in range(0, total_frames, frames_per_chunk):
            # Calculate start and end frame indices
            start_frame = i
            end_frame = min(i + frames_per_chunk, total_frames)

            # Calculate byte positions
            start_byte = start_frame * bytes_per_sample
            end_byte = end_frame * bytes_per_sample

            # Extract chunk data
            chunk_data = raw_data[start_byte:end_byte]

            # Encode to base64
            base64_chunk = base64.b64encode(chunk_data).decode('utf-8')

            # Add to chunks list
            base64_chunks.append(base64_chunk)
    else:
        # No chunking, just encode the entire audio
        base64_chunks = [base64.b64encode(raw_data).decode('utf-8')]

    if enforce_format is not None:
        new_chunks = []
        for chunk in base64_chunks:
            buffer = io.BytesIO(base64.b64decode(chunk))

            segment : AudioSegment = AudioSegment.from_raw(
                buffer,
                sample_width=bytes_per_sample,
                frame_rate=target_sample_rate,
                channels=1 if enforce_mono else merged_audio.channels
            )

            new_chunks.append(segment.export(format=enforce_format).read())

        base64_chunks = [base64.b64encode(chunk).decode('utf-8') for chunk in new_chunks]

    # Return chunks and metadata
    return {
        "chunks": base64_chunks,
        "duration_seconds": len(merged_audio) / 1000,  # pydub duration is in ms
        "total_size_bytes": len(raw_data),
        "num_chunks": len(base64_chunks)
    }

def base64_to_temp_audio_file(
    base64_data: str,
    sample_rate: int = 24000,
    channels: int = 1,
    bit_depth: Literal[8, 16, 32] = 16,
    file_format: str = "wav",
    delete_on_exit: bool = True,
    output_folder: Optional[str] = None
) -> Dict[str, str]:
    """
    Convert base64-encoded audio data to a temporary audio file.
    
    Args:
        base64_data: Base64-encoded audio data (raw PCM)
        sample_rate: Sample rate of the audio data
        channels: Number of audio channels
        bit_depth: Bit depth of the audio (8, 16, or 32)
        file_format: Output audio format (wav, mp3, ogg, etc.)
        delete_on_exit: Whether to automatically delete the file when the program exits
        output_folder: Folder to save the temporary file
        
    Returns:
        Dict containing the file path and metadata
    """
    if not base64_data:
        raise ValueError("No base64 data provided")

    try:
        # Decode the base64 data
        binary_data = base64.b64decode(base64_data)
    except Exception as e:
        raise ValueError(f"Failed to decode base64 data: {str(e)}") from e

    # Clean up the file format and prepare extension
    file_format = file_format.lower().strip()
    if file_format.startswith('.'):
        file_format = file_format[1:]
    file_extension = f".{file_format}"

    try:
        # Determine the output directory
        if output_folder:
            # Use the specified folder
            if not os.path.exists(output_folder):
                os.makedirs(output_folder, exist_ok=True)

            # Generate a unique filename in the specified folder
            temp_filename = f"audio_{tempfile.mktemp(dir='').split('/')[-1]}{file_extension}"
            temp_path = os.path.join(output_folder, temp_filename)
        else:
            # Use temporary directory
            if delete_on_exit:
                # This will be automatically deleted when the program exits
                fd, temp_path = tempfile.mkstemp(suffix=file_extension)
                os.close(fd)  # Close the file descriptor
            else:
                # Create a file that won't be automatically deleted
                temp_dir = tempfile.gettempdir()
                temp_filename = f"audio_{tempfile.mktemp(dir='').split('/')[-1]}{file_extension}"
                temp_path = os.path.join(temp_dir, temp_filename)

        # Calculate bytes per sample
        bytes_per_sample = bit_depth // 8

        # Create an in-memory file-like object
        buffer = io.BytesIO(binary_data)

        # Create an AudioSegment from the raw data
        audio_segment = AudioSegment.from_raw(
            buffer,
            sample_width=bytes_per_sample,
            frame_rate=sample_rate,
            channels=channels
        )

        # Export to the desired format
        audio_segment.export(
            temp_path,
            format=file_format
        )

        # Duration is already available from the AudioSegment
        duration_seconds = len(audio_segment) / 1000.0  # pydub duration is in ms

        return {
            "file_path": temp_path,
            "size_bytes": os.path.getsize(temp_path),
            "original_data_size": len(binary_data),
            "duration_seconds": duration_seconds,
            "will_delete": delete_on_exit
        }

    except Exception as e:
        raise RuntimeError(f"Failed to create temporary audio file: {str(e)}") from e

# Helper function for common file formats
def base64_to_wav(base64_data: str, output_folder: Optional[str] = None, **kwargs) -> str:
    """Shortcut to convert base64 data to a WAV file and return just the path"""
    result = base64_to_temp_audio_file(
        base64_data,
        file_format="wav",
        output_folder=output_folder,
        **kwargs
    )

    return result["file_path"]

def validate_json_response(response_json: Dict[str, Any]) -> str:
    """Validate and extract the content from a JSON response."""
    if "choices" not in response_json:
        raise ValueError(f"Error: missing 'choices' key in response json: {response_json}")

    if not isinstance(response_json["choices"], list):
        raise ValueError(f"Error: 'choices' key in response json must be a list: {response_json}")

    if "message" not in response_json["choices"][0]:
        raise ValueError(f"Error: missing 'message' key in response json: {response_json}")

    if "content" not in response_json["choices"][0]["message"]:
        raise ValueError(f"Error: missing 'content' key in response json: {response_json}")

    return response_json["choices"][0]["message"]["content"]

@dataclass
class NodeCompletionParameters:
    """Parameters for node completion."""
    gi: GeneratorInfo
    generation_parameters: Optional[GeneratorCompletionParameters] = None

    # Response handling
    add_child: bool = False
    parse_json: bool = False
    merge_contiguous: str = "all"  # "all", "user", "assistant", "system", "base", or None

    # Prompt modification
    force_prepend: Optional[str] = None

    # Retry and error handling
    retry: int = 4
    exp_back_off: bool = False
    back_off_time: float = 1.0  # seconds
    max_back_off: int = 15     # seconds
    crash_on_refusal: bool = False
    crash_on_empty_response: bool = False

    # Number of completions
    n: int = 1
