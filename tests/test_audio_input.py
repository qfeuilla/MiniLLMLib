"""Test audio input functionality for OpenRouter."""
import base64
import os
import struct
import tempfile
import wave
from unittest import TestCase

from minillmlib.core.chat_node import ChatNode
from minillmlib.models.generator_info import GeneratorInfo
from minillmlib.utils.message_utils import (AudioData, ImageData,
                                            process_audio_input_for_completion)


class TestAudioInputProcessing(TestCase):
    """Test audio input processing for OpenRouter format."""

    def setUp(self):
        """Create test audio files."""
        self.test_dir = tempfile.mkdtemp()
        
        # Create a WAV file
        self.wav_path = os.path.join(self.test_dir, "test.wav")
        with wave.open(self.wav_path, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(24000)  # 24kHz
            
            # Generate 0.1 second of audio
            samples = []
            for _ in range(2400):
                sample = int(32767 * 0.5)
                samples.append(struct.pack('h', sample))
            
            wav_file.writeframes(b''.join(samples))
        
        # Create an MP3 file (just copy the WAV for testing purposes)
        self.mp3_path = os.path.join(self.test_dir, "test.mp3")
        with open(self.wav_path, 'rb') as src:
            with open(self.mp3_path, 'wb') as dst:
                dst.write(src.read())

    def tearDown(self):
        """Clean up test files."""
        if os.path.exists(self.test_dir):
            for file in os.listdir(self.test_dir):
                os.remove(os.path.join(self.test_dir, file))
            os.rmdir(self.test_dir)

    def test_process_audio_input_wav(self):
        """Test processing WAV audio for input."""
        audio_data = AudioData(audio_paths=[self.wav_path])
        result = process_audio_input_for_completion(audio_data)
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["type"], "input_audio")
        self.assertIn("input_audio", result[0])
        self.assertIn("data", result[0]["input_audio"])
        self.assertIn("format", result[0]["input_audio"])
        self.assertEqual(result[0]["input_audio"]["format"], "wav")
        
        # Verify base64 encoding
        base64_data = result[0]["input_audio"]["data"]
        decoded = base64.b64decode(base64_data)
        self.assertGreater(len(decoded), 0)

    def test_process_audio_input_mp3(self):
        """Test processing MP3 audio for input."""
        audio_data = AudioData(audio_paths=[self.mp3_path])
        result = process_audio_input_for_completion(audio_data)
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["input_audio"]["format"], "mp3")

    def test_process_multiple_audio_files(self):
        """Test processing multiple audio files."""
        audio_data = AudioData(audio_paths=[self.wav_path, self.mp3_path])
        result = process_audio_input_for_completion(audio_data)
        
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["input_audio"]["format"], "wav")
        self.assertEqual(result[1]["input_audio"]["format"], "mp3")

    def test_audio_input_with_missing_file(self):
        """Test that missing audio file raises error."""
        audio_data = AudioData(audio_paths=["nonexistent.wav"])
        
        with self.assertRaises(FileNotFoundError):
            process_audio_input_for_completion(audio_data)


class TestChatNodeAudioInput(TestCase):
    """Test ChatNode with audio input."""

    def setUp(self):
        """Create test audio file."""
        self.test_dir = tempfile.mkdtemp()
        self.wav_path = os.path.join(self.test_dir, "test.wav")
        
        with wave.open(self.wav_path, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(24000)
            samples = [struct.pack('h', int(32767 * 0.5)) for _ in range(2400)]
            wav_file.writeframes(b''.join(samples))

    def tearDown(self):
        """Clean up test files."""
        if os.path.exists(self.test_dir):
            for file in os.listdir(self.test_dir):
                os.remove(os.path.join(self.test_dir, file))
            os.rmdir(self.test_dir)

    def test_create_node_with_audio_input(self):
        """Test creating a ChatNode with audio input."""
        audio_data = AudioData(audio_paths=[self.wav_path])
        node = ChatNode(
            content="Please transcribe this audio.",
            role="user",
            audio_data=audio_data
        )
        
        self.assertEqual(node.role, "user")
        self.assertEqual(node.content, "Please transcribe this audio.")
        self.assertIsNotNone(node.audio_data)

    def test_create_node_audio_only(self):
        """Test creating a ChatNode with only audio (no text)."""
        audio_data = AudioData(audio_paths=[self.wav_path])
        node = ChatNode(role="user", audio_data=audio_data)
        
        self.assertEqual(node.role, "user")
        self.assertIsNone(node.content)
        self.assertIsNotNone(node.audio_data)

    def test_assistant_audio_with_content_raises_error(self):
        """Test that assistant role cannot combine audio with content."""
        audio_data = AudioData(audio_paths=[self.wav_path])
        
        with self.assertRaises(ValueError) as context:
            ChatNode(
                content="Some text",
                role="assistant",
                audio_data=audio_data
            )
        
        self.assertIn("assistant", str(context.exception).lower())

    def test_get_messages_with_audio_input_url_format(self):
        """Test get_messages() with audio input for OpenRouter (url format)."""
        audio_data = AudioData(audio_paths=[self.wav_path])
        node = ChatNode(
            content="Transcribe this:",
            role="user",
            audio_data=audio_data
        )
        
        gi = GeneratorInfo(
            model="openai/whisper-large-v3",
            _format="url",
            api_url="https://openrouter.ai/api/v1/chat/completions",
            api_key="test_key"
        )
        
        messages = node.get_messages(gi=gi)
        
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "user")
        self.assertIsInstance(messages[0]["content"], list)
        
        # Should have text and audio content
        content_array = messages[0]["content"]
        self.assertGreaterEqual(len(content_array), 2)
        
        # Check text content
        text_content = [c for c in content_array if c["type"] == "text"]
        self.assertEqual(len(text_content), 1)
        self.assertEqual(text_content[0]["text"], "Transcribe this:")
        
        # Check audio content
        audio_content = [c for c in content_array if c["type"] == "input_audio"]
        self.assertEqual(len(audio_content), 1)
        self.assertEqual(audio_content[0]["input_audio"]["format"], "wav")

    def test_get_messages_audio_only_url_format(self):
        """Test get_messages() with audio-only input for OpenRouter."""
        audio_data = AudioData(audio_paths=[self.wav_path])
        node = ChatNode(role="user", audio_data=audio_data)
        
        gi = GeneratorInfo(
            model="openai/whisper-large-v3",
            _format="url",
            api_url="https://openrouter.ai/api/v1/chat/completions",
            api_key="test_key"
        )
        
        messages = node.get_messages(gi=gi)
        
        self.assertEqual(len(messages), 1)
        self.assertIsInstance(messages[0]["content"], list)
        
        # Should have only audio content
        content_array = messages[0]["content"]
        audio_content = [c for c in content_array if c["type"] == "input_audio"]
        self.assertEqual(len(audio_content), 1)

    def test_get_messages_audio_input_openai_format(self):
        """Test that audio input works with openai format too."""
        audio_data = AudioData(audio_paths=[self.wav_path])
        node = ChatNode(
            content="Transcribe:",
            role="user",
            audio_data=audio_data
        )
        
        gi = GeneratorInfo(
            model="gpt-4o-audio-preview",
            _format="openai",
            api_key="test_key"
        )
        
        messages = node.get_messages(gi=gi)
        
        self.assertEqual(len(messages), 1)
        self.assertIsInstance(messages[0]["content"], list)
        
        # Should have text and audio content
        content_array = messages[0]["content"]
        text_content = [c for c in content_array if c["type"] == "text"]
        audio_content = [c for c in content_array if c["type"] == "input_audio"]
        
        self.assertEqual(len(text_content), 1)
        self.assertEqual(len(audio_content), 1)

    def test_audio_and_image_combined(self):
        """Test that audio and images can be combined in the same node."""
        audio_data = AudioData(audio_paths=[self.wav_path])
        # Create a simple base64 image data URL
        image_data = ImageData(images=["data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="])
        
        node = ChatNode(
            content="What's in this image and audio?",
            role="user",
            audio_data=audio_data,
            image_data=image_data
        )
        
        gi = GeneratorInfo(
            model="gpt-4o",
            _format="url",
            api_url="https://openrouter.ai/api/v1/chat/completions",
            api_key="test_key"
        )
        
        messages = node.get_messages(gi=gi)
        
        self.assertEqual(len(messages), 1)
        self.assertIsInstance(messages[0]["content"], list)
        
        # Should have text, image, and audio content
        content_array = messages[0]["content"]
        text_content = [c for c in content_array if c["type"] == "text"]
        image_content = [c for c in content_array if c["type"] == "image_url"]
        audio_content = [c for c in content_array if c["type"] == "input_audio"]
        
        self.assertEqual(len(text_content), 1)
        self.assertEqual(len(image_content), 1)
        self.assertEqual(len(audio_content), 1)
        self.assertEqual(text_content[0]["text"], "What's in this image and audio?")
