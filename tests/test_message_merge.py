"""Test message merging with audio content."""

import os
import tempfile
from unittest import TestCase

from minillmlib.utils.message_utils import AudioData, merge_contiguous_messages


class TestMessageMerge(TestCase):
    """Test message merging with audio content."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.test_audio_path1 = os.path.join(self.test_dir, "test1.wav")
        self.test_audio_path2 = os.path.join(self.test_dir, "test2.wav")

        # Create dummy audio files
        with open(self.test_audio_path1, "wb") as f:
            f.write(b"audio data 1")
        with open(self.test_audio_path2, "wb") as f:
            f.write(b"audio data 2")

        self.audio_data1 = AudioData(
            audio_paths=[self.test_audio_path1],
            audio_raw="YXVkaW8gZGF0YSAx",  # base64 "audio data 1"
        )

        self.audio_data2 = AudioData(
            audio_paths=[self.test_audio_path2],
            audio_raw="YXVkaW8gZGF0YSAy",  # base64 "audio data 2"
        )

    def tearDown(self):
        if os.path.exists(self.test_dir):
            for file in os.listdir(self.test_dir):
                os.remove(os.path.join(self.test_dir, file))
            os.rmdir(self.test_dir)

    def test_merge_text_messages(self):
        """Test merging text messages."""
        messages = [
            {"role": "user", "content": "Hello", "audio_data": None},
            {"role": "user", "content": "How are you?", "audio_data": None},
            {"role": "assistant", "content": "I'm good", "audio_data": None},
        ]

        merged = merge_contiguous_messages(messages, merge_contiguous="all")
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0]["content"], "Hello\nHow are you?")
        self.assertEqual(merged[1]["content"], "I'm good")

    def test_merge_audio_messages(self):
        """Test merging audio messages."""
        messages = [
            {"role": "user", "content": None, "audio_data": self.audio_data1},
            {"role": "user", "content": None, "audio_data": self.audio_data2},
            {"role": "assistant", "content": "I heard you", "audio_data": None},
        ]

        merged = merge_contiguous_messages(messages, merge_contiguous="all")
        self.assertEqual(len(merged), 2)

        # Check merged audio data
        merged_audio = merged[0]["audio_data"]
        self.assertEqual(len(merged_audio.audio_paths), 2)
        self.assertEqual(merged_audio.audio_paths[0], self.test_audio_path1)
        self.assertEqual(merged_audio.audio_paths[1], self.test_audio_path2)

    def test_no_merge_different_types(self):
        """Test no merging of different types."""
        messages = [
            {"role": "user", "content": "Hello", "audio_data": None},
            {"role": "user", "content": None, "audio_data": self.audio_data1},
            {"role": "user", "content": "Text again", "audio_data": None},
        ]

        merged = merge_contiguous_messages(messages, merge_contiguous="all")
        self.assertEqual(len(merged), 3)  # Should not merge different types

    def test_merge_with_system_message(self):
        """Test merging with system message."""
        messages = [
            {"role": "system", "content": "System prompt", "audio_data": None},
            {"role": "user", "content": None, "audio_data": self.audio_data1},
            {"role": "user", "content": None, "audio_data": self.audio_data2},
        ]

        merged = merge_contiguous_messages(messages, merge_contiguous="all")
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0]["role"], "system")
        self.assertEqual(merged[1]["role"], "user")
        self.assertEqual(len(merged[1]["audio_data"].audio_paths), 2)

    def test_merge_selective_roles(self):
        """Test merging selective roles."""
        messages = [
            {"role": "user", "content": None, "audio_data": self.audio_data1},
            {"role": "user", "content": None, "audio_data": self.audio_data2},
            {"role": "assistant", "content": "Response 1", "audio_data": None},
            {"role": "assistant", "content": "Response 2", "audio_data": None},
        ]

        # Test merging only assistant messages
        merged = merge_contiguous_messages(messages, merge_contiguous="assistant")
        self.assertEqual(len(merged), 3)
        self.assertEqual(merged[0]["role"], "user")
        self.assertEqual(merged[1]["role"], "user")
        self.assertEqual(merged[2]["role"], "assistant")
        self.assertEqual(
            len(merged[0]["audio_data"].audio_paths), 1
        )  # User messages doesn't merge
        self.assertEqual(merged[2]["content"], "Response 1\nResponse 2")

    def test_invalid_messages(self):
        """Test invalid messages."""
        invalid_messages = [
            {"role": "user"},  # Missing content and audio_data
            {"role": "invalid", "content": "test", "audio_data": None},  # Invalid role
            {"content": "test", "audio_data": None},  # Missing role
        ]

        for msg in invalid_messages:
            with self.assertRaises(ValueError):
                merge_contiguous_messages([msg], merge_contiguous="all")
