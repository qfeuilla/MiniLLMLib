"""Test audio utilities."""
import base64
import os
import struct
import tempfile
import wave
from unittest import TestCase

from minillmlib.utils.message_utils import (AudioData,
                                            base64_to_temp_audio_file,
                                            base64_to_wav,
                                            process_audio_for_completion)


class TestAudioData(TestCase):
    """Test AudioData class."""

    def setUp(self):
        self.test_audio_data = AudioData(
            audio_paths=["test1.wav", "test2.wav"],
            audio_ids={
                "test_id": {
                    "transcript": "test transcript",
                    "expires_at": 1234567890
                }
            },
            audio_raw="dGVzdA=="  # base64 encoded "test"
        )

    def test_merge(self):
        """Test merging two AudioData instances."""
        other_audio = AudioData(
            audio_paths=["test3.wav"],
            audio_ids={
                "test_id_2": {
                    "transcript": "another transcript",
                    "expires_at": 1234567891
                }
            },
            audio_raw="dGVzdDI="  # base64 encoded "test2"
        )

        merged = self.test_audio_data.merge(other_audio)

        self.assertEqual(len(merged.audio_paths), 3)
        self.assertEqual(len(merged.audio_ids), 2)
        self.assertTrue("test_id" in merged.audio_ids)
        self.assertTrue("test_id_2" in merged.audio_ids)

        # Test that raw audio is properly concatenated
        merged_raw = base64.b64decode(merged.audio_raw)
        self.assertEqual(len(merged_raw), 9)  # "test" + "test2" = 9 bytes

    def test_merge_empty_raw(self):
        """Test merging when one or both audio_raw fields are empty."""
        empty_audio = AudioData(audio_paths=["test4.wav"])
        merged = self.test_audio_data.merge(empty_audio)
        self.assertEqual(
            base64.b64decode(merged.audio_raw),
            base64.b64decode(self.test_audio_data.audio_raw)
        )

        empty_merged = empty_audio.merge(empty_audio)
        self.assertEqual(empty_merged.audio_raw, "")

class TestAudioProcessing(TestCase):
    """Test audio processing utilities."""

    def setUp(self):
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()

        # Create a proper WAV file for testing
        self.test_audio_path = os.path.join(self.test_dir, "test.wav")
        with wave.open(self.test_audio_path, 'wb') as wav_file:
            # Set parameters
            wav_file.setnchannels(1)  # Mono pylint: disable=no-member
            wav_file.setsampwidth(2)  # 16-bit pylint: disable=no-member
            wav_file.setframerate(24000)  # 24kHz pylint: disable=no-member

            # Generate 1 second of audio (sine wave)
            samples = []
            for _ in range(24000):
                sample = int(32767 * 0.5)  # Constant value at 50% amplitude
                samples.append(struct.pack('h', sample))

            wav_file.writeframes(b''.join(samples))  # pylint: disable=no-member

    def tearDown(self):
        """Clean up temporary files."""
        if os.path.exists(self.test_dir):
            for file in os.listdir(self.test_dir):
                os.remove(os.path.join(self.test_dir, file))
            os.rmdir(self.test_dir)

    def test_process_audio_for_completion(self):
        """Test audio processing for completion."""
        result = process_audio_for_completion(
            file_paths=[self.test_audio_path],
            target_sample_rate=24000,
            enforce_mono=True,
            bit_depth=16,
            enable_chunking=True,
            max_chunk_size=500  # Small chunk size for testing
        )

        self.assertIn("chunks", result)
        self.assertIn("duration_seconds", result)
        self.assertIn("total_size_bytes", result)
        self.assertIn("num_chunks", result)

        self.assertGreater(result["duration_seconds"], 0)
        self.assertGreater(len(result["chunks"]), 0)

    def test_base64_to_temp_audio_file(self):
        """Test base64 to temporary audio file."""
        with open(self.test_audio_path, 'rb') as f:
            test_audio_b64 = base64.b64encode(f.read()).decode('utf-8')

        result = base64_to_temp_audio_file(
            base64_data=test_audio_b64,
            sample_rate=24000,
            channels=1,
            bit_depth=16,
            file_format="wav",
            delete_on_exit=True
        )

        self.assertTrue(os.path.exists(result["file_path"]))
        self.assertTrue(result["will_delete"])

    def test_base64_to_wav(self):
        """Test base64 to WAV conversion."""
        with open(self.test_audio_path, 'rb') as f:
            test_audio_b64 = base64.b64encode(f.read()).decode('utf-8')

        output_path = base64_to_wav(test_audio_b64, output_folder=self.test_dir)

        self.assertTrue(os.path.exists(output_path))
        self.assertTrue(output_path.endswith(".wav"))

        # Verify the output file is a valid WAV file
        with wave.open(output_path, 'rb') as wav_file:
            self.assertEqual(wav_file.getnchannels(), 1)
            self.assertEqual(wav_file.getframerate(), 24000)
            self.assertEqual(wav_file.getsampwidth(), 2)  # 16-bit
