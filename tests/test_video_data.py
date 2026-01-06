#!/usr/bin/env python3
"""
Test suite for VideoData class functionality.
"""

import os
import unittest
from unittest.mock import patch

from src.minillmlib.utils.message_utils import VideoData, process_videos_for_completion, _is_url, _is_local_path


class TestVideoData(unittest.TestCase):
    """Test cases for VideoData class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_video_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "test.mp4"
        )
        self.test_url = "https://example.com/video.mp4"
        self.test_youtube_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        self.test_data_url = "data:video/mp4;base64,AAAAIGZ0eXBpc29tAAACAGlzb21pc28yYXZjMW1wNDE="
        
    def test_init_with_videos_list(self):
        """Test initialization with videos list."""
        videos = [self.test_url, self.test_video_path]
        video_data = VideoData(videos=videos)
        
        self.assertEqual(video_data.videos, videos)
        
    def test_init_empty(self):
        """Test initialization with empty list."""
        video_data = VideoData()
        self.assertEqual(video_data.videos, [])
        
    def test_merge(self):
        """Test merging two VideoData instances."""
        video_data1 = VideoData(videos=[self.test_url])
        video_data2 = VideoData(videos=[self.test_youtube_url])
        
        merged = video_data1.merge(video_data2)
        
        expected_videos = [self.test_url, self.test_youtube_url]
        self.assertEqual(merged.videos, expected_videos)
        
    def test_merge_preserves_original(self):
        """Test that merge doesn't modify original instances."""
        video_data1 = VideoData(videos=[self.test_url])
        video_data2 = VideoData(videos=[self.test_youtube_url])
        
        merged = video_data1.merge(video_data2)
        
        # Original instances should be unchanged
        self.assertEqual(video_data1.videos, [self.test_url])
        self.assertEqual(video_data2.videos, [self.test_youtube_url])
        
    def test_is_url_detection_for_videos(self):
        """Test URL detection for video URLs."""
        # Test HTTP URLs
        self.assertTrue(_is_url("http://example.com/video.mp4"))
        self.assertTrue(_is_url("https://example.com/video.mp4"))
        
        # Test YouTube URLs
        self.assertTrue(_is_url(self.test_youtube_url))
        
        # Test data URLs
        self.assertTrue(_is_url(self.test_data_url))
        
        # Test non-URLs
        self.assertFalse(_is_url("/path/to/video.mp4"))
        self.assertFalse(_is_url("video.mp4"))
        
    def test_get_processed_videos_with_url(self):
        """Test processing videos with URLs."""
        video_data = VideoData(videos=[self.test_url])
        processed = video_data.get_processed_videos()
        
        # URL should remain unchanged
        self.assertEqual(processed, [self.test_url])
        
    def test_get_processed_videos_with_youtube_url(self):
        """Test processing videos with YouTube URLs."""
        video_data = VideoData(videos=[self.test_youtube_url])
        processed = video_data.get_processed_videos()
        
        # YouTube URL should remain unchanged
        self.assertEqual(processed, [self.test_youtube_url])
        
    def test_get_processed_videos_with_data_url(self):
        """Test processing videos with data URLs."""
        video_data = VideoData(videos=[self.test_data_url])
        processed = video_data.get_processed_videos()
        
        # Data URL should remain unchanged
        self.assertEqual(processed, [self.test_data_url])
        
    def test_get_processed_videos_with_local_file(self):
        """Test processing videos with local files."""
        if not os.path.exists(self.test_video_path):
            self.skipTest("Test video file not found")
            
        video_data = VideoData(videos=[self.test_video_path])
        processed = video_data.get_processed_videos()
        
        # Should have one processed video
        self.assertEqual(len(processed), 1)
        
        # Should be converted to data URL
        self.assertTrue(processed[0].startswith("data:video/"))
        self.assertIn("base64,", processed[0])
        
    def test_get_processed_videos_mixed(self):
        """Test processing mixed video types."""
        if not os.path.exists(self.test_video_path):
            self.skipTest("Test video file not found")
            
        videos = [
            self.test_url,           # URL
            self.test_video_path,    # Local file
            self.test_data_url       # Data URL
        ]
        
        video_data = VideoData(videos=videos)
        processed = video_data.get_processed_videos()
        
        # Should have 3 processed videos
        self.assertEqual(len(processed), 3)
        
        # First should be unchanged URL
        self.assertEqual(processed[0], self.test_url)
        
        # Second should be converted to data URL
        self.assertTrue(processed[1].startswith("data:video/"))
        
        # Third should be unchanged data URL
        self.assertEqual(processed[2], self.test_data_url)
        
    def test_get_processed_videos_with_nonexistent_file(self):
        """Test processing with non-existent local file."""
        nonexistent_path = "/nonexistent/video.mp4"
        video_data = VideoData(videos=[nonexistent_path])
        
        # Should handle gracefully and return empty list
        with patch('builtins.print') as mock_print:
            processed = video_data.get_processed_videos()
            
        self.assertEqual(processed, [])
        mock_print.assert_called_once()
        
    def test_process_videos_for_completion(self):
        """Test the process_videos_for_completion function."""
        video_data = VideoData(videos=[
            self.test_url,
            self.test_youtube_url
        ])
        
        result = process_videos_for_completion(video_data)
        
        # Should return list of content dictionaries
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        
        # Each item should have correct structure for OpenRouter video format
        for item in result:
            self.assertEqual(item["type"], "video_url")
            self.assertIn("video_url", item)
            self.assertIn("url", item["video_url"])
            
        # First should be the original URL
        self.assertEqual(result[0]["video_url"]["url"], self.test_url)
        
        # Second should be the YouTube URL
        self.assertEqual(result[1]["video_url"]["url"], self.test_youtube_url)
        
    def test_process_videos_for_completion_with_local_file(self):
        """Test process_videos_for_completion with local file."""
        if not os.path.exists(self.test_video_path):
            self.skipTest("Test video file not found")
            
        video_data = VideoData(videos=[self.test_video_path])
        result = process_videos_for_completion(video_data)
        
        # Should have one result
        self.assertEqual(len(result), 1)
        
        # Should have correct structure
        self.assertEqual(result[0]["type"], "video_url")
        
        # Should be converted to data URL
        self.assertTrue(result[0]["video_url"]["url"].startswith("data:video/"))
        
    def test_process_videos_for_completion_empty(self):
        """Test process_videos_for_completion with empty VideoData."""
        video_data = VideoData()
        result = process_videos_for_completion(video_data)
        
        self.assertEqual(result, [])
        
    def test_video_file_exists(self):
        """Verify the test video file exists."""
        if not os.path.exists(self.test_video_path):
            self.skipTest("Test video file not found")
            
        self.assertTrue(os.path.exists(self.test_video_path))
        self.assertTrue(os.path.isfile(self.test_video_path))
        
    def test_base64_conversion_produces_valid_data_url(self):
        """Test that base64 conversion produces valid data URL format."""
        if not os.path.exists(self.test_video_path):
            self.skipTest("Test video file not found")
            
        video_data = VideoData(videos=[self.test_video_path])
        processed = video_data.get_processed_videos()
        
        data_url = processed[0]
        
        # Should start with data: scheme
        self.assertTrue(data_url.startswith("data:"))
        
        # Should contain MIME type
        self.assertIn("video/", data_url)
        
        # Should contain base64 marker
        self.assertIn("base64,", data_url)
        
        # Should have base64 data after the comma
        base64_part = data_url.split("base64,")[1]
        self.assertGreater(len(base64_part), 0)


if __name__ == "__main__":
    # Run the tests
    unittest.main(verbosity=2)
