#!/usr/bin/env python3
"""
Test suite for ImageData class functionality.
"""

import os
import tempfile
import unittest
from unittest.mock import patch

from src.minillmlib.utils.message_utils import ImageData, process_images_for_completion


class TestImageData(unittest.TestCase):
    """Test cases for ImageData class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_image_path = os.path.join(os.path.dirname(__file__), "hqdefault.jpg")
        self.test_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
        self.test_data_url = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAYEBQYFBAYGBQYHBwYIChAKCgkJChQODwwQFxQYGBcUFhYaHSUfGhsjHBYWICwgIyYnKSopGR8tMC0oMCUoKSj/2wBDAQcHBwoIChMKChMoGhYaKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCj/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCdABmX/9k="
        
    def test_init_with_images_list(self):
        """Test initialization with images list."""
        images = [self.test_url, self.test_image_path]
        image_data = ImageData(images=images)
        
        self.assertEqual(image_data.images, images)
        
    def test_init_empty(self):
        """Test initialization with empty list."""
        image_data = ImageData()
        self.assertEqual(image_data.images, [])
        
        
    def test_merge(self):
        """Test merging two ImageData instances."""
        image_data1 = ImageData(images=[self.test_url])
        image_data2 = ImageData(images=[self.test_image_path])
        
        merged = image_data1.merge(image_data2)
        
        expected_images = [self.test_url, self.test_image_path]
        self.assertEqual(merged.images, expected_images)
        
    def test_is_url_detection(self):
        """Test URL detection."""
        # Test HTTP URLs
        self.assertTrue(ImageData._is_url("http://example.com/image.jpg"))
        self.assertTrue(ImageData._is_url("https://example.com/image.jpg"))
        
        # Test data URLs
        self.assertTrue(ImageData._is_url(self.test_data_url))
        
        # Test non-URLs
        self.assertFalse(ImageData._is_url("/path/to/image.jpg"))
        self.assertFalse(ImageData._is_url("image.jpg"))
        
    def test_is_local_path_detection(self):
        """Test local path detection."""
        # Test existing file
        self.assertTrue(ImageData._is_local_path(self.test_image_path))
        
        # Test non-existing file
        self.assertFalse(ImageData._is_local_path("/nonexistent/path.jpg"))
        
        # Test URL (should not be detected as local path)
        self.assertFalse(ImageData._is_local_path(self.test_url))
        
    def test_get_processed_images_with_url(self):
        """Test processing images with URLs."""
        image_data = ImageData(images=[self.test_url])
        processed = image_data.get_processed_images()
        
        # URL should remain unchanged
        self.assertEqual(processed, [self.test_url])
        
    def test_get_processed_images_with_data_url(self):
        """Test processing images with data URLs."""
        image_data = ImageData(images=[self.test_data_url])
        processed = image_data.get_processed_images()
        
        # Data URL should remain unchanged
        self.assertEqual(processed, [self.test_data_url])
        
    def test_get_processed_images_with_local_file(self):
        """Test processing images with local files."""
        image_data = ImageData(images=[self.test_image_path])
        processed = image_data.get_processed_images()
        
        # Should have one processed image
        self.assertEqual(len(processed), 1)
        
        # Should be converted to data URL
        self.assertTrue(processed[0].startswith("data:image/"))
        self.assertIn("base64,", processed[0])
        
    def test_get_processed_images_mixed(self):
        """Test processing mixed image types."""
        images = [
            self.test_url,           # URL
            self.test_image_path,    # Local file
            self.test_data_url       # Data URL
        ]
        
        image_data = ImageData(images=images)
        processed = image_data.get_processed_images()
        
        # Should have 3 processed images
        self.assertEqual(len(processed), 3)
        
        # First should be unchanged URL
        self.assertEqual(processed[0], self.test_url)
        
        # Second should be converted to data URL
        self.assertTrue(processed[1].startswith("data:image/"))
        
        # Third should be unchanged data URL
        self.assertEqual(processed[2], self.test_data_url)
        
    def test_get_processed_images_with_nonexistent_file(self):
        """Test processing with non-existent local file."""
        nonexistent_path = "/nonexistent/image.jpg"
        image_data = ImageData(images=[nonexistent_path])
        
        # Should handle gracefully and return empty list
        with patch('builtins.print') as mock_print:
            processed = image_data.get_processed_images()
            
        self.assertEqual(processed, [])
        mock_print.assert_called_once()
        
    def test_process_images_for_completion(self):
        """Test the process_images_for_completion function."""
        image_data = ImageData(images=[
            self.test_url,
            self.test_image_path
        ])
        
        result = process_images_for_completion(image_data)
        
        # Should return list of content dictionaries
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        
        # Each item should have correct structure
        for item in result:
            self.assertEqual(item["type"], "image_url")
            self.assertIn("image_url", item)
            self.assertIn("url", item["image_url"])
            
        # First should be the original URL
        self.assertEqual(result[0]["image_url"]["url"], self.test_url)
        
        # Second should be converted to data URL
        self.assertTrue(result[1]["image_url"]["url"].startswith("data:image/"))
        
    def test_process_images_for_completion_empty(self):
        """Test process_images_for_completion with empty ImageData."""
        image_data = ImageData()
        result = process_images_for_completion(image_data)
        
        self.assertEqual(result, [])
        
    def test_image_file_exists(self):
        """Verify the test image file exists."""
        self.assertTrue(os.path.exists(self.test_image_path))
        self.assertTrue(os.path.isfile(self.test_image_path))
        
    def test_base64_conversion_produces_valid_data_url(self):
        """Test that base64 conversion produces valid data URL format."""
        image_data = ImageData(images=[self.test_image_path])
        processed = image_data.get_processed_images()
        
        data_url = processed[0]
        
        # Should start with data: scheme
        self.assertTrue(data_url.startswith("data:"))
        
        # Should contain MIME type
        self.assertIn("image/", data_url)
        
        # Should contain base64 marker
        self.assertIn("base64,", data_url)
        
        # Should have base64 data after the comma
        base64_part = data_url.split("base64,")[1]
        self.assertGreater(len(base64_part), 0)


if __name__ == "__main__":
    # Run the tests
    unittest.main(verbosity=2)
