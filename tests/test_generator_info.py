"""Tests for the GeneratorInfo class."""
import unittest

from minillmlib.models.generator_info import GeneratorInfo, GeneratorCompletionParameters, HUGGINGFACE_ACTIVATED

class TestGeneratorInfo(unittest.TestCase):
    def test_init(self):
        gi = GeneratorInfo(
            model="test-model",
            _format="openai",
            api_key="test-key"
        )
        
        self.assertEqual(gi.model, "test-model")
        self.assertEqual(gi._format, "openai")
        self.assertEqual(gi.api_key, "test-key")
        self.assertTrue(gi.is_chat)
        self.assertIsInstance(gi.completion_parameters, GeneratorCompletionParameters)

    def test_generation_parameters(self):
        params = GeneratorCompletionParameters(
            temperature=0.7,
            max_tokens=100,
            n=2,
            top_p= 0.9
        )
        
        gi = GeneratorInfo(
            model="test-model",
            _format="openai",
            api_key="test-key",
            completion_parameters=params
        )
        
        self.assertEqual(gi.completion_parameters.temperature, 0.7)
        self.assertEqual(gi.completion_parameters.max_tokens, 100)
        self.assertEqual(gi.completion_parameters.n, 2)
        self.assertEqual(gi.completion_parameters.kwargs["top_p"], 0.9)

    def test_generator_completion_parameters_with_kwargs(self):
        # Test with only standard parameters
        params = GeneratorCompletionParameters(
            temperature=0.7,
            max_tokens=100,
            n=2
        )
        self.assertEqual(params.temperature, 0.7)
        self.assertEqual(params.max_tokens, 100)
        self.assertEqual(params.n, 2)
        self.assertEqual(params.kwargs, {})

        # Test with additional kwargs
        params = GeneratorCompletionParameters(
            temperature=0.7,
            custom_param="value",
            another_param=123,
            special_flag=True
        )
        self.assertEqual(params.temperature, 0.7)
        self.assertEqual(params.kwargs, {
            "custom_param": "value",
            "another_param": 123,
            "special_flag": True
        })

        # Test with mixed standard and custom parameters
        params = GeneratorCompletionParameters(
            temperature=0.7,
            max_tokens=100,
            custom_param="value",
            top_p=0.9,
            special_setting={"key": "value"}
        )
        self.assertEqual(params.temperature, 0.7)
        self.assertEqual(params.max_tokens, 100)
        self.assertEqual(params.kwargs["top_p"], 0.9)
        self.assertEqual(params.kwargs, {
            "custom_param": "value",
            "special_setting": {"key": "value"},
            "top_p": 0.9
        })

        # Test default values
        params = GeneratorCompletionParameters()
        self.assertEqual(params.temperature, .8)
        self.assertEqual(params.n, 1)
        self.assertEqual(params.kwargs, {})

        # Test with None values
        params = GeneratorCompletionParameters(
            max_tokens=None,
            top_k=None,
            quantile=None,
            custom_none=None
        )
        self.assertIsNone(params.max_tokens)
        self.assertIsNone(params.kwargs["top_k"])
        self.assertIsNone(params.kwargs["quantile"])
        self.assertIsNone(params.kwargs["custom_none"])

        # Test with complex nested structures in kwargs
        params = GeneratorCompletionParameters(
            temperature=0.7,
            complex_param={
                "nested": {
                    "list": [1, 2, 3],
                    "dict": {"a": 1, "b": 2}
                },
                "tuple": (1, 2, 3)
            }
        )
        self.assertEqual(params.temperature, 0.7)
        self.assertEqual(params.kwargs["complex_param"]["nested"]["list"], [1, 2, 3])
        self.assertEqual(params.kwargs["complex_param"]["nested"]["dict"], {"a": 1, "b": 2})
        self.assertEqual(params.kwargs["complex_param"]["tuple"], (1, 2, 3))

    @unittest.skipUnless(HUGGINGFACE_ACTIVATED, "HuggingFace not installed")
    def test_build_hf_model(self):
        gi = GeneratorInfo(
            model="gpt2",  # Use a small model for testing
            _format="hf",
        )
        
        gi.build_hf_model()
        
        self.assertIsNotNone(gi.hf_auto_model)
        self.assertIsNotNone(gi.hf_tokenizer)
        self.assertIsNotNone(gi.hf_processor)

if __name__ == '__main__':
    unittest.main()
