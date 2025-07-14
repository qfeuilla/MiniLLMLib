"""Tests for the ChatNode class."""

import json
import os
import tempfile
import unittest
from unittest.mock import AsyncMock, Mock, patch

from minillmlib.core.chat_node import HUGGINGFACE_ACTIVATED, ChatNode, torch
from minillmlib.models.generator_info import GeneratorInfo
from minillmlib.utils.message_utils import NodeCompletionParameters


def make_mock_openai_response(content):
    """Helper to create a mock OpenAI response."""
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content=content))]
    return mock_response

def make_mock_anthropic_response(content):
    """Helper to create a mock Anthropic response."""
    mock_response = Mock()
    mock_response.content = [Mock(text=content)]
    return mock_response

class TestChatNode(unittest.IsolatedAsyncioTestCase):
    """Test ChatNode class."""

    async def test_collapse_thread_truncate(self):
        """Test collapse_thread with truncation marker (no summary)."""
        # Build a thread: root -> u1 -> a1 -> u2 -> a2 -> u3
        root = ChatNode(content="system", role="system")
        n1 = root.add_child(ChatNode(content="user1", role="user"))
        n2 = n1.add_child(ChatNode(content="assistant1", role="assistant"))
        n3 = n2.add_child(ChatNode(content="user2", role="user"))
        n4 = n3.add_child(ChatNode(content="assistant2", role="assistant"))
        n5 = n4.add_child(ChatNode(content="user3", role="user"))

        # Collapse keeping first 2 and last 2
        last, _ = await n5.collapse_thread(keep_last_n=2, keep_n=4, gi=None)
        # Walk the new thread
        thread = []
        node = last
        while node:
            thread.append(node)
            node = node._parent
        thread = thread[::-1]
        self.assertEqual(len(thread), 5)
        self.assertEqual(thread[0].content, "system")
        self.assertEqual(thread[1].content, "user1")
        # The marker node
        self.assertIn("truncated", thread[2].content.lower())
        self.assertEqual(thread[2].role, "assistant")
        self.assertEqual(thread[4].content, "user3")
        self.assertIsNone(thread[0]._parent)
        self.assertEqual(thread[-1].children, [])

    async def test_collapse_thread_summary(self):
        """Test collapse_thread with summarization (mocked complete_async)."""
        # Build a thread: root -> u1 -> a1 -> u2 -> a2 -> u3
        root = ChatNode(content="system", role="system")
        n1 = root.add_child(ChatNode(content="user1", role="user"))
        n2 = n1.add_child(ChatNode(content="assistant1", role="assistant"))
        n3 = n2.add_child(ChatNode(content="user2", role="user"))
        n4 = n3.add_child(ChatNode(content="assistant2", role="assistant"))
        n5 = n4.add_child(ChatNode(content="user3", role="user"))

        fake_summary = "This is a fake summary."
        gi = GeneratorInfo(model="test-model", _format="openai", api_key="test-key")
        # Patch complete_async to return a node with the fake summary
        with patch.object(ChatNode, "complete_async", new_callable=AsyncMock) as mock_complete_async:
            mock_complete_async.return_value = ChatNode(content='{"summary": "%s"}' % fake_summary, role="assistant")
            last, _ = await n5.collapse_thread(keep_last_n=2, keep_n=4, gi=gi)
        # Walk the new thread
        thread = []
        node = last
        while node:
            thread.append(node)
            node = node._parent
        thread = thread[::-1]
        self.assertEqual(len(thread), 5)
        self.assertEqual(thread[0].content, "system")
        self.assertEqual(thread[1].content, "user1")
        # The summary marker node
        self.assertIn("truncated", thread[2].content.lower())
        self.assertIn("summary", thread[2].content.lower())
        self.assertIn(fake_summary, thread[2].content)
        self.assertEqual(thread[2].role, "assistant")
        self.assertEqual(thread[3].content, "assistant2")
        self.assertEqual(thread[4].content, "user3")
        self.assertIsNone(thread[0]._parent)
        self.assertEqual(thread[-1].children, [])

    async def test_collapse_thread_edge_cases(self):
        """Test edge cases for collapse_thread."""
        # Helper to build a thread of given length
        def build_thread(length):
            root = ChatNode(content="system", role="system")
            prev = root
            for i in range(1, length):
                prev = prev.add_child(ChatNode(content=f"node{i}", role="user" if i % 2 else "assistant"))
            return root, prev

        # 1. keep_n >= thread length (should return full thread, no marker)
        root, last = build_thread(5)
        out, _ = await last.collapse_thread(keep_last_n=2, keep_n=10, gi=None)
        # Walk
        nodes = []
        node = out
        while node:
            nodes.append(node)
            node = node._parent
        nodes = nodes[::-1]
        self.assertEqual(len(nodes), 5)
        self.assertNotIn("truncated", "".join(n.content for n in nodes).lower())

        # 2. keep_last_n >= thread length (should return full thread, no marker)
        root, last = build_thread(5)
        out, _ = await last.collapse_thread(keep_last_n=5, keep_n=2, gi=None)
        nodes = []
        node = out
        while node:
            nodes.append(node)
            node = node._parent
        nodes = nodes[::-1]
        self.assertEqual(len(nodes), 5)
        self.assertNotIn("truncated", "".join(n.content for n in nodes).lower())

        # 3. keep_n < 2 (should keep at least root and last, or just last)
        root, last = build_thread(5)
        out, _ = await last.collapse_thread(keep_last_n=1, keep_n=1, gi=None)
        nodes = []
        node = out
        while node:
            nodes.append(node)
            node = node._parent
        nodes = nodes[::-1]
        self.assertGreaterEqual(len(nodes), 1)
        # If only last node is kept, should be user node
        self.assertEqual(nodes[-1].content, "node4")

        # 4. Thread of length 1 (single node)
        root = ChatNode(content="root", role="system")
        out, _ = await root.collapse_thread(keep_last_n=1, keep_n=1, gi=None)
        self.assertEqual(out.content, "root")
        self.assertIsNone(out._parent)
        self.assertEqual(out.children, [])

        # 5. keep_last_n = 0 (should keep only first nodes)
        root, last = build_thread(5)
        out, _ = await last.collapse_thread(keep_last_n=0, keep_n=2, gi=None)
        nodes = []
        node = out
        while node:
            nodes.append(node)
            node = node._parent
        nodes = nodes[::-1]
        self.assertEqual(len(nodes), 3)
        self.assertEqual(nodes[0].content, "system")
        self.assertEqual(nodes[1].content, "node1")
        self.assertIn("truncated", nodes[2].content.lower())

        # 6. keep_n = 0 (should fallback to deepcopy of self)
        root, last = build_thread(5)
        out, _ = await last.collapse_thread(keep_last_n=0, keep_n=0, gi=None)
        self.assertIsNone(out._parent)
        self.assertEqual(out.children, [])

    def test_detach_child(self):
        """Test detaching a child node from its parent."""
        parent_node = ChatNode(content="Parent", role="system")
        child_node = ChatNode(content="Child", role="user")

        parent_node.add_child(child_node)

        # Assert initial state
        self.assertIn(child_node, parent_node.children)
        self.assertEqual(child_node._parent, parent_node)  # pylint: disable=protected-access

        # Detach the child
        detached_node = child_node.detach()

        # Assert final state
        self.assertNotIn(child_node, parent_node.children)
        self.assertIsNone(child_node._parent)  # pylint: disable=protected-access
        self.assertEqual(detached_node, child_node)

    async def asyncSetUp(self):
        self.gi = GeneratorInfo(
            model="test-model", _format="openai", api_key="test-key"
        )
        self.chat_node = ChatNode(content="Hello", role="user")

    def test_init_with_format_kwargs(self):
        """Test initialization with format kwargs."""
        node = ChatNode(
            content="Hello {name}!",
            role="user",
            format_kwargs={"name": "Alice", "unused": "value"},
        )
        self.assertEqual(node.format_kwargs, {"name": "Alice"})
        self.assertEqual(node.role, "user")

    def test_init_invalid_role(self):
        """Test initialization with invalid role."""
        with self.assertRaises(ValueError):
            ChatNode(content="Hello", role="invalid_role")

    def test_complete_one_anthropic(self):
        """Test Anthropic completion with system message."""
        mock_client = Mock()
        mock_response = make_mock_anthropic_response("Test response")
        mock_client.messages.create.return_value = mock_response

        gi = GeneratorInfo(model="claude-3", _format="anthropic", api_key="test-key")
        system_node = ChatNode(content="System prompt", role="system")
        user_node = system_node.add_child(ChatNode(content="User message", role="user"))

        with patch("minillmlib.core.chat_node.Anthropic", return_value=mock_client):
            params = NodeCompletionParameters(gi=gi, add_child=True)
            result = user_node.complete_one(params)
            self.assertEqual(result.content, "Test response")
            self.assertEqual(result.role, "assistant")
            mock_client.messages.create.assert_called_once()
            call_kwargs = mock_client.messages.create.call_args.kwargs
            self.assertEqual(call_kwargs["system"], "System prompt")

    def test_complete_with_retry(self):
        """Test completion with retry on error."""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = [
            Exception("API Error"),
            make_mock_openai_response("Test response"),
        ]
        with patch("minillmlib.core.chat_node.OpenAI", return_value=mock_client):
            params = NodeCompletionParameters(gi=self.gi, add_child=True, retry=1)
            result = self.chat_node.complete_one(params)
            self.assertEqual(result.content, "Test response")
            self.assertEqual(mock_client.chat.completions.create.call_count, 2)

    def test_get_messages_merge_contiguous(self):
        """Test message merging with different roles."""
        system = ChatNode(content="System message", role="system")
        user1 = system.add_child(ChatNode(content="User message 1", role="user"))
        user2 = user1.add_child(ChatNode(content="User message 2", role="user"))
        assistant = user2.add_child(
            ChatNode(content="Assistant message", role="assistant")
        )
        messages = assistant.get_messages(self.gi, merge_contiguous="all")
        self.assertEqual(len(messages), 3)
        self.assertEqual(messages[1]["content"], "User message 1\nUser message 2")
        messages = assistant.get_messages(self.gi, merge_contiguous="user")
        self.assertEqual(len(messages), 3)
        self.assertEqual(messages[1]["content"], "User message 1\nUser message 2")

    def test_tree_manipulation(self):
        """Test tree structure operations."""
        child1 = self.chat_node.add_child(
            ChatNode(content="First response", role="assistant")
        )
        child2 = child1.add_child(ChatNode(content="Second message", role="user"))
        self.assertEqual(child1._parent, self.chat_node)  # pylint: disable=protected-access
        self.assertEqual(child2._parent, child1)  # pylint: disable=protected-access
        self.assertEqual(len(self.chat_node.children), 1)
        self.assertEqual(child2.get_root(), self.chat_node)
        self.assertEqual(self.chat_node.get_child([0, 0]), child2)
        self.assertEqual(self.chat_node.get_child([0]), child1)
        self.assertEqual(self.chat_node.get_last_child(), child2)

    async def test_complete_async(self):
        """Test async completion."""
        mock_client = AsyncMock()
        mock_response = make_mock_openai_response("Test response")
        mock_client.chat.completions.create.return_value = mock_response
        with patch("minillmlib.core.chat_node.AsyncOpenAI", return_value=mock_client):
            params = NodeCompletionParameters(gi=self.gi, add_child=True)
            result = await self.chat_node.complete_one_async(params)
            self.assertEqual(result.content, "Test response")
            self.assertEqual(result.role, "assistant")

    def test_save_load_thread(self):
        """Test saving and loading conversation thread."""
        child1 = self.chat_node.add_child(
            ChatNode(
                content="Response with {var}",
                role="assistant",
                format_kwargs={"var": "value"},
            )
        )
        child2 = child1.add_child(ChatNode(content="User follow-up", role="user"))
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            child2.save_thread(f.name)
            loaded_thread = ChatNode.from_thread(f.name).get_root()
        self.assertEqual(loaded_thread.content, "Hello")
        self.assertEqual(loaded_thread.children[0].content, "Response with {var}")
        self.assertEqual(loaded_thread.children[0].format_kwargs["var"], "value")
        self.assertEqual(
            loaded_thread.children[0].children[0].content, "User follow-up"
        )
        os.unlink(f.name)

    def test_format_kwargs_propagation(self):
        """Test format kwargs propagation through the tree."""
        _format_kwargs = {"name": "Alice", "weather": "sunny"}
        root = ChatNode(
            content="Hello {name}", role="user", format_kwargs=_format_kwargs
        )
        child = root.add_child(
            ChatNode(
                content="Hi {name}, how's {weather}?",
                role="assistant",
                format_kwargs=_format_kwargs,
            )
        )
        self.assertEqual(
            root.get_root().format_kwargs, {"name": "Alice", "weather": "sunny"}
        )
        child.update_format_kwargs(propagate=True, name="Bob", weather="rainy")
        messages = child.get_messages(self.gi)
        self.assertIn("Hello Bob", messages[0]["content"])
        self.assertIn("Hi Bob, how's rainy?", messages[1]["content"])

    @unittest.skipUnless(HUGGINGFACE_ACTIVATED, "HuggingFace not installed")
    def test_hf_compute_logits(self):
        """Test HuggingFace logits computation."""
        gi = GeneratorInfo(model="gpt2", _format="hf")
        gi.build_hf_model()
        node = ChatNode(content="Hello world", role="user")
        logits = node.hf_compute_logits(gi=gi)
        self.assertIsNotNone(logits)
        avg_logits = node.hf_compute_logits_average(
            gi=gi, quantile=0.5, repeat_penalty=True
        )
        self.assertIsInstance(avg_logits, torch.Tensor)

    def test_complete_one(self):
        """Test completion with OpenAI."""
        mock_client = Mock()
        mock_response = make_mock_openai_response("Test response")
        mock_client.chat.completions.create.return_value = mock_response
        with patch("minillmlib.core.chat_node.OpenAI", return_value=mock_client):
            params = NodeCompletionParameters(gi=self.gi, add_child=True)
            result = self.chat_node.complete_one(params)
            self.assertEqual(result.content, "Test response")
            self.assertEqual(result.role, "assistant")
            self.assertEqual(len(self.chat_node.children), 1)
            mock_client.chat.completions.create.assert_called_once()
            call_kwargs = mock_client.chat.completions.create.call_args.kwargs
            self.assertEqual(call_kwargs["model"], "test-model")
            self.assertIn("messages", call_kwargs)

    def test_get_messages(self):
        """Test getting messages from a chat node."""
        child = self.chat_node.add_child(
            ChatNode(content="How are you?", role="assistant")
        )
        child = child.add_child(ChatNode(content="I'm good!", role="user"))

        messages = child.get_messages(self.gi)

        self.assertEqual(len(messages), 3)
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[0]["content"], "Hello")
        self.assertEqual(messages[1]["role"], "assistant")
        self.assertEqual(messages[1]["content"], "How are you?")
        self.assertEqual(messages[2]["role"], "user")
        self.assertEqual(messages[2]["content"], "I'm good!")

    def test_complete_one_with_error(self):
        """Test error handling during completion."""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")

        with patch("minillmlib.core.chat_node.OpenAI", return_value=mock_client):
            params = NodeCompletionParameters(
                gi=self.gi,
                add_child=True,
                retry=0,
            )

            with self.assertRaises(Exception):
                self.chat_node.complete_one(params)

            # Verify no child was added due to error
            self.assertEqual(len(self.chat_node.children), 0)

    def test_complete_one_no_add_child(self):
        """Test completion without adding child."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Test response"))]
        mock_client.chat.completions.create.return_value = mock_response

        with patch("minillmlib.core.chat_node.OpenAI", return_value=mock_client):
            params = NodeCompletionParameters(gi=self.gi, add_child=False)

            result = self.chat_node.complete_one(params)

            self.assertEqual(result.content, "Test response")
            self.assertEqual(len(self.chat_node.children), 0)

    def test_complete_multiple(self):
        """Test completing multiple responses."""
        mock_client = Mock()
        mock_response1 = Mock()
        mock_response2 = Mock()
        mock_response1.choices = [Mock(message=Mock(content="Response 1"))]
        mock_response2.choices = [Mock(message=Mock(content="Response 2"))]
        mock_client.chat.completions.create.side_effect = [
            mock_response1,
            mock_response2,
        ]

        with patch("minillmlib.core.chat_node.OpenAI", return_value=mock_client):
            params = NodeCompletionParameters(gi=self.gi, add_child=True, n=2)

            results = self.chat_node.complete(params)

            self.assertIsInstance(results, list)
            self.assertEqual(len(results), 2)
            self.assertEqual(results[0].content, "Response 1")
            self.assertEqual(results[1].content, "Response 2")

    def test_complete_with_force_prepend(self):
        """Test completion with force_prepend."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="additional response"))]
        mock_client.chat.completions.create.return_value = mock_response

        with patch("minillmlib.core.chat_node.OpenAI", return_value=mock_client):
            params = NodeCompletionParameters(
                gi=self.gi, add_child=True, force_prepend="Prepended text "
            )

            result = self.chat_node.complete_one(params)

            self.assertEqual(result.content, "Prepended text additional response")

    def test_complete_with_json_parsing(self):
        """Test completion with JSON parsing."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content='{"key": "value"}'))]
        mock_client.chat.completions.create.return_value = mock_response

        with patch("minillmlib.core.chat_node.OpenAI", return_value=mock_client):
            params = NodeCompletionParameters(
                gi=self.gi, add_child=True, parse_json=True
            )

            result = self.chat_node.complete_one(params)

            self.assertEqual(result.content, '{"key": "value"}')

    def test_complete_with_json_parsing_error(self):
        """Test completion with JSON parsing error."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="invalid json"))]
        mock_client.chat.completions.create.return_value = mock_response

        with patch(
            "minillmlib.core.chat_node.OpenAI", return_value=mock_client
        ) and patch("minillmlib.core.chat_node.time.sleep"):
            params = NodeCompletionParameters(
                gi=self.gi, add_child=True, parse_json=True, crash_on_refusal=True
            )

            with self.assertRaises(Exception):
                self.chat_node.complete_one(params)

    def test_save_loom(self):
        """Test saving conversation as a loom (tree structure)."""
        root = ChatNode(content="Root", role="system")
        child1 = root.add_child(ChatNode(content="Child 1", role="user"))
        child2 = root.add_child(ChatNode(content="Child 2", role="user"))
        child1.add_child(ChatNode(content="Child 1.1", role="assistant"))
        child2.add_child(ChatNode(content="Child 2.1", role="assistant"))

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            root.save_loom(f.name)

            # Verify the saved structure
            with open(f.name, "r", encoding="utf-8") as saved:
                data = json.load(saved)
                self.assertEqual(data["content"], "Root")
                self.assertEqual(len(data["children"]), 2)
                self.assertEqual(data["children"][0]["content"], "Child 1")
                self.assertEqual(data["children"][1]["content"], "Child 2")

        os.unlink(f.name)

    async def test_complete_multiple_async(self):
        """Test async completion of multiple responses."""
        mock_client = AsyncMock()
        mock_response1 = Mock()
        mock_response2 = Mock()
        mock_response1.choices = [Mock(message=Mock(content="Async Response 1"))]
        mock_response2.choices = [Mock(message=Mock(content="Async Response 2"))]
        mock_client.chat.completions.create.side_effect = [
            mock_response1,
            mock_response2,
        ]

        with patch("minillmlib.core.chat_node.AsyncOpenAI", return_value=mock_client):
            params = NodeCompletionParameters(gi=self.gi, add_child=True, n=2)

            results = await self.chat_node.complete_async(params)

            self.assertIsInstance(results, list)
            self.assertEqual(len(results), 2)
            self.assertNotEqual(results[0].content, results[1].content)
            self.assertIn(results[0].content, ["Async Response 1", "Async Response 2"])

    def test_merge_trees(self):
        """Test merging two conversation trees."""
        tree1 = ChatNode(content="Root 1", role="system")
        tree1.add_child(ChatNode(content="Child 1", role="user"))

        tree2 = ChatNode(content="Root 2", role="system")
        tree2.add_child(ChatNode(content="Child 2", role="user"))

        # Merge tree2 into tree1
        merged = tree1.merge(tree2)

        self.assertEqual(merged.content, "Root 2")
        self.assertEqual(merged.get_root().content, "Root 1")
        self.assertEqual(len(tree1.children), 2)

    def test_illegitimate_child(self):
        """
        Test adding an illegitimate child (child knows _parent but _parent doesn't know child).
        """
        _parent = ChatNode(content="_parent", role="user")
        child = ChatNode(content="Child", role="assistant")

        _parent.add_child(child, illegitimate=True)

        self.assertEqual(child._parent, _parent)  # pylint: disable=protected-access
        self.assertEqual(len(_parent.children), 0)

    def test_complete_with_back_off_disabled(self):
        """Test completion with back_off disabled for JSON parsing."""
        mock_client = Mock()
        mock_response1 = Mock()
        mock_response2 = Mock()
        mock_response1.choices = [Mock(message=Mock(content="invalid json"))]
        mock_response2.choices = [Mock(message=Mock(content='{"key": "value"}'))]
        mock_client.chat.completions.create.side_effect = [
            mock_response1,
            mock_response2,
        ]

        with (
            patch("minillmlib.core.chat_node.OpenAI", return_value=mock_client),
            patch("minillmlib.core.chat_node.time.sleep") as mock_sleep,
        ):  # Mock sleep to speed up tests
            params = NodeCompletionParameters(
                gi=self.gi,
                add_child=True,
                parse_json=True,
                crash_on_refusal=True,
                retry=1,
            )

            result = self.chat_node.complete_one(params)
            self.assertEqual(result.content, '{"key": "value"}')
            mock_sleep.assert_not_called()  # Should not sleep if the crash is due to JSON

    def test_complete_with_exp_back_off(self):
        """Test completion with exponential back_off."""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = [
            Exception("API Error 1"),
            Exception("API Error 2"),
            Mock(choices=[Mock(message=Mock(content="Success"))]),
        ]

        with (
            patch("minillmlib.core.chat_node.OpenAI", return_value=mock_client),
            patch("minillmlib.core.chat_node.time.sleep") as mock_sleep,
        ):
            params = NodeCompletionParameters(
                gi=self.gi, add_child=True, retry=2, exp_back_off=True
            )

            result = self.chat_node.complete_one(params)

            self.assertEqual(result.content, "Success")

            # Should have called sleep with increasing durations
            self.assertEqual(mock_sleep.call_count, 2)
            calls = mock_sleep.call_args_list
            self.assertLess(
                calls[0][0][0], calls[1][0][0]
            )  # Second sleep should be longer

    def test_complete_with_force_merge(self):
        """Test completion with force_merge enabled."""
        gi = GeneratorInfo(
            model="test-model", _format="openai", api_key="test-key", force_merge=True
        )

        system = ChatNode(content="System 1", role="system")
        system2 = system.add_child(ChatNode(content="System 2", role="system"))
        user = system2.add_child(ChatNode(content="User message", role="user"))

        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Response"))]
        mock_client.chat.completions.create.return_value = mock_response

        with patch("minillmlib.core.chat_node.OpenAI", return_value=mock_client):
            params = NodeCompletionParameters(gi=gi, add_child=True)
            user.complete_one(params)

            # Verify that the messages were merged
            call_kwargs = mock_client.chat.completions.create.call_args.kwargs
            messages = call_kwargs["messages"]
            self.assertEqual(len(messages), 2)  # Should merge system messages
            self.assertEqual(messages[0]["content"], "System 1\nSystem 2")

    def test_complete_with_no_system(self):
        """Test completion with no_system enabled."""
        gi = GeneratorInfo(
            model="test-model", _format="openai", api_key="test-key", no_system=True
        )

        system = ChatNode(content="System message", role="system")
        user = system.add_child(ChatNode(content="User message", role="user"))

        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Response"))]
        mock_client.chat.completions.create.return_value = mock_response

        with patch("minillmlib.core.chat_node.OpenAI", return_value=mock_client):
            params = NodeCompletionParameters(gi=gi, add_child=True)
            user.complete_one(params)

            # Verify that system messages were converted to user messages
            call_kwargs = mock_client.chat.completions.create.call_args.kwargs
            messages = call_kwargs["messages"]
            self.assertEqual(messages[0]["role"], "user")

    def test_complete_with_prettify_format(self):
        """Test completion with prettify format."""
        gi = GeneratorInfo(
            model="test-model",
            _format="prettify",
            api_key="test-key",
            translation_table={
                "user": "Human: ",
                "assistant": "Assistant: ",
                "system": "System: ",
                "base": "Base: ",
            },
        )

        system = ChatNode(content="Be helpful", role="system")
        user = system.add_child(ChatNode(content="Hello", role="user"))
        assistant = user.add_child(ChatNode(content="Hi!", role="assistant"))

        messages = assistant.get_messages(gi)
        self.assertIsInstance(messages, str)
        self.assertEqual(messages, "System: Be helpfulHuman: HelloAssistant: Hi!")

    def test_complete_with_force_prepend_and_json(self):
        """Test completion with both force_prepend and JSON parsing."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content='{"result": "value"}'))]
        mock_client.chat.completions.create.return_value = mock_response

        with patch("minillmlib.core.chat_node.OpenAI", return_value=mock_client):
            params = NodeCompletionParameters(
                gi=self.gi, add_child=True, parse_json=True, force_prepend="JSON: "
            )

            result = self.chat_node.complete_one(params)
            self.assertEqual(result.content, '{"result": "value"}')

            # Verify that force_prepend was used in the API call
            messages = mock_client.chat.completions.create.call_args.kwargs["messages"]
            self.assertTrue(any("JSON: " in msg["content"] for msg in messages))

    def test_complete_with_empty_json_response(self):
        """Test handling of empty JSON responses."""
        mock_client = Mock()
        mock_responses = [
            Mock(choices=[Mock(message=Mock(content="{}"))]),
            Mock(choices=[Mock(message=Mock(content='""'))]),
            Mock(choices=[Mock(message=Mock(content=""))]),
        ]
        mock_client.chat.completions.create.side_effect = mock_responses

        with patch(
            "minillmlib.core.chat_node.OpenAI", return_value=mock_client
        ) and patch("minillmlib.core.chat_node.time.sleep"):
            params = NodeCompletionParameters(
                gi=self.gi, add_child=True, parse_json=True, crash_on_refusal=True
            )

            # Test that all empty JSON variants raise an exception
            for _ in range(3):
                with self.assertRaises(Exception):
                    self.chat_node.complete_one(params)

    def test_complete_with_invalid_format(self):
        """Test completion with invalid format."""
        gi = GeneratorInfo(
            model="test-model", _format="invalid_format", api_key="test-key"
        )

        with self.assertRaises(NotImplementedError):
            params = NodeCompletionParameters(gi=gi, add_child=True)
            self.chat_node.complete_one(params)

    def test_get_messages_with_invalid_merge(self):
        """Test get_messages with invalid merge_contiguous value."""
        with self.assertRaises(ValueError):
            self.chat_node.get_messages(self.gi, merge_contiguous="invalid")

    def test_complete_with_custom_back_off_time(self):
        """Test completion with custom back_off time."""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = [
            Exception("API Error"),
            Mock(choices=[Mock(message=Mock(content="Success"))]),
        ]

        with (
            patch("minillmlib.core.chat_node.OpenAI", return_value=mock_client),
            patch("minillmlib.core.chat_node.time.sleep") as mock_sleep,
        ):
            params = NodeCompletionParameters(
                gi=self.gi,
                add_child=True,
                retry=1,
                back_off_time=10.0,  # Custom back_off time
            )

            result = self.chat_node.complete_one(params)
            self.assertEqual(result.content, "Success")

            # Verify that sleep was called with the custom back_off time
            mock_sleep.assert_called_once_with(10.0)

    def test_complete_with_exp_back_off_and_custom_time(self):
        """Test exponential back_off with custom base time."""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = [
            Exception("API Error 1"),
            Exception("API Error 2"),
            Mock(choices=[Mock(message=Mock(content="Success"))]),
        ]

        with (
            patch("minillmlib.core.chat_node.OpenAI", return_value=mock_client),
            patch("minillmlib.core.chat_node.time.sleep") as mock_sleep,
        ):
            params = NodeCompletionParameters(
                gi=self.gi,
                add_child=True,
                retry=2,
                exp_back_off=True,
                back_off_time=2.0,  # Custom base time for exp back_off
            )

            result = self.chat_node.complete_one(params)
            self.assertEqual(result.content, "Success")

            # Verify exponential back_off with custom base time
            calls = mock_sleep.call_args_list
            self.assertEqual(len(calls), 2)
            self.assertEqual(calls[0][0][0], 2.0)  # First retry: 2.0
            self.assertEqual(calls[1][0][0], 4.0)  # Second retry: 2.0 * 2

    def test_complete_with_zero_back_off_time(self):
        """Test completion with zero back_off time."""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = [
            Exception("API Error"),
            Mock(choices=[Mock(message=Mock(content="Success"))]),
        ]

        with (
            patch("minillmlib.core.chat_node.OpenAI", return_value=mock_client),
            patch("minillmlib.core.chat_node.time.sleep") as mock_sleep,
        ):
            params = NodeCompletionParameters(
                gi=self.gi,
                add_child=True,
                retry=1,
                back_off_time=0.0,  # No delay between retries
            )

            result = self.chat_node.complete_one(params)
            self.assertEqual(result.content, "Success")

            # Verify that sleep was called with zero time
            mock_sleep.assert_called_once_with(0.0)

    def test_error_handling_behavior(self):
        """Test different error handling behaviors."""
        mock_client = Mock()

        # Test API error (should use back_off)
        mock_client.chat.completions.create.side_effect = [
            Exception("API Error"),
            Mock(choices=[Mock(message=Mock(content="Success"))]),
        ]

        with (
            patch("minillmlib.core.chat_node.OpenAI", return_value=mock_client),
            patch("minillmlib.core.chat_node.time.sleep") as mock_sleep,
        ):
            params = NodeCompletionParameters(
                gi=self.gi, add_child=True, retry=1, back_off_time=5.0
            )

            result = self.chat_node.complete_one(params)
            self.assertEqual(result.content, "Success")
            mock_sleep.assert_called_once_with(5.0)  # Should back_off for API error
            mock_sleep.reset_mock()

            # Test JSON parsing error (should not use back_off)
            mock_client.chat.completions.create.side_effect = [
                Mock(choices=[Mock(message=Mock(content="invalid json"))]),
                Mock(choices=[Mock(message=Mock(content='{"key": "value"}'))]),
            ]

            params = NodeCompletionParameters(
                gi=self.gi,
                add_child=True,
                parse_json=True,
                crash_on_refusal=True,
                retry=1,
                back_off_time=5.0,
            )

            result = self.chat_node.complete_one(params)
            self.assertEqual(result.content, '{"key": "value"}')
            mock_sleep.assert_not_called()  # Should not back_off for JSON error

    def test_complete_with_anthropic_format(self):
        """Test completion with Anthropic format."""
        gi = GeneratorInfo(model="claude-2", _format="anthropic", api_key="test-key")

        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock(text="Test response")]
        mock_client.messages.create.return_value = mock_response

        with (
            patch("minillmlib.core.chat_node.Anthropic", return_value=mock_client),
            patch("minillmlib.core.chat_node.time.sleep"),
        ):
            params = NodeCompletionParameters(gi=gi, add_child=True)
            result = self.chat_node.complete_one(params)

            self.assertEqual(result.content, "Test response")
            # Verify Anthropic message format
            call_kwargs = mock_client.messages.create.call_args.kwargs
            self.assertIn("messages", call_kwargs)

    def test_complex_message_merging(self):
        """Test complex message merging scenarios."""
        gi = GeneratorInfo(
            model="test-model", _format="openai", api_key="test-key", force_merge=True
        )

        # Create a complex conversation tree with multiple system messages
        root = ChatNode(content="System 1", role="system")
        branch1 = root.add_child(ChatNode(content="System 2", role="system"))
        branch2 = root.add_child(ChatNode(content="Alt System", role="system"))

        user1 = branch1.add_child(ChatNode(content="User 1", role="user"))
        user2 = branch2.add_child(ChatNode(content="User 2", role="user"))

        # Add some assistant responses
        asst1 = user1.add_child(ChatNode(content="Asst 1", role="assistant"))
        asst2 = user2.add_child(ChatNode(content="Asst 2", role="assistant"))

        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Response"))]
        mock_client.chat.completions.create.return_value = mock_response

        with (
            patch("minillmlib.core.chat_node.OpenAI", return_value=mock_client),
            patch("minillmlib.core.chat_node.time.sleep"),
        ):
            # Test merging from different branches
            params = NodeCompletionParameters(gi=gi, add_child=True)
            asst1.complete_one(params)
            asst2.complete_one(params)

            # Verify message merging
            calls = mock_client.chat.completions.create.call_args_list
            self.assertEqual(len(calls), 2)

            # First branch should have merged system messages
            messages1 = calls[0].kwargs["messages"]
            self.assertTrue(
                any("System 1\nSystem 2" in msg["content"] for msg in messages1)
            )

            # Second branch should have different merged system messages
            messages2 = calls[1].kwargs["messages"]
            self.assertTrue(
                any("System 1\nAlt System" in msg["content"] for msg in messages2)
            )

    def test_thread_serialization_edge_cases(self):
        """Test thread serialization edge cases."""
        # Create a complex tree with various special characters and empty messages
        root = ChatNode(content="", role="system")  # Empty content
        special = root.add_child(
            ChatNode(content="Special\nchars:\n\t\r\n\"'\\", role="user")
        )
        unicode_node = special.add_child(
            ChatNode(content="Unicode: 🌟🔥🚀", role="assistant")
        )
        very_long = unicode_node.add_child(ChatNode(content="x" * 1000, role="user"))

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            very_long.save_thread(f.name)
            loaded_thread = ChatNode.from_thread(f.name)

            # Verify all nodes were preserved exactly
            current_orig = very_long
            current_loaded = loaded_thread
            while current_orig is not None:
                self.assertEqual(current_orig.content, current_loaded.content)
                self.assertEqual(current_orig.role, current_loaded.role)
                current_orig = current_orig._parent  # pylint: disable=protected-access
                current_loaded = current_loaded._parent  # pylint: disable=protected-access

    def test_different_api_response_formats(self):
        """Test handling of different API response formats."""
        mock_client = Mock()

        # Test OpenAI format
        mock_response_openai = Mock()
        mock_response_openai.choices = [Mock(message=Mock(content="OpenAI Response"))]

        # Test Anthropic format
        mock_response_anthropic = Mock()
        mock_response_anthropic.content = [Mock(text="Anthropic Response")]

        with (
            patch("minillmlib.core.chat_node.time.sleep"),
            patch("minillmlib.core.chat_node.OpenAI", return_value=mock_client),
            patch("minillmlib.core.chat_node.Anthropic", return_value=mock_client),
        ):

            # Test OpenAI format
            mock_client.chat.completions.create.return_value = mock_response_openai
            gi_openai = GeneratorInfo(model="gpt-4", _format="openai", api_key="test")
            result = self.chat_node.complete_one(
                NodeCompletionParameters(gi=gi_openai, add_child=True)
            )
            self.assertEqual(result.content, "OpenAI Response")

            # Test Anthropic format
            mock_client.messages.create.return_value = mock_response_anthropic
            gi_anthropic = GeneratorInfo(
                model="claude-2", _format="anthropic", api_key="test"
            )
            result = self.chat_node.complete_one(
                NodeCompletionParameters(gi=gi_anthropic, add_child=True)
            )
            self.assertEqual(result.content, "Anthropic Response")

    def test_enforce_json_compatible_prompt(self):
        """Test GeneratorInfo with enforce_json_compatible_prompt enabled."""
        gi = GeneratorInfo(
            model="test-model",
            _format="openai",
            api_key="test-key",
            enforce_json_compatible_prompt=True,
        )

        # Create a chat node with content that might break JSON
        node = ChatNode(content='Hello "world" with {special} chars!', role="user")

        # Get messages and verify they're properly escaped
        messages = node.get_messages(gi=gi)
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(
            messages[0]["content"], 'Hello \\"world\\" with {special} chars!'
        )

        # Test with multiple messages
        child = node.add_child(
            ChatNode(
                content='Response with "quotes" and \\backslashes\\', role="assistant"
            )
        )
        messages = child.get_messages(gi=gi)
        self.assertEqual(len(messages), 2)
        self.assertEqual(
            messages[1]["content"], 'Response with \\"quotes\\" and \\\\backslashes\\\\'
        )

    def test_from_thread_with_multiple_paths(self):
        """Test from_thread with multiple paths."""
        # Create two temporary thread files
        with (
            tempfile.NamedTemporaryFile(mode="w", delete=False) as f1,
            tempfile.NamedTemporaryFile(mode="w", delete=False) as f2,
        ):

            # First thread
            json.dump(
                {
                    "required_kwargs": {"var1": "value1"},
                    "prompts": [
                        {"role": "system", "content": "System {var1} prompt 1"},
                        {"role": "user", "content": "User message 1"},
                    ],
                },
                f1,
            )
            f1.flush()  # Ensure content is written to disk

            # Second thread
            json.dump(
                {
                    "required_kwargs": {"var2": "value2"},
                    "prompts": [
                        {"role": "assistant", "content": "Assistant response 1"},
                        {"role": "user", "content": "User {var2} message 2"},
                        {"role": "assistant", "content": "Assistant response 2"},
                    ],
                },
                f2,
            )
            f2.flush()  # Ensure content is written to disk

            # Test loading multiple threads
            thread_paths = [f1.name, f2.name]
            root = ChatNode.from_thread(thread_paths).get_root()

            # Verify the structure
            self.assertEqual(root.role, "system")
            self.assertEqual(root.content, "System {var1} prompt 1")
            self.assertEqual(root.format_kwargs, {"var1": "value1", "var2": "value2"})

            child1 = root.children[0]
            self.assertEqual(child1.role, "user")
            self.assertEqual(child1.content, "User message 1")
            self.assertEqual(child1.format_kwargs, {})

            child2 = child1.children[0]
            self.assertEqual(child2.role, "assistant")
            self.assertEqual(child2.content, "Assistant response 1")
            self.assertEqual(child2.format_kwargs, {"var2": "value2"})

            child3 = child2.children[0]
            self.assertEqual(child3.role, "user")
            self.assertEqual(child3.content, "User {var2} message 2")
            self.assertEqual(child3.format_kwargs, {"var2": "value2"})

            child4 = child3.children[0]
            self.assertEqual(child4.role, "assistant")
            self.assertEqual(child4.content, "Assistant response 2")
            self.assertEqual(child4.format_kwargs, {})

            # Test with empty list
            with self.assertRaises(ValueError):
                ChatNode.from_thread([])

            # Clean up
            os.unlink(f1.name)
            os.unlink(f2.name)


if __name__ == "__main__":
    unittest.main()
