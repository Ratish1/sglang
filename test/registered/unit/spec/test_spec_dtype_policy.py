import unittest
from dataclasses import dataclass
from unittest.mock import MagicMock

import torch

from sglang.srt.model_executor.input_buffers import (
    ForwardInputBuffers,
    _forward_input_buffer_pool,
)
from sglang.srt.speculative.dtype_policy import (
    align_draft_input_embeds,
    build_speculative_dtype_policy,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestSpeculativeDtypePolicy(unittest.TestCase):
    def test_casts_target_hidden_states_to_draft_dtype_at_runtime(self):
        logger = MagicMock()
        policy = build_speculative_dtype_policy(
            target_dtype=torch.bfloat16,
            draft_dtype=torch.float16,
            logger=logger,
            algorithm="EAGLE3",
        )
        logger.warning.assert_not_called()

        hidden_states = torch.ones((2, 4), dtype=torch.bfloat16)
        prepared = policy.prepare_target_hidden_for_draft(hidden_states)

        self.assertEqual(prepared.dtype, torch.float16)
        self.assertEqual(hidden_states.dtype, torch.bfloat16)
        logger.warning.assert_called_once()

    def test_cast_logs_when_actual_hidden_dtype_differs_from_config(self):
        logger = MagicMock()
        policy = build_speculative_dtype_policy(
            target_dtype=torch.float16,
            draft_dtype=torch.float16,
            logger=logger,
            algorithm="EAGLE3",
        )
        logger.warning.assert_not_called()

        hidden_states = torch.ones((2, 4), dtype=torch.bfloat16)
        policy.prepare_target_hidden_for_draft(hidden_states)
        policy.prepare_target_hidden_for_draft(hidden_states)

        logger.warning.assert_called_once()

    def test_does_not_cast_when_boundary_is_disabled(self):
        logger = MagicMock()
        policy = build_speculative_dtype_policy(
            target_dtype=torch.bfloat16,
            draft_dtype=torch.float16,
            logger=logger,
            algorithm="STANDALONE",
            cast_target_hidden_states=False,
        )

        hidden_states = torch.ones((2, 4), dtype=torch.bfloat16)

        self.assertIs(
            policy.prepare_target_hidden_for_draft(hidden_states), hidden_states
        )
        logger.warning.assert_not_called()

    def test_aligns_shared_embedding_output_to_hidden_dtype(self):
        embeds = torch.ones((2, 4), dtype=torch.bfloat16)
        hidden_states = torch.ones((2, 4), dtype=torch.float16)

        prepared = align_draft_input_embeds(embeds, hidden_states)

        self.assertEqual(prepared.dtype, torch.float16)


@dataclass
class _PlainHiddenBuffers(ForwardInputBuffers):
    hidden_states: torch.Tensor


@dataclass
class _DraftHiddenBuffers(ForwardInputBuffers):
    hidden_states: torch.Tensor

    def _share_buffer_key(self, name: str) -> str:
        return f"draft.{name}"


@dataclass
class _DraftExtendHiddenBuffers(ForwardInputBuffers):
    hidden_states: torch.Tensor

    def _share_buffer_key(self, name: str) -> str:
        return f"draft_extend.{name}"


class TestForwardInputBufferKeys(unittest.TestCase):
    def setUp(self):
        _forward_input_buffer_pool.clear()

    def tearDown(self):
        _forward_input_buffer_pool.clear()

    def test_same_key_rejects_dtype_mismatch(self):
        _PlainHiddenBuffers(
            hidden_states=torch.empty((2, 4), dtype=torch.float16)
        ).share_buffers()

        with self.assertRaisesRegex(AssertionError, "different dtype"):
            _PlainHiddenBuffers(
                hidden_states=torch.empty((2, 4), dtype=torch.bfloat16)
            ).share_buffers()

    def test_namespaced_keys_allow_semantic_hidden_buffers(self):
        _DraftHiddenBuffers(
            hidden_states=torch.empty((2, 4), dtype=torch.float16)
        ).share_buffers()
        _DraftExtendHiddenBuffers(
            hidden_states=torch.empty((2, 8), dtype=torch.bfloat16)
        ).share_buffers()

        self.assertIn("draft.hidden_states", _forward_input_buffer_pool)
        self.assertIn("draft_extend.hidden_states", _forward_input_buffer_pool)


if __name__ == "__main__":
    unittest.main()
