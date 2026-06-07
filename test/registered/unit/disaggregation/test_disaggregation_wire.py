import gc
import unittest

import numpy as np

from sglang.srt.disaggregation.common.utils import (
    group_concurrent_contiguous,
    pack_int_lists,
    pack_list_of_buffers,
    unpack_int_lists,
    unpack_list_of_buffers,
)
from sglang.srt.disaggregation.mooncake.conn import (
    MooncakeKVManager,
    _transfer_gc_guard,
)
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDisaggregationWire(unittest.TestCase):
    def test_int_lists_roundtrip(self):
        cases = [
            ("Q", [[1, 2, 3], [4]]),
            ("I", [[10, 20], [30, 40, 50]]),
            ("i", [[-1, 2], [3, -4, 5]]),
        ]
        for fmt, sample in cases:
            packed = pack_int_lists(sample, fmt)
            self.assertEqual(unpack_int_lists(packed, fmt), sample, msg=fmt)

    def test_pack_accepts_ndarray(self):
        arrs = [
            np.array([1, 2, 3], dtype=np.int32),
            np.array([4, 5], dtype=np.int32),
        ]
        packed = pack_int_lists(arrs, "i")
        self.assertEqual(unpack_int_lists(packed, "i"), [[1, 2, 3], [4, 5]])

    def test_empty_outer_list(self):
        self.assertEqual(pack_int_lists([], "Q"), b"")
        self.assertEqual(unpack_int_lists(b"", "Q"), [])

    def test_empty_inner_list(self):
        packed = pack_int_lists([[]], "I")
        self.assertEqual(unpack_int_lists(packed, "I"), [[]])

    def test_list_of_buffers_roundtrip(self):
        bufs = [b"abc", b"", b"de", b"x" * 17]
        self.assertEqual(unpack_list_of_buffers(pack_list_of_buffers(bufs)), bufs)


class TestGroupConcurrentContiguous(unittest.TestCase):
    @staticmethod
    def _arr(values):
        return np.array(values, dtype=np.int32)

    def test_single_contiguous_group(self):
        src = self._arr([10, 11, 12])
        dst = self._arr([5, 6, 7])
        self.assertEqual(
            group_concurrent_contiguous(src, dst),
            ([[10, 11, 12]], [[5, 6, 7]]),
        )

    def test_splits_on_discontiguous_indices(self):
        src = self._arr([10, 11, 20])
        dst = self._arr([5, 6, 7])
        self.assertEqual(
            group_concurrent_contiguous(src, dst),
            ([[10, 11], [20]], [[5, 6], [7]]),
        )

    def test_both_empty(self):
        self.assertEqual(
            group_concurrent_contiguous(self._arr([]), self._arr([])), ([], [])
        )

    def test_empty_src_nonempty_dst(self):
        self.assertEqual(
            group_concurrent_contiguous(self._arr([]), self._arr([1, 2])), ([], [])
        )

    def test_nonempty_src_empty_dst(self):
        # Regression: a non-empty source paired with an empty destination must not
        # raise a NumPy broadcast error (observed transferring DSA sparse-attention
        # state on a disaggregated GLM deployment when decode registered zero dst indices).
        self.assertEqual(
            group_concurrent_contiguous(self._arr([1, 2]), self._arr([])), ([], [])
        )

    def test_mismatched_nonempty_lengths_raise(self):
        with self.assertRaises(ValueError):
            group_concurrent_contiguous(self._arr([1, 2, 3]), self._arr([1, 2]))


class TestMooncakeTransferGCGuard(unittest.TestCase):
    def setUp(self):
        self._gc_was_enabled = gc.isenabled()
        if not self._gc_was_enabled:
            gc.enable()

    def tearDown(self):
        if not self._gc_was_enabled:
            gc.disable()

    def test_transfer_data_disables_gc_during_engine_call(self):
        class FakeEngine:
            def __init__(self):
                self.gc_enabled_during_call = None
                self.args = None

            def batch_transfer_sync(self, session_id, src_addrs, dst_addrs, lengths):
                self.gc_enabled_during_call = gc.isenabled()
                self.args = (session_id, src_addrs, dst_addrs, lengths)
                return 0

        mgr = object.__new__(MooncakeKVManager)
        mgr.engine = FakeEngine()

        with envs.SGLANG_MOONCAKE_DISABLE_GC_DURING_TRANSFER.override(True):
            ret = mgr._transfer_data("session", [(1, 2, 3), (4, 5, 6)])

        self.assertEqual(ret, 0)
        self.assertFalse(mgr.engine.gc_enabled_during_call)
        self.assertTrue(gc.isenabled())
        self.assertEqual(mgr.engine.args, ("session", [1, 4], [2, 5], [3, 6]))

    def test_transfer_gc_guard_is_ref_counted(self):
        with envs.SGLANG_MOONCAKE_DISABLE_GC_DURING_TRANSFER.override(True):
            with _transfer_gc_guard.suspend():
                self.assertFalse(gc.isenabled())
                with _transfer_gc_guard.suspend():
                    self.assertFalse(gc.isenabled())
                self.assertFalse(gc.isenabled())
            self.assertTrue(gc.isenabled())


if __name__ == "__main__":
    unittest.main()
