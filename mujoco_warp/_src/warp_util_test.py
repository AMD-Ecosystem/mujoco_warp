# Copyright 2026 The Newton Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Tests for Warp utilities."""

import types
import warnings
from unittest import mock

import warp as wp
from absl.testing import absltest
from absl.testing import parameterized

from mujoco_warp._src import warp_util


@warp_util.event_scope
def _inner():
  wp.zeros(16, dtype=float)


@warp_util.event_scope
def _outer():
  _inner()
  _inner()


class EventTracerTest(absltest.TestCase):
  """Tests for EventTracer and event_scope."""

  def setUp(self):
    super().setUp()
    if not wp.get_device().is_cuda:
      self.skipTest("event timing requires a GPU device")

  def test_untraced_call_passes_through(self):
    @warp_util.event_scope
    def fn(a, b=0):
      return a + b

    self.assertEqual(fn(1, b=2), 3)
    self.assertIsNone(warp_util._STACK)

  def test_trace_nested_scopes(self):
    with warp_util.EventTracer() as tracer:
      _outer()
      _outer()
      trace = tracer.trace()

    self.assertEqual(set(trace), {"_outer"})
    outer_times, outer_sub = trace["_outer"]
    self.assertLen(outer_times, 2)
    self.assertEqual(set(outer_sub), {"_inner"})
    inner_times, inner_sub = outer_sub["_inner"]
    # two _outer calls, each calling _inner twice
    self.assertLen(inner_times, 4)
    self.assertEqual(inner_sub, {})
    for t in outer_times + inner_times:
      self.assertGreaterEqual(t, 0.0)

  def test_event_scope_name_override(self):
    fn = warp_util.event_scope(lambda: 7, name="custom")
    with warp_util.EventTracer() as tracer:
      self.assertEqual(fn(), 7)
      self.assertEqual(set(tracer.trace()), {"custom"})

  def test_exit_resets_stack(self):
    with warp_util.EventTracer():
      self.assertIsNotNone(warp_util._STACK)
    self.assertIsNone(warp_util._STACK)

  def test_only_one_tracer_at_a_time(self):
    with warp_util.EventTracer():
      with self.assertRaisesRegex(ValueError, "only one EventTracer"):
        warp_util.EventTracer()

  def test_disabled_tracer(self):
    with warp_util.EventTracer(enabled=False) as tracer:
      _outer()
      self.assertEqual(tracer.trace(), {})

  def test_skip_tracing_inside_graph_capture(self):
    def capture_while():
      return _outer()

    with warp_util.EventTracer() as tracer:
      capture_while()
      self.assertEqual(tracer.trace(), {})


class MergeTest(absltest.TestCase):
  """Tests for merging event trace stacks."""

  def test_merge_with_empty(self):
    a = {"f": ((1,), {})}
    self.assertEqual(warp_util._merge(a, {}), a)
    self.assertEqual(warp_util._merge({}, a), a)
    self.assertEqual(warp_util._merge({}, {}), {})

  def test_merge_concatenates_events_recursively(self):
    a = {"f": ((1,), {"g": ((10,), {})})}
    b = {"f": ((2,), {"g": ((20,), {})})}
    self.assertEqual(warp_util._merge(a, b), {"f": ((1, 2), {"g": ((10, 20), {})})})

  def test_merge_incompatible_stacks(self):
    with self.assertRaisesRegex(ValueError, "incompatible stacks"):
      warp_util._merge({"f": ((), {})}, {"g": ((), {})})


class CacheKernelTest(absltest.TestCase):
  """Tests for cache_kernel."""

  def test_cache_by_argument(self):
    calls = []

    @warp_util.cache_kernel
    def factory(*args):
      calls.append(args)
      return object()

    arr = types.SimpleNamespace(size=3)
    k1 = factory(arr, [1, 2], True)
    k2 = factory(types.SimpleNamespace(size=3), [1, 2], True)
    k3 = factory(arr, [1, 3], True)

    self.assertIs(k1, k2)
    self.assertIsNot(k1, k3)
    self.assertLen(calls, 2)


class CheckToolkitDriverTest(parameterized.TestCase):
  """Tests for check_toolkit_driver."""

  def _check(self, device, conditional_supported):
    with (
      mock.patch.object(wp, "get_device", return_value=device),
      mock.patch.object(wp, "is_conditional_graph_supported", return_value=conditional_supported),
      warnings.catch_warnings(record=True) as caught,
    ):
      warnings.simplefilter("always")
      warp_util.check_toolkit_driver()
    return [str(w.message) for w in caught]

  def test_cpu_device(self):
    device = types.SimpleNamespace(is_cuda=False)
    self.assertEqual(self._check(device, conditional_supported=False), [])

  @parameterized.parameters(
    (False, False, "CUDA version < 12.4"),
    (True, False, "HIP/ROCm device detected"),
    (False, True, None),
    (True, True, None),
  )
  def test_gpu_device(self, is_hip, conditional_supported, expected):
    device = types.SimpleNamespace(is_cuda=True, is_hip=is_hip)
    messages = self._check(device, conditional_supported)
    if expected is None:
      self.assertEqual(messages, [])
    else:
      self.assertLen(messages, 1)
      self.assertIn(expected, messages[0])


class ScopedMathdxGemmDisabledTest(absltest.TestCase):
  """Tests for scoped_mathdx_gemm_disabled."""

  def test_disable_and_restore(self):
    config = types.SimpleNamespace(enable_mathdx_gemm=True)
    with mock.patch.object(wp, "config", config):
      with warp_util.scoped_mathdx_gemm_disabled():
        self.assertFalse(config.enable_mathdx_gemm)
      self.assertTrue(config.enable_mathdx_gemm)

  def test_restore_on_exception(self):
    config = types.SimpleNamespace(enable_mathdx_gemm=True)
    with mock.patch.object(wp, "config", config):
      with self.assertRaises(RuntimeError):
        with warp_util.scoped_mathdx_gemm_disabled():
          raise RuntimeError
      self.assertTrue(config.enable_mathdx_gemm)

  def test_disable_false_is_noop(self):
    config = types.SimpleNamespace(enable_mathdx_gemm=True)
    with mock.patch.object(wp, "config", config):
      with warp_util.scoped_mathdx_gemm_disabled(disable=False):
        self.assertTrue(config.enable_mathdx_gemm)

  def test_config_without_mathdx_gemm(self):
    config = types.SimpleNamespace()
    with mock.patch.object(wp, "config", config):
      with warp_util.scoped_mathdx_gemm_disabled():
        self.assertFalse(hasattr(config, "enable_mathdx_gemm"))


if __name__ == "__main__":
  absltest.main()
