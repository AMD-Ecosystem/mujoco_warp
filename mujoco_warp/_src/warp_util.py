# Copyright 2025 The Newton Developers
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

import functools
import inspect
import os
import warnings
import weakref

import numpy as np
import warp as wp
import warp._src.context as _wp_context

_STACK = None


class EventTracer:
  """Calculates elapsed times of functions annotated with `event_scope`.

  Use as a context manager like so:

    @event_trace
    def my_warp_function(...):
      ...

    with EventTracer() as tracer:
      my_warp_function(...)
      print(tracer.trace())
  """

  def __init__(self, enabled: bool = True):
    global _STACK
    if _STACK is not None:
      raise ValueError("only one EventTracer can run at a time")
    if enabled:
      _STACK = {}

  def __enter__(self):
    return self

  def trace(self) -> dict:
    """Calculates elapsed times for every node of the trace."""
    global _STACK

    if _STACK is None:
      return {}

    ret = {}

    for k, v in _STACK.items():
      events, sub_stack = v
      # push into next level of stack
      saved_stack, _STACK = _STACK, sub_stack
      sub_trace = self.trace()
      # pop!
      _STACK = saved_stack
      events = tuple(wp.get_event_elapsed_time(beg, end) for beg, end in events)
      ret[k] = (events, sub_trace)

    return ret

  def __exit__(self, type, value, traceback):
    global _STACK
    _STACK = None


def _merge(a: dict, b: dict) -> dict:
  """Merges two event trace stacks."""
  ret = {}
  if not a or not b:
    return dict(**a, **b)
  if set(a) != set(b):
    raise ValueError("incompatible stacks")
  for key in a:
    a1_events, a1_substack = a[key]
    a2_events, a2_substack = b[key]
    ret[key] = (a1_events + a2_events, _merge(a1_substack, a2_substack))
  return ret


def event_scope(fn, name: str = ""):
  """Wraps a function and records an event before and after the function invocation."""
  name = name or getattr(fn, "__name__")

  @functools.wraps(fn)
  def wrapper(*args, **kwargs):
    global _STACK
    if _STACK is None:
      return fn(*args, **kwargs)

    for frame_info in inspect.stack():
      if frame_info.function in ("capture_while", "capture_if"):
        return fn(*args, **kwargs)

    # push into next level of stack
    saved_stack, _STACK = _STACK, {}
    beg = wp.Event(enable_timing=True)
    end = wp.Event(enable_timing=True)
    wp.record_event(beg)
    res = fn(*args, **kwargs)
    wp.record_event(end)
    # pop back up to current level
    sub_stack, _STACK = _STACK, saved_stack
    # append events and substack
    prev_events, prev_substack = _STACK.get(name, ((), {}))
    events = prev_events + ((beg, end),)
    sub_stack = _merge(prev_substack, sub_stack)
    _STACK[name] = (events, sub_stack)
    return res

  return wrapper


_KERNEL_CACHE = {}


def cache_kernel(func):
  # caching kernels to avoid crashes in graph_conditional code
  @functools.wraps(func)
  def wrapper(*args):
    def _hash_arg(a):
      if hasattr(a, "size"):
        return a.size
      if isinstance(a, list):
        return hash(tuple(a))
      return hash(a)

    key = tuple(_hash_arg(a) for a in args) + (hash(func.__name__),)
    if key not in _KERNEL_CACHE:
      _KERNEL_CACHE[key] = func(*args)
    return _KERNEL_CACHE[key]

  return wrapper


_LAUNCH_CACHE_ENABLED = os.environ.get("MJW_LAUNCH_CACHE", "1") != "0"
_LAUNCH_CACHE_MAX = 8192
_LAUNCH_CACHE = {}
_LAUNCH_SEEN = {}
_VALUE_ARG_TYPES = frozenset((int, float, bool, type(None), np.int32, np.int64, np.float32, np.float64, np.bool_))


def _launch_key(kernel, dim, inputs, outputs, device, block_dim, max_blocks):
  key = [kernel, dim if type(dim) is int else tuple(dim), id(device), block_dim, max_blocks]
  for args in (inputs, outputs):
    for a in args:
      t = type(a)
      if t in _VALUE_ARG_TYPES:
        key.append(a)
      elif isinstance(a, wp.array):
        key.append(id(a))
      else:
        return None
  return tuple(key)


def launch(kernel, dim, inputs=(), outputs=(), device=None, stream=None, max_blocks: int = 0, block_dim: int = 256):
  """Drop-in for wp.launch that replays recorded launches for repeated argument sets.

  wp.launch re-packs every argument on each call (~15 us of host time for a typical
  solver kernel), which dominates step time when the GPU work per kernel is small. A
  launch whose kernel, dims and arguments (arrays by identity, scalars by value) repeat is
  recorded once with record_cmd=True and replayed afterwards. Entries hold references to
  their arrays, so an id in a key always refers to the same live array; keys are only
  cached on their second occurrence so per-call temporaries are not kept alive. The first
  occurrence is remembered through weak references: a freed temporary's id is often
  reused by the next call's temporary, which must not count as a repeat (each false
  repeat would pin that temporary in the cache).
  """
  if stream is not None:
    device = stream.device
  device = _wp_context.runtime.get_device(device)

  if not _LAUNCH_CACHE_ENABLED or _wp_context.runtime.tape is not None or wp.config.print_launches or _wp_context.runtime._apic_capture is not None:
    return wp.launch(
      kernel, dim, inputs=inputs, outputs=outputs, device=device, stream=stream, max_blocks=max_blocks, block_dim=block_dim
    )

  key = _launch_key(kernel, dim, inputs, outputs, device, block_dim, max_blocks)
  entry = _LAUNCH_CACHE.get(key) if key is not None else None
  if entry is None:
    arrays = [a for args in (inputs, outputs) for a in args if isinstance(a, wp.array)] if key is not None else ()
    seen = _LAUNCH_SEEN.get(key) if key is not None else None
    if seen is None or any(r() is not a for r, a in zip(seen, arrays)):
      if key is not None:
        if len(_LAUNCH_SEEN) >= _LAUNCH_CACHE_MAX:
          _LAUNCH_SEEN.clear()
        _LAUNCH_SEEN[key] = tuple(weakref.ref(a) for a in arrays)
      return wp.launch(
        kernel, dim, inputs=inputs, outputs=outputs, device=device, stream=stream, max_blocks=max_blocks, block_dim=block_dim
      )
    cmd = wp.launch(
      kernel,
      dim,
      inputs=inputs,
      outputs=outputs,
      device=device,
      max_blocks=max_blocks,
      block_dim=block_dim,
      record_cmd=True,
    )
    if len(_LAUNCH_CACHE) >= _LAUNCH_CACHE_MAX:
      _LAUNCH_CACHE.clear()
    entry = _LAUNCH_CACHE[key] = (cmd, tuple(inputs), tuple(outputs))

  cmd = entry[0]
  if cmd is not None:
    cmd.launch(stream)


def launch_tiled(kernel, dim, inputs=(), outputs=(), device=None, stream=None, max_blocks: int = 0, block_dim: int = 256):
  """Drop-in for wp.launch_tiled backed by the launch cache."""
  if stream is not None:
    device = stream.device
  device = _wp_context.runtime.get_device(device)
  if device.is_cpu:
    block_dim = 1
  dim = (dim,) if type(dim) is int else tuple(dim)
  return launch(kernel, dim + (block_dim,), inputs, outputs, device, stream, max_blocks, block_dim)


def check_toolkit_driver():
  wp.init()
  device = wp.get_device()
  if not device.is_cuda:
    return

  # HIP/ROCm devices do not (yet) expose conditional graph nodes; this is an
  # architectural limitation rather than a misconfigured CUDA toolkit. Warn
  # informationally and let io.put_model() default graph_conditional to False.
  if getattr(device, "is_hip", False):
    if not wp.is_conditional_graph_supported():
      warnings.warn(
        "HIP/ROCm device detected: CUDA graph conditional nodes are not "
        "supported on this backend. Model.opt.graph_conditional will be "
        "disabled automatically; the solver falls back to a Python for-loop "
        "around the iteration kernel.",
        stacklevel=2,
      )
    return

  if not wp.is_conditional_graph_supported():
    warnings.warn(
      """
      CUDA version < 12.4 detected
      - graph capture may be unreliable for < 12.3
      - conditional graph nodes are not available for < 12.4
        Model.opt.graph_conditional should be set to False
      """,
      stacklevel=2,
    )


class scoped_mathdx_gemm_disabled:
  """Temporarily disable Warp MathDx GEMM kernels within this scope."""

  def __init__(self, disable: bool = True):
    self._disable = disable
    self._config = None
    self._prev = None

  def __enter__(self):
    if not self._disable:
      return self
    config = getattr(wp, "config", None)
    if config is None or not hasattr(config, "enable_mathdx_gemm"):
      return self
    self._config = config
    self._prev = config.enable_mathdx_gemm
    self._config.enable_mathdx_gemm = False
    return self

  def __exit__(self, exc_type, exc, tb):
    if self._config is not None:
      self._config.enable_mathdx_gemm = self._prev
    return False
