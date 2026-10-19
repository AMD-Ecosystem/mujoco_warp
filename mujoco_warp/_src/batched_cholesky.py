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
"""Lane-cooperative batched dense Cholesky factorize+solve for small nv (AMD/HIP).

Motivation: on HIP/gfx942 wp.tile_cholesky has no MathDx/MFMA path and falls back
to a scalar column-serial algorithm that every one of the 64 block threads executes
REDUNDANTLY on a shared tile, routing every element access through LDS. Profiling
showed ~17.8k dependent LDS ops per 27x27 factorize+solve -> the kernel is a
dependent-LDS-latency-bound serial chain (~69% of the Newton step at ~4% of peak).

This kernel instead assigns one 64-lane wavefront per world (preserving occupancy)
and distributes the work across lanes: lane i owns row i. Each column step does a
serial sqrt on the pivot lane plus a parallel rank-1 style update across the active
lanes, publishing shared state with a single tile_scatter_masked (which emits the
required __syncthreads). This turns the O(n^3) dependent-LDS chain into O(n^2)
barrier-free shared reads with O(n) barriers.

Reads of the ORIGINAL matrix are taken from the UPPER triangle (A[k,i], k<i) to match
the existing fill_mode="upper" contract of update_gradient_cholesky (h may only have
its upper triangle populated). The lower triangle is overwritten with the L factor
(A = L L^T); the solves then read L from the lower triangle.
"""

import os

import warp as wp

from mujoco_warp._src.warp_util import cache_kernel

wp.set_module_options({"enable_backward": False})

# Use the lane-cooperative batched dense Cholesky (HIP/gfx942 fast path) instead of
# the redundant scalar wp.tile_cholesky. Defaults on; set MJW_BATCHED_CHOLESKY=0 to
# fall back to the stock kernels for A/B comparison.
USE_BATCHED_CHOLESKY = os.environ.get("MJW_BATCHED_CHOLESKY", "1") != "0"

# Only route matrices no larger than this through the batched kernel. One matrix maps
# to one 64-lane wavefront with lane == row/column, so the matrix dimension must not
# exceed the wavefront width (64). Larger matrices keep the existing blocked path.
BATCHED_CHOLESKY_MAX_DIM = 64

# Largest matrix routed through the one-thread-per-world register kernel. Its packed
# factor needs nv*(nv+1)/2 registers per thread; beyond ~29 it spills to scratch on
# gfx942 and falls behind the lane-cooperative kernel (break-even at nv=32).
SERIAL_CHOLESKY_MAX_DIM = int(os.environ.get("MJW_SERIAL_CHOLESKY_MAX_DIM", "29"))

# Above SERIAL_CHOLESKY_MAX_DIM (and up to the wavefront width), route through the
# register/readlane kernel instead of the shared-memory lane-cooperative one.
USE_READLANE_CHOLESKY = os.environ.get("MJW_READLANE_CHOLESKY", "1") != "0"

# Broadcast lane l's value of x to the whole wavefront. l must be wave-uniform (it is a
# compile-time constant after unrolling), so this lowers to v_readlane_b32 instead of a
# ds_bpermute round trip.
_READLANE = "[](float x, int l) { return __int_as_float(__builtin_amdgcn_readlane(__float_as_int(x), l)); }"


@cache_kernel
def batched_cholesky_factorize_solve(nv: int, nv_pad: int):
  """Factory: returns a kernel that solves A x = b for a batch of SPD systems.

  Kernel signature matches update_gradient_cholesky so it is a drop-in replacement:
    inputs  = [grad (nworld, nv_pad), h (nworld, nv_pad, nv_pad), done (nworld,)]
    outputs = [Mgrad (nworld, nv_pad)]
  Launch with wp.launch_tiled(dim=nworld, block_dim=>=nv).
  """
  N = wp.static(nv)

  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # In:
    ctx_grad_in: wp.array2d(dtype=float),
    h_in: wp.array3d(dtype=float),
    ctx_done_in: wp.array(dtype=bool),
    # Out:
    ctx_Mgrad_out: wp.array2d(dtype=float),
  ):
    worldid, lane = wp.tid()

    # Uniform across the block (one world per block) -> no divergence.
    if ctx_done_in[worldid]:
      return

    # A: shared working tile. Upper triangle keeps the original SPD matrix;
    # lower triangle + diagonal are overwritten with the L factor.
    A = wp.tile_load(h_in[worldid], shape=(N, N), storage="shared")
    # y: shared RHS/solution vector (starts as b, ends as x = A^-1 b).
    y = wp.tile_load(ctx_grad_in[worldid], shape=(N,), storage="shared")

    # ---- Factorization: A = L L^T (left-looking, lower) ----
    for k in range(N):
      # Pivot: L[k,k] = sqrt(A[k,k] - sum_{j<k} L[k,j]^2)   (lane k only)
      d = float(0.0)
      if lane == k:
        s = wp.tile_extract(A, k, k)
        for j in range(k):
          r = wp.tile_extract(A, k, j)
          s -= r * r
        d = wp.sqrt(s)
      wp.tile_scatter_masked(A, k, k, d, lane == k)  # publish L[k,k] (+ barrier)
      inv = 1.0 / wp.tile_extract(A, k, k)

      # Column k below the diagonal: L[i,k] = (A[k,i] - sum_{j<k} L[i,j] L[k,j]) / L[k,k]
      val = float(0.0)
      has = bool(False)
      if lane > k and lane < N:
        s = wp.tile_extract(A, k, lane)  # UPPER triangle = original off-diagonal
        for j in range(k):
          s -= wp.tile_extract(A, lane, j) * wp.tile_extract(A, k, j)
        val = s * inv
        has = True
      row = wp.min(lane, N - 1)
      wp.tile_scatter_masked(A, row, k, val, has)  # write L[i,k] (+ barrier)

    # ---- Forward solve: L y = b (column-oriented) ----
    for k in range(N):
      yk = float(0.0)
      if lane == k:
        yk = wp.tile_extract(y, k) / wp.tile_extract(A, k, k)
      wp.tile_scatter_masked(y, k, yk, lane == k)
      ykk = wp.tile_extract(y, k)
      upd = float(0.0)
      has = bool(False)
      if lane > k and lane < N:
        upd = wp.tile_extract(y, lane) - wp.tile_extract(A, lane, k) * ykk
        has = True
      row = wp.min(lane, N - 1)
      wp.tile_scatter_masked(y, row, upd, has)

    # ---- Back solve: L^T x = y  (L^T[i,k] = L[k,i]) ----
    for kk in range(N):
      k = N - 1 - kk
      xk = float(0.0)
      if lane == k:
        xk = wp.tile_extract(y, k) / wp.tile_extract(A, k, k)
      wp.tile_scatter_masked(y, k, xk, lane == k)
      xkk = wp.tile_extract(y, k)
      upd = float(0.0)
      has = bool(False)
      if lane < k:  # lane < k <= N-1, always in-range
        upd = wp.tile_extract(y, lane) - wp.tile_extract(A, k, lane) * xkk
        has = True
      wp.tile_scatter_masked(y, lane, upd, has)

    wp.tile_store(ctx_Mgrad_out[worldid], y)

  return kernel


@cache_kernel
def serial_cholesky_factorize_solve(nv: int):
  """Factory: one-thread-per-world dense Cholesky factorize+solve held in registers.

  Same inputs/outputs and upper-triangle contract as batched_cholesky_factorize_solve,
  but launched with wp.launch(dim=nworld, block_dim=64). Fully unrolled for a fixed nv,
  so the packed factor stays in registers and there are no shared-memory round trips
  or barriers; the launch bound lets the compiler use the full register file.
  """
  tri = nv * (nv + 1) // 2
  snippet = f"""
  constexpr int N = {nv};
  if (done[w]) return;
  float L[{tri}];
  float y[N];
  float inv[N];
  #pragma unroll
  for (int i = 0; i < N; i++) {{
    #pragma unroll
    for (int j = 0; j <= i; j++) L[i * (i + 1) / 2 + j] = *wp::address(h, w, j, i);
    y[i] = *wp::address(grad, w, i);
  }}
  #pragma unroll
  for (int i = 0; i < N; i++) {{
    #pragma unroll
    for (int j = 0; j <= i; j++) {{
      float s = L[i * (i + 1) / 2 + j];
      #pragma unroll
      for (int k = 0; k < j; k++) s -= L[i * (i + 1) / 2 + k] * L[j * (j + 1) / 2 + k];
      if (i == j) {{
        const float d = sqrtf(s);
        L[i * (i + 1) / 2 + i] = d;
        inv[i] = 1.0f / d;
      }} else {{
        L[i * (i + 1) / 2 + j] = s * inv[j];
      }}
    }}
  }}
  #pragma unroll
  for (int i = 0; i < N; i++) {{
    float s = y[i];
    #pragma unroll
    for (int k = 0; k < i; k++) s -= L[i * (i + 1) / 2 + k] * y[k];
    y[i] = s * inv[i];
  }}
  #pragma unroll
  for (int ii = 0; ii < N; ii++) {{
    const int i = N - 1 - ii;
    float s = y[i];
    #pragma unroll
    for (int k = i + 1; k < N; k++) s -= L[k * (k + 1) / 2 + i] * y[k];
    y[i] = s * inv[i];
  }}
  #pragma unroll
  for (int i = 0; i < N; i++) *wp::address(Mgrad, w, i) = y[i];
  """

  def factorize_solve(
    grad: wp.array2d(dtype=float), h: wp.array3d(dtype=float), done: wp.array(dtype=bool), Mgrad: wp.array2d(dtype=float), w: int
  ): ...

  # native functions are keyed by qualified name; make it unique per nv
  factorize_solve.__qualname__ = f"serial_cholesky_factorize_solve_{nv}"
  factorize_solve = wp.func_native(snippet)(factorize_solve)

  @wp.kernel(module="unique", enable_backward=False, launch_bounds=64)
  def kernel(
    # In:
    ctx_grad_in: wp.array2d(dtype=float),
    h_in: wp.array3d(dtype=float),
    ctx_done_in: wp.array(dtype=bool),
    # Out:
    ctx_Mgrad_out: wp.array2d(dtype=float),
  ):
    worldid = wp.tid()
    factorize_solve(ctx_grad_in, h_in, ctx_done_in, ctx_Mgrad_out, worldid)

  return kernel


@cache_kernel
def readlane_cholesky_factorize_solve(nv: int):
  """Factory: one-wavefront-per-world Cholesky factorize+solve with rows in registers.

  Same inputs/outputs and upper-triangle contract as batched_cholesky_factorize_solve,
  but launched with wp.launch(dim=nworld * 64, block_dim=64). Lane i holds row i of the
  matrix in registers; each right-looking column step broadcasts the pivot and column
  with v_readlane, so the factorization has no shared-memory traffic or barriers. The
  back solve needs columns of L, so the factor goes through shared memory once.
  """
  snippet = f"""
  constexpr int N = {nv};
  constexpr int S = N + 1;
  auto rl = {_READLANE};
  const int w = tid >> 6;
  const int lane = tid & 63;
  if (done[w]) return;
  __shared__ float Ls[N * S];
  const int r = lane < N ? lane : N - 1;
  float a[N];
  #pragma unroll
  for (int j = 0; j < N; j++) a[j] = *wp::address(h, w, j < r ? j : r, j < r ? r : j);
  float dinv = 0.0f;
  #pragma unroll
  for (int k = 0; k < N; k++) {{
    const float dk = sqrtf(rl(a[k], k));
    const float ik = 1.0f / dk;
    const float l = lane == k ? dk : a[k] * ik;
    if (lane == k) dinv = ik;
    a[k] = l;
    #pragma unroll
    for (int j = k + 1; j < N; j++) a[j] -= l * rl(l, j);
  }}
  float b = *wp::address(grad, w, r);
  #pragma unroll
  for (int k = 0; k < N; k++) {{
    const float yk = rl(b, k) * rl(dinv, k);
    b = lane == k ? yk : (lane > k ? b - a[k] * yk : b);
  }}
  if (lane < N) {{
    #pragma unroll
    for (int j = 0; j < N; j++) Ls[lane * S + j] = a[j];
  }}
  __syncthreads();
  #pragma unroll
  for (int kk = 0; kk < N; kk++) {{
    const int k = N - 1 - kk;
    const float xk = rl(b, k) * rl(dinv, k);
    b = lane == k ? xk : (lane < k ? b - Ls[k * S + lane] * xk : b);
  }}
  if (lane < N) *wp::address(Mgrad, w, lane) = b;
  """

  def factorize_solve(
    grad: wp.array2d(dtype=float), h: wp.array3d(dtype=float), done: wp.array(dtype=bool), Mgrad: wp.array2d(dtype=float), tid: int
  ): ...

  factorize_solve.__qualname__ = f"readlane_cholesky_factorize_solve_{nv}"
  factorize_solve = wp.func_native(snippet)(factorize_solve)

  @wp.kernel(module="unique", enable_backward=False, launch_bounds=64)
  def kernel(
    # In:
    ctx_grad_in: wp.array2d(dtype=float),
    h_in: wp.array3d(dtype=float),
    ctx_done_in: wp.array(dtype=bool),
    # Out:
    ctx_Mgrad_out: wp.array2d(dtype=float),
  ):
    tid = wp.tid()
    factorize_solve(ctx_grad_in, h_in, ctx_done_in, ctx_Mgrad_out, tid)

  return kernel


@cache_kernel
def readlane_jtdaj_dense(nv: int, quadratic_state: int):
  """Factory: h = M + J^T diag(D * (state == QUADRATIC)) J for a dense Jacobian.

  Drop-in for solver.update_gradient_JTDAJ_dense_tiled on HIP, launched with
  wp.launch(dim=nworld * 64, block_dim=64). Lane c accumulates column c of h in
  registers; each constraint row is loaded coalesced and broadcast with v_readlane.
  Writes the nv x nv block of h only (padding is left as allocated).
  """
  snippet = f"""
  constexpr int N = {nv};
  auto rl = {_READLANE};
  const int w = tid >> 6;
  const int lane = tid & 63;
  if (done[w]) return;
  const int c = lane < N ? lane : N - 1;
  float col[N];
  #pragma unroll
  for (int j = 0; j < N; j++) col[j] = *wp::address(M, w, j, c);
  const int ne = *wp::address(nefc, w);
  for (int k = 0; k < ne; k++) {{
    if (*wp::address(state, w, k) != {quadratic_state}) continue;
    const float jc = *wp::address(J, w, k, c);
    const float djc = *wp::address(D, w, k) * jc;
    #pragma unroll
    for (int j = 0; j < N; j++) col[j] += djc * rl(jc, j);
  }}
  if (lane < N) {{
    #pragma unroll
    for (int j = 0; j < N; j++) *wp::address(h, w, j, lane) = col[j];
  }}
  """

  def jtdaj(
    nefc: wp.array(dtype=int),
    M: wp.array3d(dtype=float),
    J: wp.array3d(dtype=float),
    D: wp.array2d(dtype=float),
    state: wp.array2d(dtype=int),
    done: wp.array(dtype=bool),
    h: wp.array3d(dtype=float),
    tid: int,
  ): ...

  jtdaj.__qualname__ = f"readlane_jtdaj_dense_{nv}"
  jtdaj = wp.func_native(snippet)(jtdaj)

  @wp.kernel(module="unique", enable_backward=False, launch_bounds=64)
  def kernel(
    # Data in:
    nefc_in: wp.array(dtype=int),
    M_in: wp.array3d(dtype=float),
    efc_J_in: wp.array3d(dtype=float),
    efc_D_in: wp.array2d(dtype=float),
    efc_state_in: wp.array2d(dtype=int),
    # In:
    ctx_done_in: wp.array(dtype=bool),
    # Out:
    ctx_h_out: wp.array3d(dtype=float),
  ):
    tid = wp.tid()
    jtdaj(nefc_in, M_in, efc_J_in, efc_D_in, efc_state_in, ctx_done_in, ctx_h_out, tid)

  return kernel


@cache_kernel
def readlane_h_incremental(nv: int, quadratic_state: int):
  """Factory: register/readlane version of solver.update_gradient_h_incremental.

  Launched with wp.launch(dim=nworld * 64, block_dim=64): one wavefront per world, so
  worlds with no changed constraints retire as a whole wave. Lane c accumulates the
  delta for column c over the changed rows, then adds it to the upper triangle h[j, c],
  j <= c, in the same order as the per-element kernel.
  """
  snippet = f"""
  constexpr int N = {nv};
  auto rl = {_READLANE};
  const int w = tid >> 6;
  const int lane = tid & 63;
  const int nc = *wp::address(changed_count, w);
  if (nc == 0) return;
  const int c = lane < N ? lane : N - 1;
  float delta[N];
  #pragma unroll
  for (int j = 0; j < N; j++) delta[j] = 0.0f;
  for (int i = 0; i < nc; i++) {{
    const int e = *wp::address(changed_ids, w, i);
    const float jc = *wp::address(J, w, e, c);
    const float dk = *wp::address(D, w, e);
    const float s = *wp::address(state, w, e) == {quadratic_state} ? dk * jc : -(dk * jc);
    #pragma unroll
    for (int j = 0; j < N; j++) delta[j] += s * rl(jc, j);
  }}
  if (lane < N) {{
    #pragma unroll
    for (int j = 0; j < N; j++) {{
      if (j <= lane && delta[j] != 0.0f) *wp::address(h, w, j, lane) += delta[j];
    }}
  }}
  """

  def h_incremental(
    J: wp.array3d(dtype=float),
    D: wp.array2d(dtype=float),
    state: wp.array2d(dtype=int),
    changed_ids: wp.array2d(dtype=int),
    changed_count: wp.array(dtype=int),
    h: wp.array3d(dtype=float),
    tid: int,
  ): ...

  h_incremental.__qualname__ = f"readlane_h_incremental_{nv}"
  h_incremental = wp.func_native(snippet)(h_incremental)

  @wp.kernel(module="unique", enable_backward=False, launch_bounds=64)
  def kernel(
    # Data in:
    efc_J_in: wp.array3d(dtype=float),
    efc_D_in: wp.array2d(dtype=float),
    efc_state_in: wp.array2d(dtype=int),
    # In:
    changed_ids_in: wp.array2d(dtype=int),
    changed_count_in: wp.array(dtype=int),
    # Out:
    ctx_h_out: wp.array3d(dtype=float),
  ):
    tid = wp.tid()
    h_incremental(efc_J_in, efc_D_in, efc_state_in, changed_ids_in, changed_count_in, ctx_h_out, tid)

  return kernel


@cache_kernel
def readlane_factor_solve_i(tile_size: int):
  """Factory: register/readlane version of batched_factor_solve_i.

  Same inputs/outputs plus nnode, launched with   wp.launch(dim=nworld * nnode * 64,
  block_dim=64): one wavefront per (world, node) diagonal block. Stores U = L^T (zero
  below the diagonal) into L (fill_mode="upper") for the solve-only path.
  """
  snippet = f"""
  constexpr int N = {tile_size};
  constexpr int S = N + 1;
  auto rl = {_READLANE};
  const int blk = tid >> 6;
  const int lane = tid & 63;
  const int w = blk / nnode;
  const int o = *wp::address(adr, blk - w * nnode);
  __shared__ float Ls[N * S];
  const int r = lane < N ? lane : N - 1;
  float a[N];
  #pragma unroll
  for (int j = 0; j < N; j++) a[j] = *wp::address(M, w, o + (j < r ? j : r), o + (j < r ? r : j));
  float dinv = 0.0f;
  #pragma unroll
  for (int k = 0; k < N; k++) {{
    const float dk = sqrtf(rl(a[k], k));
    const float ik = 1.0f / dk;
    const float l = lane == k ? dk : a[k] * ik;
    if (lane == k) dinv = ik;
    a[k] = l;
    #pragma unroll
    for (int j = k + 1; j < N; j++) a[j] -= l * rl(l, j);
  }}
  float b = *wp::address(y, w, o + r);
  if (lane < N) {{
    #pragma unroll
    for (int k = 0; k < N; k++) {{
      *wp::address(L, w, o + k, o + lane) = k <= lane ? a[k] : 0.0f;
      Ls[lane * S + k] = a[k];
    }}
  }}
  #pragma unroll
  for (int k = 0; k < N; k++) {{
    const float zk = rl(b, k) * rl(dinv, k);
    b = lane == k ? zk : (lane > k ? b - a[k] * zk : b);
  }}
  __syncthreads();
  #pragma unroll
  for (int kk = 0; kk < N; kk++) {{
    const int k = N - 1 - kk;
    const float xk = rl(b, k) * rl(dinv, k);
    b = lane == k ? xk : (lane < k ? b - Ls[k * S + lane] * xk : b);
  }}
  if (lane < N) *wp::address(x, w, o + lane) = b;
  """

  def factor_solve(
    M: wp.array3d(dtype=float),
    y: wp.array2d(dtype=float),
    adr: wp.array(dtype=int),
    nnode: int,
    x: wp.array2d(dtype=float),
    L: wp.array3d(dtype=float),
    tid: int,
  ): ...

  factor_solve.__qualname__ = f"readlane_factor_solve_i_{tile_size}"
  factor_solve = wp.func_native(snippet)(factor_solve)

  @wp.kernel(module="unique", enable_backward=False, launch_bounds=64)
  def kernel(
    # In:
    M_in: wp.array3d(dtype=float),
    y_in: wp.array2d(dtype=float),
    adr: wp.array(dtype=int),
    nnode: int,
    # Out:
    x_out: wp.array2d(dtype=float),
    L_out: wp.array3d(dtype=float),
  ):
    tid = wp.tid()
    factor_solve(M_in, y_in, adr, nnode, x_out, L_out, tid)

  return kernel


@cache_kernel
def batched_factor_solve_i(tile_size: int):
  """Factory: lane-cooperative UPPER Cholesky (A = U^T U) + solve for a diagonal
  block of the mass matrix, drop-in for smooth._tile_cholesky_factorize_solve.

  Stores the UPPER factor U into L (matching fill_mode="upper") so the separate
  solve-only path (_tile_cholesky_solve) can reuse the cached factor. Signature:
    inputs  = [M (nworld, nv_pad, nv_pad), y (nworld, nv_pad), adr (nnode,)]
    outputs = [x (nworld, nv_pad), L (nworld, nv_pad, nv_pad)]
  Launch with wp.launch_tiled(dim=(nworld, nnode), block_dim=>=tile_size).
  """
  N = wp.static(tile_size)

  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # In:
    M_in: wp.array3d(dtype=float),
    y_in: wp.array2d(dtype=float),
    adr: wp.array(dtype=int),
    # Out:
    x_out: wp.array2d(dtype=float),
    L_out: wp.array3d(dtype=float),
  ):
    worldid, nodeid, lane = wp.tid()
    dofid = adr[nodeid]

    # Shared working tile; upper triangle + diagonal hold U after factorization.
    A = wp.tile_load(M_in[worldid], shape=(N, N), offset=(dofid, dofid), storage="shared")
    y = wp.tile_load(y_in[worldid], shape=(N,), offset=(dofid,), storage="shared")

    # ---- Factorization: A = U^T U (left-looking, upper); lane == column ----
    for k in range(N):
      # U[k,k] = sqrt(A[k,k] - sum_{i<k} U[i,k]^2)   (lane k only)
      d = float(0.0)
      if lane == k:
        s = wp.tile_extract(A, k, k)
        for i in range(k):
          u = wp.tile_extract(A, i, k)
          s -= u * u
        d = wp.sqrt(s)
      wp.tile_scatter_masked(A, k, k, d, lane == k)
      inv = 1.0 / wp.tile_extract(A, k, k)

      # Row k, cols j>k: U[k,j] = (A[k,j] - sum_{i<k} U[i,k] U[i,j]) / U[k,k]
      val = float(0.0)
      has = bool(False)
      if lane > k and lane < N:
        s = wp.tile_extract(A, k, lane)
        for i in range(k):
          s -= wp.tile_extract(A, i, k) * wp.tile_extract(A, i, lane)
        val = s * inv
        has = True
      col = wp.min(lane, N - 1)
      wp.tile_scatter_masked(A, k, col, val, has)

    # Persist the upper factor U (matches fill_mode="upper" for later reuse).
    wp.tile_store(L_out[worldid], A, offset=(dofid, dofid))

    # ---- Forward solve: U^T z = y  ((U^T)[i,k] = U[k,i]) ----
    for k in range(N):
      zk = float(0.0)
      if lane == k:
        zk = wp.tile_extract(y, k) / wp.tile_extract(A, k, k)
      wp.tile_scatter_masked(y, k, zk, lane == k)
      zkk = wp.tile_extract(y, k)
      upd = float(0.0)
      has = bool(False)
      if lane > k and lane < N:
        upd = wp.tile_extract(y, lane) - wp.tile_extract(A, k, lane) * zkk
        has = True
      row = wp.min(lane, N - 1)
      wp.tile_scatter_masked(y, row, upd, has)

    # ---- Back solve: U x = z  (U[i,k] = A[i,k], i<k) ----
    for kk in range(N):
      k = N - 1 - kk
      xk = float(0.0)
      if lane == k:
        xk = wp.tile_extract(y, k) / wp.tile_extract(A, k, k)
      wp.tile_scatter_masked(y, k, xk, lane == k)
      xkk = wp.tile_extract(y, k)
      upd = float(0.0)
      has = bool(False)
      if lane < k:
        upd = wp.tile_extract(y, lane) - wp.tile_extract(A, lane, k) * xkk
        has = True
      wp.tile_scatter_masked(y, lane, upd, has)

    wp.tile_store(x_out[worldid], y, offset=(dofid,))

  return kernel
