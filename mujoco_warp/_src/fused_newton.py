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
"""Whole Newton solve loop in one kernel, one 64-lane wavefront per world (HIP).

Covers the dense-Jacobian, pyramidal-cone Newton path with nv <= 64. The per-iteration
kernels of solver._solver_iteration (mul_m, linesearch, update_constraint, qfrc/grad,
gauss cost, incremental H, Cholesky, search update, convergence check) become one pass
over register-resident state, and each world's wavefront retires as soon as that world
converges, so straggler worlds no longer keep every other world's kernels launching.

Lane i owns dof i (qacc, Ma, grad, search and column i of the upper triangle of H) and
constraint rows i, i + 64, ... (Jaref, jv, D, state, force). Per-world sums use an xor
butterfly, which leaves the bitwise-identical total in every lane, so all control flow
derived from them is wavefront-uniform. The Cholesky factor stays in shared memory
across iterations and is only recomputed when a constraint changed QUADRATIC state.
"""

import os

import warp as wp

from mujoco_warp._src import types
from mujoco_warp._src import warp_util
from mujoco_warp._src.batched_cholesky import _READLANE
from mujoco_warp._src.warp_util import cache_kernel

wp.set_module_options({"enable_backward": False})

USE_FUSED_NEWTON = os.environ.get("MJW_FUSED_NEWTON", "1") != "0"

# constraint rows are held 64 per register chunk; larger njmax keeps the per-iteration kernels
_MAX_ROW_CHUNKS = 8


def use_fused_newton(m: types.Model, d: types.Data) -> bool:
  return (
    USE_FUSED_NEWTON
    and wp.get_device().is_hip
    and m.opt.solver == types.SolverType.NEWTON
    and m.opt.cone == types.ConeType.PYRAMIDAL
    and not m.is_sparse
    and not m.opt.ls_parallel
    and m.opt.iterations > 0
    and 0 < m.nv <= 64
    and 0 < d.njmax <= 64 * _MAX_ROW_CHUNKS
  )


@cache_kernel
def fused_newton_solve(nv: int, row_chunks: int, ls_iterations: int):
  """Factory: Newton iterations until convergence for each world, launched with
  wp.launch(dim=nworld * 64, block_dim=64) after init_context and solve_init_search."""
  snippet = f"""
  constexpr int N = {nv};
  constexpr int NCH = {row_chunks};
  constexpr int LS = {ls_iterations};
  constexpr int S = (N % 2) ? N : N + 1;
  constexpr int SATISFIED = {types.ConstraintState.SATISFIED.value};
  constexpr int QUADRATIC = {types.ConstraintState.QUADRATIC.value};
  constexpr int LINEARNEG = {types.ConstraintState.LINEARNEG.value};
  constexpr int LINEARPOS = {types.ConstraintState.LINEARPOS.value};
  struct P {{ float c, g, h; }};
  auto rl = {_READLANE};
  auto rli = [](int x, int l) {{ return __builtin_amdgcn_readlane(x, l); }};
  auto wsum = [](float v) {{
    #pragma unroll
    for (int o = 32; o > 0; o >>= 1) v += __shfl_xor(v, o, 64);
    return v;
  }};
  auto wsum3 = [&](P p) {{ return P{{wsum(p.c), wsum(p.g), wsum(p.h)}}; }};
  auto add = [](P a, P b) {{ return P{{a.c + b.c, a.g + b.g, a.h + b.h}}; }};
  auto sdiv = [](float x, float y) {{ return x / (y != 0.0f ? y : {types.MJ_MINVAL}f); }};
  auto quad_pt = [](float q0, float q1, float q2, float alpha) {{
    const float aq2 = alpha * q2;
    return P{{alpha * aq2 + alpha * q1 + q0, 2.0f * aq2 + q1, 2.0f * q2}};
  }};
  auto direct_pt = [](float x, float jv, float D) {{
    const float jvD = jv * D;
    return P{{0.5f * D * x * x, jvD * x, jv * jvD}};
  }};

  const int w = tid >> 6;
  const int lane = tid & 63;
  if (*wp::address(done, w)) return;
  __shared__ float Ls[N * S];

  const int ne = *wp::address(ne_in, w);
  const int nf = *wp::address(nf_in, w);
  const int nefc = min(njmax, *wp::address(nefc_in, w));

  // (cost, d/dalpha, d2/dalpha2) of row e at Jaref + alpha * jv, pyramidal cones
  auto row_pt = [&](int e, float x, float jv, float D, float f) {{
    if (e >= ne + nf) return x < 0.0f ? direct_pt(x, jv, D) : P{{0.0f, 0.0f, 0.0f}};
    if (e >= ne) {{
      const float rf = sdiv(f, D);
      if (-rf < x && x < rf) return direct_pt(x, jv, D);
      if (x <= -rf) return P{{f * (-0.5f * rf - x), -f * jv, 0.0f}};
      return P{{f * (-0.5f * rf + x), f * jv, 0.0f}};
    }}
    return direct_pt(x, jv, D);
  }};

  const bool isdof = lane < N;
  const int di = isdof ? lane : N - 1;
  float qacc = *wp::address(qacc_io, w, di);
  float Ma = *wp::address(Ma_io, w, di);
  const float qs = *wp::address(qfrc_smooth, w, di);
  const float qas = *wp::address(qacc_smooth, w, di);
  float grad = *wp::address(grad_io, w, di);
  float Mgrad = *wp::address(Mgrad_io, w, di);
  float search = *wp::address(search_io, w, di);
  float qfrc = *wp::address(qfrc_io, w, di);

  // column di of the upper triangle of H (entries j > di are never read)
  float hc[N];
  #pragma unroll
  for (int j = 0; j < N; j++) hc[j] = *wp::address(h, w, j < di ? j : di, j < di ? di : j);

  float jaref[NCH], jv[NCH], Dd[NCH], fl[NCH], force[NCH];
  int st[NCH];
  #pragma unroll
  for (int c = 0; c < NCH; c++) {{
    const int e = c * 64 + lane;
    const bool has = e < nefc;
    jaref[c] = has ? *wp::address(Jaref_io, w, e) : 0.0f;
    Dd[c] = has ? *wp::address(efc_D, w, e) : 0.0f;
    fl[c] = has ? *wp::address(efc_frictionloss, w, e) : 0.0f;
    force[c] = has ? *wp::address(force_io, w, e) : 0.0f;
    st[c] = has ? *wp::address(state_io, w, e) : SATISFIED;
    jv[c] = 0.0f;
  }}
  float Jr[N];
  #pragma unroll
  for (int j = 0; j < N; j++) Jr[j] = lane < nefc ? *wp::address(J, w, lane, j) : 0.0f;

  const float tolerance = *wp::address(opt_tolerance, w % opt_tolerance.shape[0]);
  const float ls_tolerance = *wp::address(opt_ls_tolerance, w % opt_ls_tolerance.shape[0]);
  const float scale = *wp::address(stat_meaninertia, w % stat_meaninertia.shape[0]) * static_cast<float>(N);
  float search_dot = *wp::address(search_dot_io, w);
  float gauss = *wp::address(gauss_io, w);
  float cost = *wp::address(cost_io, w);
  float prev_cost = *wp::address(prev_cost_io, w);
  float grad_dot = *wp::address(grad_dot_io, w);
  int niter = *wp::address(niter_io, w);
  bool factored = false;
  float dinv = 0.0f;

  while (true) {{
    // mv = M @ search (M is symmetric; column reads are coalesced)
    float mv = 0.0f;
    #pragma unroll
    for (int j = 0; j < N; j++) mv += *wp::address(M, w, j, di) * rl(search, j);

    // jv = J @ search
    #pragma unroll
    for (int c = 0; c < NCH; c++) {{
      if (c * 64 >= nefc) break;
      float s = 0.0f;
      if (c == 0) {{
        #pragma unroll
        for (int j = 0; j < N; j++) s += Jr[j] * rl(search, j);
      }} else {{
        const int e = c * 64 + lane;
        const bool has = e < nefc;
        #pragma unroll
        for (int j = 0; j < N; j++) s += (has ? *wp::address(J, w, e, j) : 0.0f) * rl(search, j);
      }}
      jv[c] = s;
    }}

    // ---- linesearch (same bracketing as solver.linesearch_iterative_group) ----
    const float gtol = fmaxf(tolerance * ls_tolerance * sqrtf(search_dot) * scale, 1e-6f);
    const float q0 = gauss;
    const float q1 = wsum(isdof ? search * (Ma - qs) : 0.0f);
    const float q2 = wsum(isdof ? 0.5f * search * mv : 0.0f);

    P loc = {{0.0f, 0.0f, 0.0f}};
    #pragma unroll
    for (int c = 0; c < NCH; c++) {{
      if (c * 64 >= nefc) break;
      if (c * 64 + lane < nefc) loc = add(loc, row_pt(c * 64 + lane, jaref[c], jv[c], Dd[c], fl[c]));
    }}
    const P p0 = add(P{{q0, q1, 2.0f * q2}}, wsum3(loc));

    const float lo_alpha_in = -sdiv(p0.g, p0.h);
    loc = P{{0.0f, 0.0f, 0.0f}};
    #pragma unroll
    for (int c = 0; c < NCH; c++) {{
      if (c * 64 >= nefc) break;
      if (c * 64 + lane < nefc) loc = add(loc, row_pt(c * 64 + lane, jaref[c] + lo_alpha_in * jv[c], jv[c], Dd[c], fl[c]));
    }}
    const P lo_in = add(quad_pt(q0, q1, q2, lo_alpha_in), wsum3(loc));

    float alpha = lo_alpha_in;
    if (!(fabsf(lo_in.g) < gtol && lo_in.c < p0.c)) {{
      alpha = 0.0f;
      const bool lo_less = lo_in.g < p0.g;
      P lo = lo_less ? lo_in : p0;
      float lo_alpha = lo_less ? lo_alpha_in : 0.0f;
      P hi = lo_less ? p0 : lo_in;
      float hi_alpha = lo_less ? 0.0f : lo_alpha_in;
      auto in_bracket = [](P x, P y) {{ return (x.g < y.g && y.g < 0.0f) || (x.g > y.g && y.g > 0.0f); }};

      for (int it = 0; it < LS; it++) {{
        const float lo_next_alpha = lo_alpha - sdiv(lo.g, lo.h);
        const float hi_next_alpha = hi_alpha - sdiv(hi.g, hi.h);
        const float mid_alpha = 0.5f * (lo_alpha + hi_alpha);

        P l_lo = {{0.0f, 0.0f, 0.0f}}, l_hi = {{0.0f, 0.0f, 0.0f}}, l_mid = {{0.0f, 0.0f, 0.0f}};
        #pragma unroll
        for (int c = 0; c < NCH; c++) {{
          if (c * 64 >= nefc) break;
          const int e = c * 64 + lane;
          if (e < nefc) {{
            l_lo = add(l_lo, row_pt(e, jaref[c] + lo_next_alpha * jv[c], jv[c], Dd[c], fl[c]));
            l_hi = add(l_hi, row_pt(e, jaref[c] + hi_next_alpha * jv[c], jv[c], Dd[c], fl[c]));
            l_mid = add(l_mid, row_pt(e, jaref[c] + mid_alpha * jv[c], jv[c], Dd[c], fl[c]));
          }}
        }}
        const P lo_next = add(quad_pt(q0, q1, q2, lo_next_alpha), wsum3(l_lo));
        const P hi_next = add(quad_pt(q0, q1, q2, hi_next_alpha), wsum3(l_hi));
        const P mid = add(quad_pt(q0, q1, q2, mid_alpha), wsum3(l_mid));

        const bool s_lo_lo_next = in_bracket(lo, lo_next);
        if (s_lo_lo_next) {{ lo = lo_next; lo_alpha = lo_next_alpha; }}
        const bool s_lo_mid = in_bracket(lo, mid);
        if (s_lo_mid) {{ lo = mid; lo_alpha = mid_alpha; }}
        const bool s_lo_hi_next = in_bracket(lo, hi_next);
        if (s_lo_hi_next) {{ lo = hi_next; lo_alpha = hi_next_alpha; }}
        const bool swap_lo = s_lo_lo_next || s_lo_mid || s_lo_hi_next;

        const bool s_hi_hi_next = in_bracket(hi, hi_next);
        if (s_hi_hi_next) {{ hi = hi_next; hi_alpha = hi_next_alpha; }}
        const bool s_hi_mid = in_bracket(hi, mid);
        if (s_hi_mid) {{ hi = mid; hi_alpha = mid_alpha; }}
        const bool s_hi_lo_next = in_bracket(hi, lo_next);
        if (s_hi_lo_next) {{ hi = lo_next; hi_alpha = lo_next_alpha; }}
        const bool swap_hi = s_hi_hi_next || s_hi_mid || s_hi_lo_next;

        const bool ls_done = (!swap_lo && !swap_hi) || (lo.g < 0.0f && lo.g > -gtol) || (hi.g > 0.0f && hi.g < gtol);
        const bool improved = lo.c < p0.c || hi.c < p0.c;
        if (improved) alpha = lo.c < hi.c ? lo_alpha : hi_alpha;
        if (ls_done) break;
      }}
    }}

    qacc += alpha * search;
    Ma += alpha * mv;
    #pragma unroll
    for (int c = 0; c < NCH; c++) jaref[c] += alpha * jv[c];

    // ---- update_constraint: force, state, cost; track QUADRATIC changes ----
    prev_cost = cost;
    float lcost = 0.0f;
    unsigned long long changed[NCH];
    #pragma unroll
    for (int c = 0; c < NCH; c++) {{
      const int e = c * 64 + lane;
      const bool has = e < nefc;
      int ns = SATISFIED;
      float fr = 0.0f;
      if (has) {{
        const float D = Dd[c];
        const float ja = jaref[c];
        if (e < ne) {{
          fr = -D * ja;
          ns = QUADRATIC;
          lcost += 0.5f * D * ja * ja;
        }} else if (e < ne + nf) {{
          const float f = fl[c];
          const float rf = sdiv(f, D);
          if (ja <= -rf) {{
            fr = f;
            ns = LINEARNEG;
            lcost += -f * (0.5f * rf + ja);
          }} else if (ja >= rf) {{
            fr = -f;
            ns = LINEARPOS;
            lcost += -f * (0.5f * rf - ja);
          }} else {{
            fr = -D * ja;
            ns = QUADRATIC;
            lcost += 0.5f * D * ja * ja;
          }}
        }} else if (ja < 0.0f) {{
          fr = -D * ja;
          ns = QUADRATIC;
          lcost += 0.5f * D * ja * ja;
        }}
      }}
      changed[c] = __builtin_amdgcn_ballot_w64(has && ((st[c] == QUADRATIC) != (ns == QUADRATIC)));
      if (has) {{
        st[c] = ns;
        force[c] = fr;
      }}
    }}

    // ---- qfrc_constraint = J^T force, grad, gauss ----
    float qf = 0.0f;
    #pragma unroll
    for (int c = 0; c < NCH; c++) {{
      const int rows = min(64, nefc - c * 64);
      for (int l = 0; l < rows; l++) qf += *wp::address(J, w, c * 64 + l, di) * rl(force[c], l);
    }}
    qfrc = qf;
    grad = Ma - qs - qf;
    grad_dot = wsum(isdof ? grad * grad : 0.0f);
    const float gauss_sum = wsum(isdof ? (Ma - qs) * (qacc - qas) : 0.0f);
    gauss = 0.5f * gauss_sum;
    cost = wsum(lcost) + 0.5f * gauss_sum;

    // ---- incremental H update for rows that entered/left QUADRATIC ----
    int nchanged = 0;
    #pragma unroll
    for (int c = 0; c < NCH; c++) nchanged += __builtin_popcountll(changed[c]);
    if (nchanged > 0) {{
      float delta[N];
      #pragma unroll
      for (int j = 0; j < N; j++) delta[j] = 0.0f;
      #pragma unroll
      for (int c = 0; c < NCH; c++) {{
        unsigned long long mask = changed[c];
        while (mask) {{
          const int l = __builtin_ctzll(mask);
          mask &= mask - 1;
          const float jc = *wp::address(J, w, c * 64 + l, di);
          const float De = rl(Dd[c], l);
          const float s = rli(st[c], l) == QUADRATIC ? De * jc : -(De * jc);
          #pragma unroll
          for (int j = 0; j < N; j++) delta[j] += s * rl(jc, j);
        }}
      }}
      #pragma unroll
      for (int j = 0; j < N; j++) {{
        if (j <= lane && delta[j] != 0.0f) hc[j] += delta[j];
      }}
    }}

    // ---- Mgrad = H^-1 grad ----
    if (nchanged > 0 || !factored) {{
      float a[N];
      #pragma unroll
      for (int j = 0; j < N; j++) a[j] = hc[j];
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
      __syncthreads();
      if (isdof) {{
        #pragma unroll
        for (int j = 0; j < N; j++) Ls[lane * S + j] = a[j];
      }}
      __syncthreads();
      factored = true;
    }}
    float b = grad;
    #pragma unroll
    for (int k = 0; k < N; k++) {{
      const float yk = rl(b, k) * rl(dinv, k);
      b = lane == k ? yk : (isdof && lane > k ? b - Ls[lane * S + k] * yk : b);
    }}
    #pragma unroll
    for (int kk = 0; kk < N; kk++) {{
      const int k = N - 1 - kk;
      const float xk = rl(b, k) * rl(dinv, k);
      b = lane == k ? xk : (lane < k ? b - Ls[k * S + lane] * xk : b);
    }}
    Mgrad = b;

    // ---- search update and convergence ----
    search = -Mgrad;
    search_dot = wsum(isdof ? search * search : 0.0f);
    niter++;
    const float improvement = (prev_cost - cost) / scale;
    const float gradient = sqrtf(grad_dot) / scale;
    if (improvement < tolerance || gradient < tolerance || niter == iterations) break;
  }}

  if (isdof) {{
    *wp::address(qacc_io, w, lane) = qacc;
    *wp::address(Ma_io, w, lane) = Ma;
    *wp::address(qfrc_io, w, lane) = qfrc;
    *wp::address(grad_io, w, lane) = grad;
    *wp::address(Mgrad_io, w, lane) = Mgrad;
    *wp::address(search_io, w, lane) = search;
  }}
  #pragma unroll
  for (int c = 0; c < NCH; c++) {{
    const int e = c * 64 + lane;
    if (e < nefc) {{
      *wp::address(Jaref_io, w, e) = jaref[c];
      *wp::address(force_io, w, e) = force[c];
      *wp::address(state_io, w, e) = st[c];
    }}
  }}
  if (lane == 0) {{
    *wp::address(niter_io, w) = niter;
    *wp::address(search_dot_io, w) = search_dot;
    *wp::address(gauss_io, w) = gauss;
    *wp::address(cost_io, w) = cost;
    *wp::address(prev_cost_io, w) = prev_cost;
    *wp::address(grad_dot_io, w) = grad_dot;
    *wp::address(done, w) = true;
  }}
  """

  def solve_loop(
    opt_tolerance: wp.array(dtype=float),
    opt_ls_tolerance: wp.array(dtype=float),
    stat_meaninertia: wp.array(dtype=float),
    iterations: int,
    ne_in: wp.array(dtype=int),
    nf_in: wp.array(dtype=int),
    nefc_in: wp.array(dtype=int),
    njmax: int,
    qfrc_smooth: wp.array2d(dtype=float),
    qacc_smooth: wp.array2d(dtype=float),
    M: wp.array3d(dtype=float),
    J: wp.array3d(dtype=float),
    efc_D: wp.array2d(dtype=float),
    efc_frictionloss: wp.array2d(dtype=float),
    h: wp.array3d(dtype=float),
    qacc_io: wp.array2d(dtype=float),
    Ma_io: wp.array2d(dtype=float),
    qfrc_io: wp.array2d(dtype=float),
    force_io: wp.array2d(dtype=float),
    state_io: wp.array2d(dtype=int),
    niter_io: wp.array(dtype=int),
    Jaref_io: wp.array2d(dtype=float),
    grad_io: wp.array2d(dtype=float),
    Mgrad_io: wp.array2d(dtype=float),
    search_io: wp.array2d(dtype=float),
    search_dot_io: wp.array(dtype=float),
    gauss_io: wp.array(dtype=float),
    cost_io: wp.array(dtype=float),
    prev_cost_io: wp.array(dtype=float),
    grad_dot_io: wp.array(dtype=float),
    done: wp.array(dtype=bool),
    tid: int,
  ): ...

  solve_loop.__qualname__ = f"fused_newton_solve_{nv}_{row_chunks}_{ls_iterations}"
  solve_loop = wp.func_native(snippet)(solve_loop)

  @wp.kernel(module="unique", enable_backward=False, launch_bounds=64)
  def kernel(
    # Model:
    opt_tolerance: wp.array(dtype=float),
    opt_ls_tolerance: wp.array(dtype=float),
    stat_meaninertia: wp.array(dtype=float),
    opt_iterations: int,
    # Data in:
    ne_in: wp.array(dtype=int),
    nf_in: wp.array(dtype=int),
    nefc_in: wp.array(dtype=int),
    njmax_in: int,
    qfrc_smooth_in: wp.array2d(dtype=float),
    qacc_smooth_in: wp.array2d(dtype=float),
    M_in: wp.array3d(dtype=float),
    efc_J_in: wp.array3d(dtype=float),
    efc_D_in: wp.array2d(dtype=float),
    efc_frictionloss_in: wp.array2d(dtype=float),
    # In:
    ctx_h_in: wp.array3d(dtype=float),
    # Data out:
    qacc_out: wp.array2d(dtype=float),
    efc_Ma_out: wp.array2d(dtype=float),
    qfrc_constraint_out: wp.array2d(dtype=float),
    efc_force_out: wp.array2d(dtype=float),
    efc_state_out: wp.array2d(dtype=int),
    solver_niter_out: wp.array(dtype=int),
    # Out:
    ctx_Jaref_out: wp.array2d(dtype=float),
    ctx_grad_out: wp.array2d(dtype=float),
    ctx_Mgrad_out: wp.array2d(dtype=float),
    ctx_search_out: wp.array2d(dtype=float),
    ctx_search_dot_out: wp.array(dtype=float),
    ctx_gauss_out: wp.array(dtype=float),
    ctx_cost_out: wp.array(dtype=float),
    ctx_prev_cost_out: wp.array(dtype=float),
    ctx_grad_dot_out: wp.array(dtype=float),
    ctx_done_out: wp.array(dtype=bool),
  ):
    tid = wp.tid()
    solve_loop(
      opt_tolerance,
      opt_ls_tolerance,
      stat_meaninertia,
      opt_iterations,
      ne_in,
      nf_in,
      nefc_in,
      njmax_in,
      qfrc_smooth_in,
      qacc_smooth_in,
      M_in,
      efc_J_in,
      efc_D_in,
      efc_frictionloss_in,
      ctx_h_in,
      qacc_out,
      efc_Ma_out,
      qfrc_constraint_out,
      efc_force_out,
      efc_state_out,
      solver_niter_out,
      ctx_Jaref_out,
      ctx_grad_out,
      ctx_Mgrad_out,
      ctx_search_out,
      ctx_search_dot_out,
      ctx_gauss_out,
      ctx_cost_out,
      ctx_prev_cost_out,
      ctx_grad_dot_out,
      ctx_done_out,
      tid,
    )

  return kernel


def solve_fused(m: types.Model, d: types.Data, ctx: types.SolverContext):
  """Runs the Newton iterations of solver._solve; ctx must be initialized (init_context, solve_init_search)."""
  row_chunks = (d.njmax + 63) // 64
  warp_util.launch(
    fused_newton_solve(m.nv, row_chunks, m.opt.ls_iterations),
    dim=d.nworld * 64,
    inputs=[
      m.opt.tolerance,
      m.opt.ls_tolerance,
      m.stat.meaninertia,
      m.opt.iterations,
      d.ne,
      d.nf,
      d.nefc,
      d.njmax,
      d.qfrc_smooth,
      d.qacc_smooth,
      d.M,
      d.efc.J,
      d.efc.D,
      d.efc.frictionloss,
      ctx.h,
    ],
    outputs=[
      d.qacc,
      d.efc.Ma,
      d.qfrc_constraint,
      d.efc.force,
      d.efc.state,
      d.solver_niter,
      ctx.Jaref,
      ctx.grad,
      ctx.Mgrad,
      ctx.search,
      ctx.search_dot,
      ctx.gauss,
      ctx.cost,
      ctx.prev_cost,
      ctx.grad_dot,
      ctx.done,
    ],
    block_dim=64,
  )
