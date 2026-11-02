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

"""Renders the trained cartpole policy from train_cartpole.py to an animated GIF."""

import mujoco
import torch
import warp as wp
from PIL import Image
from train_cartpole import FRAME_SKIP
from train_cartpole import ActorCritic
from train_cartpole import CartpoleEnv

import mujoco_warp as mjw

WIDTH, HEIGHT = 480, 360
NSTEPS = 300
RENDER_EVERY = 2

wp.init()
with wp.ScopedStream(wp.stream_from_torch()):
  env = CartpoleEnv(nworld=1)
  policy = ActorCritic().to(env.obs.device)
  policy.load_state_dict(torch.load("cartpole_policy.pt"))

  mjd = mujoco.MjData(env.mjm)
  renderer = mujoco.Renderer(env.mjm, height=HEIGHT, width=WIDTH)
  frames = []

  obs = env.reset()
  with torch.no_grad():
    for t in range(NSTEPS):
      # Use the mean action, without exploration noise, for the demo.
      obs, _, _, _ = env.step(policy.actor(obs))
      if t % RENDER_EVERY == 0:
        # Copy world 0 back into a CPU MjData and render it with MuJoCo's renderer.
        mjw.get_data_into(mjd, env.mjm, env.d)
        mujoco.mj_forward(env.mjm, mjd)
        renderer.update_scene(mjd, camera="front")
        frames.append(Image.fromarray(renderer.render()))

frame_ms = int(1000 * env.mjm.opt.timestep * FRAME_SKIP * RENDER_EVERY)
frames[0].save("cartpole.gif", save_all=True, append_images=frames[1:], duration=frame_ms, loop=0)
print(f"Saved {len(frames)} frames to cartpole.gif")
