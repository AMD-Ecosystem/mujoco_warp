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

"""Renders the trained Go1 policy from train_go1.py following a sequence of velocity commands."""

import mujoco
import torch
import warp as wp
from PIL import Image
from train_go1 import POLICY_DT
from train_go1 import ActorCritic
from train_go1 import Go1Env

import mujoco_warp as mjw

WIDTH, HEIGHT = 640, 360
RENDER_EVERY = 2

# Each command is (seconds, forward m/s, sideways m/s, turning rad/s).
COMMANDS = [
  (1.0, 0.0, 0.0, 0.0),
  (3.0, 1.0, 0.0, 0.0),
  (3.0, 0.6, 0.0, 0.5),
  (2.0, 0.0, 0.5, 0.0),
  (1.5, 0.0, 0.0, 0.0),
]

wp.init()
with wp.ScopedStream(wp.stream_from_torch()):
  env = Go1Env(nworld=1, resample_commands=False)
  policy = ActorCritic().to(env.obs.device)
  policy.load_state_dict(torch.load("go1_policy.pt"))

  mjd = mujoco.MjData(env.mjm)
  renderer = mujoco.Renderer(env.mjm, height=HEIGHT, width=WIDTH)
  # Follow the robot with a camera that tracks the trunk.
  camera = mujoco.MjvCamera()
  camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
  camera.trackbodyid = env.mjm.body("trunk").id
  camera.distance, camera.elevation, camera.azimuth = 1.8, -20.0, 135.0
  frames = []

  obs = env.reset()
  step = 0
  with torch.no_grad():
    for seconds, *command in COMMANDS:
      env.command[:] = torch.tensor(command, device=env.command.device)
      for _ in range(round(seconds / POLICY_DT)):
        obs, _, _, _ = env.step(policy.act(obs))
        if step % RENDER_EVERY == 0:
          # Copy world 0 back into a CPU MjData and render it with MuJoCo's renderer.
          mjw.get_data_into(mjd, env.mjm, env.d)
          mujoco.mj_forward(env.mjm, mjd)
          renderer.update_scene(mjd, camera=camera)
          frames.append(Image.fromarray(renderer.render()))
        step += 1

frame_ms = int(1000 * POLICY_DT * RENDER_EVERY)
frames[0].save("go1.gif", save_all=True, append_images=frames[1:], duration=frame_ms, loop=0)
print(f"Saved {len(frames)} frames to go1.gif")
