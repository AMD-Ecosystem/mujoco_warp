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

import math

import torch
import warp as wp
from PIL import Image
from PIL import ImageDraw
from train_cartpole import FRAME_SKIP
from train_cartpole import ActorCritic
from train_cartpole import CartpoleEnv

WIDTH, HEIGHT = 480, 240
PIXELS_PER_METER = 80
NFRAMES = 300


def draw_frame(x: float, theta: float) -> Image.Image:
  """Draws the cart and pole as seen from the front, with the rail centered in the image."""
  image = Image.new("RGB", (WIDTH, HEIGHT), "white")
  draw = ImageDraw.Draw(image)
  rail_y = HEIGHT - 50
  to_px = lambda px, pz: (WIDTH / 2 + PIXELS_PER_METER * px, rail_y - PIXELS_PER_METER * pz)

  draw.line([to_px(-2.4, 0), to_px(2.4, 0)], fill=(80, 80, 80), width=3)
  draw.rectangle([to_px(x - 0.2, 0.1), to_px(x + 0.2, -0.1)], fill=(230, 130, 50))
  draw.line([to_px(x, 0), to_px(x + math.sin(theta), math.cos(theta))], fill=(50, 130, 230), width=7)
  return image


wp.init()
with wp.ScopedStream(wp.stream_from_torch()):
  env = CartpoleEnv(nworld=1)
  policy = ActorCritic().to(env.obs.device)
  policy.load_state_dict(torch.load("cartpole_policy.pt"))
  # Zero-copy view of the cart position (qpos[:, 0]) and pole angle (qpos[:, 1]).
  qpos = wp.to_torch(env.d.qpos)

  frames = []
  obs = env.reset()
  with torch.no_grad():
    for _ in range(NFRAMES):
      # Use the mean action, without exploration noise, for the demo.
      obs, _, _, _ = env.step(policy.actor(obs))
      x, theta = qpos[0, :2].tolist()
      frames.append(draw_frame(x, theta))

frame_ms = int(1000 * env.mjm.opt.timestep * FRAME_SKIP)
frames[0].save("cartpole.gif", save_all=True, append_images=frames[1:], duration=frame_ms, loop=0)
print(f"Saved {len(frames)} frames to cartpole.gif")
