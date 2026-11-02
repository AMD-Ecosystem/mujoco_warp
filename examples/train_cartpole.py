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

"""Trains a cartpole balancing policy with PPO, keeping simulation and training data on the GPU."""

import mujoco
import torch
import torch.nn as nn
import warp as wp

import mujoco_warp as mjw

CARTPOLE_XML = """
<mujoco model="cartpole">
  <option timestep="0.01">
    <flag contact="disable"/>
  </option>
  <visual>
    <headlight ambient="0.3 0.3 0.3" diffuse="0.5 0.5 0.5"/>
    <quality shadowsize="4096"/>
    <rgba haze="0.15 0.25 0.35 1"/>
  </visual>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.4 0.6 0.8" rgb2="0 0 0" width="512" height="512"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="8 4" reflectance="0.2"/>
    <material name="rail" rgba="0.6 0.6 0.6 1" specular="0.5"/>
    <material name="cart" rgba="0.9 0.5 0.2 1" specular="0.3" shininess="0.5"/>
    <material name="pole" rgba="0.2 0.5 0.9 1" specular="0.3" shininess="0.5"/>
  </asset>
  <worldbody>
    <light pos="0 -1 4" dir="0 0.25 -1" directional="true" castshadow="true" diffuse="0.6 0.6 0.6"/>
    <camera name="front" pos="0 -3.2 0.9" xyaxes="1 0 0 0 0.3 1"/>
    <geom name="floor" type="plane" size="0 0 0.05" pos="0 0 -0.15" material="grid"/>
    <geom name="rail" type="box" size="2.4 0.02 0.02" material="rail"/>
    <body name="cart">
      <joint name="slider" type="slide" axis="1 0 0"/>
      <geom name="cart" type="box" size="0.2 0.15 0.1" mass="1" material="cart"/>
      <body name="pole">
        <joint name="hinge" type="hinge" axis="0 1 0" damping="0.01"/>
        <geom name="pole" type="capsule" fromto="0 0 0 0 0 1" size="0.045" mass="0.1" material="pole"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="slider" gear="10" ctrlrange="-1 1" ctrllimited="true"/>
  </actuator>
</mujoco>
"""

NWORLD = 4096
FRAME_SKIP = 2
MAX_EPISODE_STEPS = 500
HORIZON = 32
ITERATIONS = 100
EPOCHS = 4
MINIBATCHES = 4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP = 0.2
LR = 1e-3
OBS_DIM = 5
ACT_DIM = 1


@wp.kernel
def observe(qpos: wp.array2d(dtype=float), qvel: wp.array2d(dtype=float), obs: wp.array2d(dtype=float)):
  world = wp.tid()
  theta = qpos[world, 1]
  obs[world, 0] = qpos[world, 0]
  obs[world, 1] = wp.cos(theta)
  obs[world, 2] = wp.sin(theta)
  obs[world, 3] = qvel[world, 0]
  obs[world, 4] = qvel[world, 1]


@wp.kernel
def reward_and_fall(
  qpos: wp.array2d(dtype=float),
  reward: wp.array(dtype=float),
  fell: wp.array(dtype=bool),
):
  world = wp.tid()
  x = qpos[world, 0]
  theta = qpos[world, 1]
  fallen = wp.abs(theta) > 0.8 or wp.abs(x) > 2.4
  fell[world] = fallen
  if fallen:
    reward[world] = 0.0
  else:
    # +1 for every step the pole stays up, minus a small penalty for drifting away from the center.
    reward[world] = 1.0 - 0.01 * x * x


@wp.kernel
def randomize_state(
  seed: int,
  done: wp.array(dtype=bool),
  qpos: wp.array2d(dtype=float),
  qvel: wp.array2d(dtype=float),
):
  world = wp.tid()
  if not done[world]:
    return
  rng = wp.rand_init(seed, world)
  qpos[world, 0] = wp.randf(rng, -0.5, 0.5)
  qpos[world, 1] = wp.randf(rng, -0.2, 0.2)
  qvel[world, 0] = wp.randf(rng, -0.1, 0.1)
  qvel[world, 1] = wp.randf(rng, -0.1, 0.1)


class CartpoleEnv:
  """Thousands of cartpoles simulated in parallel by MuJoCo Warp."""

  def __init__(self, nworld: int):
    """Creates the model, the simulation state for nworld cartpoles, and the observation buffers."""
    self.nworld = nworld
    self.mjm = mujoco.MjModel.from_xml_string(CARTPOLE_XML)
    self.m = mjw.put_model(self.mjm)
    self.d = mjw.make_data(self.mjm, nworld=nworld)

    self.obs_wp = wp.zeros((nworld, OBS_DIM), dtype=float)
    self.reward_wp = wp.zeros(nworld, dtype=float)
    self.fell_wp = wp.zeros(nworld, dtype=bool)
    self.done_wp = wp.ones(nworld, dtype=bool)

    # Zero-copy PyTorch views of Warp and MuJoCo Warp memory.
    self.obs = wp.to_torch(self.obs_wp)
    self.reward = wp.to_torch(self.reward_wp)
    self.fell = wp.to_torch(self.fell_wp)
    self.done = wp.to_torch(self.done_wp)
    self.ctrl = wp.to_torch(self.d.ctrl)
    self.episode_steps = torch.zeros(nworld, dtype=torch.int32, device=self.obs.device)
    self.seed = 0

  def _reset_done_worlds(self):
    mjw.reset_data(self.m, self.d, self.done_wp)
    self.seed += 1
    wp.launch(randomize_state, dim=self.nworld, inputs=[self.seed, self.done_wp, self.d.qpos, self.d.qvel])
    self.episode_steps[self.done] = 0

  def reset(self) -> torch.Tensor:
    """Resets every world to a random start state and returns the observations."""
    self.done.fill_(True)
    self._reset_done_worlds()
    wp.launch(observe, dim=self.nworld, inputs=[self.d.qpos, self.d.qvel], outputs=[self.obs_wp])
    return self.obs

  def step(self, action: torch.Tensor):
    """Applies actions, steps physics, and resets finished worlds.

    Returns observations, rewards, done flags, and the lengths of the episodes that just finished.
    """
    self.ctrl.copy_(action.clamp(-1.0, 1.0))
    for _ in range(FRAME_SKIP):
      mjw.step(self.m, self.d)

    wp.launch(reward_and_fall, dim=self.nworld, inputs=[self.d.qpos], outputs=[self.reward_wp, self.fell_wp])
    self.episode_steps += 1
    torch.logical_or(self.fell, self.episode_steps >= MAX_EPISODE_STEPS, out=self.done)
    finished_lengths = self.episode_steps[self.done].clone()

    self._reset_done_worlds()
    wp.launch(observe, dim=self.nworld, inputs=[self.d.qpos, self.d.qvel], outputs=[self.obs_wp])
    return self.obs, self.reward, self.done, finished_lengths


class ActorCritic(nn.Module):
  """Gaussian policy (actor) and state-value estimator (critic)."""

  def __init__(self):
    """Creates two small MLPs and a learnable action standard deviation."""
    super().__init__()
    self.actor = nn.Sequential(nn.Linear(OBS_DIM, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, ACT_DIM))
    self.critic = nn.Sequential(nn.Linear(OBS_DIM, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))
    self.log_std = nn.Parameter(torch.full((ACT_DIM,), -0.5))

  def dist(self, obs):
    """Returns the action distribution for the given observations."""
    return torch.distributions.Normal(self.actor(obs), self.log_std.exp())

  def value(self, obs):
    """Returns the estimated value of the given observations."""
    return self.critic(obs).squeeze(-1)


def train():
  torch.manual_seed(0)
  env = CartpoleEnv(NWORLD)
  device = env.obs.device
  policy = ActorCritic().to(device)
  optimizer = torch.optim.Adam(policy.parameters(), lr=LR)

  obs_buf = torch.zeros(HORIZON, NWORLD, OBS_DIM, device=device)
  act_buf = torch.zeros(HORIZON, NWORLD, ACT_DIM, device=device)
  logp_buf = torch.zeros(HORIZON, NWORLD, device=device)
  val_buf = torch.zeros(HORIZON, NWORLD, device=device)
  rew_buf = torch.zeros(HORIZON, NWORLD, device=device)
  done_buf = torch.zeros(HORIZON, NWORLD, device=device)

  obs = env.reset()
  for it in range(ITERATIONS):
    # 1. Collect a rollout from every world in parallel.
    lengths = []
    falls = 0
    with torch.no_grad():
      for t in range(HORIZON):
        dist = policy.dist(obs)
        action = dist.sample()
        obs_buf[t] = obs
        act_buf[t] = action
        logp_buf[t] = dist.log_prob(action).sum(-1)
        val_buf[t] = policy.value(obs)

        obs, reward, done, finished = env.step(action)
        rew_buf[t] = reward
        done_buf[t] = done.float()
        lengths.append(finished)
        falls += env.fell.sum()

      # 2. Generalized advantage estimation.
      next_value = policy.value(obs)
      adv_buf = torch.zeros_like(rew_buf)
      last_adv = torch.zeros(NWORLD, device=device)
      for t in reversed(range(HORIZON)):
        not_done = 1.0 - done_buf[t]
        delta = rew_buf[t] + GAMMA * next_value * not_done - val_buf[t]
        last_adv = delta + GAMMA * GAE_LAMBDA * not_done * last_adv
        adv_buf[t] = last_adv
        next_value = val_buf[t]
      ret_buf = adv_buf + val_buf

    # 3. PPO update.
    b_obs, b_act = obs_buf.reshape(-1, OBS_DIM), act_buf.reshape(-1, ACT_DIM)
    b_logp, b_adv, b_ret = logp_buf.reshape(-1), adv_buf.reshape(-1), ret_buf.reshape(-1)
    b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)
    batch = b_obs.shape[0]
    for _ in range(EPOCHS):
      for idx in torch.randperm(batch, device=device).chunk(MINIBATCHES):
        dist = policy.dist(b_obs[idx])
        ratio = (dist.log_prob(b_act[idx]).sum(-1) - b_logp[idx]).exp()
        pg_loss = -torch.min(ratio * b_adv[idx], ratio.clamp(1 - CLIP, 1 + CLIP) * b_adv[idx]).mean()
        v_loss = 0.5 * (policy.value(b_obs[idx]) - b_ret[idx]).pow(2).mean()
        loss = pg_loss + v_loss - 0.001 * dist.entropy().sum(-1).mean()
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

    finished = torch.cat(lengths)
    if (it + 1) % 10 == 0 or it == 0:
      mean_len = finished.float().mean().item() if finished.numel() else float("nan")
      print(
        f"iteration {it + 1:3d} | mean reward per step {rew_buf.mean().item():.3f} | "
        f"falls {int(falls):5d} | mean episode length {mean_len:6.1f} / {MAX_EPISODE_STEPS}"
      )

  torch.save(policy.state_dict(), "cartpole_policy.pt")
  print("Saved policy to cartpole_policy.pt")


if __name__ == "__main__":
  wp.init()
  with wp.ScopedStream(wp.stream_from_torch()):
    train()
