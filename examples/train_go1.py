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

"""Trains a Unitree Go1 quadruped to walk at commanded velocities with PPO, entirely on the GPU."""

import pathlib
import subprocess

import mujoco
import torch
import torch.nn as nn
import warp as wp

import mujoco_warp as mjw

MENAGERIE_URL = "https://github.com/google-deepmind/mujoco_menagerie.git"
MENAGERIE_REF = "affef0836947b64cc06c4ab1cbf0152835693374"
MENAGERIE_DIR = pathlib.Path("mujoco_menagerie")

NWORLD = 4096
DECIMATION = 4  # Physics steps per policy step: 4 x 5 ms = 20 ms, so the policy runs at 50 Hz.
POLICY_DT = 0.02
EPISODE_STEPS = 1000  # 20 seconds.
HORIZON = 24
ITERATIONS = 500
EPOCHS = 5
MINIBATCHES = 4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP = 0.2
LR = 1e-3
DESIRED_KL = 0.01
ENTROPY_COEF = 0.01

NJOINT = 12
NFOOT = 4
OBS_DIM = 3 + 3 + 3 + NJOINT + NJOINT + NJOINT + 3
ACT_DIM = NJOINT
ACTION_SCALE = 0.25  # Radians of joint target offset per unit of action.

# Commanded forward, sideways, and turning velocity ranges (m/s, m/s, rad/s).
COMMAND_RANGE = (1.0, 0.5, 0.5)
STANDING_FRACTION = 0.1


def load_go1_model() -> mujoco.MjModel:
  """Loads the Unitree Go1 scene from MuJoCo Menagerie, downloading it on first use."""
  if not (MENAGERIE_DIR / "unitree_go1").exists():
    subprocess.run(["git", "clone", "-q", "--filter=blob:none", "--no-checkout", MENAGERIE_URL, MENAGERIE_DIR], check=True)
    subprocess.run(["git", "-C", MENAGERIE_DIR, "sparse-checkout", "set", "unitree_go1"], check=True)
    subprocess.run(["git", "-C", MENAGERIE_DIR, "checkout", "-q", MENAGERIE_REF], check=True)

  spec = mujoco.MjSpec.from_file(str(MENAGERIE_DIR / "unitree_go1" / "scene.xml"))
  # A 5 ms time step with fewer solver iterations keeps thousands of worlds fast and stable.
  spec.option.timestep = 0.005
  spec.option.iterations = 10
  spec.option.ls_iterations = 20
  return spec.compile()


@wp.func
def base_state(qpos: wp.array2d(dtype=float), qvel: wp.array2d(dtype=float), world: int):
  """Returns the trunk velocity, angular velocity, and gravity direction in the trunk frame."""
  # MuJoCo stores quaternions as (w, x, y, z) and Warp as (x, y, z, w).
  quat = wp.quat(qpos[world, 4], qpos[world, 5], qpos[world, 6], qpos[world, 3])
  lin_vel = wp.quat_rotate_inv(quat, wp.vec3(qvel[world, 0], qvel[world, 1], qvel[world, 2]))
  ang_vel = wp.vec3(qvel[world, 3], qvel[world, 4], qvel[world, 5])
  gravity = wp.quat_rotate_inv(quat, wp.vec3(0.0, 0.0, -1.0))
  return lin_vel, ang_vel, gravity


@wp.kernel
def observe(
  qpos: wp.array2d(dtype=float),
  qvel: wp.array2d(dtype=float),
  default_qpos: wp.array(dtype=float),
  last_action: wp.array2d(dtype=float),
  command: wp.array2d(dtype=float),
  obs: wp.array2d(dtype=float),
):
  world = wp.tid()
  lin_vel, ang_vel, gravity = base_state(qpos, qvel, world)
  for i in range(3):
    obs[world, i] = lin_vel[i]
    obs[world, 3 + i] = ang_vel[i]
    obs[world, 6 + i] = gravity[i]
    obs[world, 9 + 3 * NJOINT + i] = command[world, i]
  for j in range(NJOINT):
    obs[world, 9 + j] = qpos[world, 7 + j] - default_qpos[j]
    obs[world, 9 + NJOINT + j] = qvel[world, 6 + j]
    obs[world, 9 + 2 * NJOINT + j] = last_action[world, j]


@wp.kernel
def compute_reward(
  qpos: wp.array2d(dtype=float),
  qvel: wp.array2d(dtype=float),
  site_xpos: wp.array2d(dtype=wp.vec3),
  feet: wp.vec4i,
  command: wp.array2d(dtype=float),
  action: wp.array2d(dtype=float),
  last_action: wp.array2d(dtype=float),
  default_qpos: wp.array(dtype=float),
  pose_std: wp.array(dtype=float),
  joint_lo: wp.array(dtype=float),
  joint_hi: wp.array(dtype=float),
  just_reset: wp.array(dtype=bool),
  foot_pos_prev: wp.array2d(dtype=wp.vec3),
  reward: wp.array(dtype=float),
  fell: wp.array(dtype=bool),
):
  world = wp.tid()
  lin_vel, ang_vel, gravity = base_state(qpos, qvel, world)
  cmd_x = command[world, 0]
  cmd_y = command[world, 1]
  cmd_yaw = command[world, 2]
  moving = wp.sqrt(cmd_x * cmd_x + cmd_y * cmd_y) + wp.abs(cmd_yaw) > 0.05

  # Track the commanded velocity while keeping the trunk level.
  lin_err = (cmd_x - lin_vel[0]) ** 2.0 + (cmd_y - lin_vel[1]) ** 2.0 + lin_vel[2] ** 2.0
  ang_err = (cmd_yaw - ang_vel[2]) ** 2.0 + ang_vel[0] ** 2.0 + ang_vel[1] ** 2.0
  tilt = gravity[0] ** 2.0 + gravity[1] ** 2.0

  # Stay close to the default pose, more strictly when standing still, and inside the joint limits.
  pose_err = float(0.0)
  limit_err = float(0.0)
  for j in range(NJOINT):
    q = qpos[world, 7 + j]
    std = pose_std[j]
    if not moving:
      std = 0.2 * std
    pose_err += (q - default_qpos[j]) ** 2.0 / (std * std)
    limit_err += wp.max(joint_lo[j] - q, 0.0) + wp.max(q - joint_hi[j], 0.0)

  # Penalize jerky actions.
  action_rate = float(0.0)
  for j in range(NJOINT):
    action_rate += (action[world, j] - last_action[world, j]) ** 2.0

  # Lift swinging feet to 10 cm and don't let feet slide while they're on the ground.
  clearance = float(0.0)
  slip = float(0.0)
  for f in range(NFOOT):
    pos = site_xpos[world, feet[f]]
    speed = float(0.0)
    if not just_reset[world]:
      vel = (pos - foot_pos_prev[world, f]) / POLICY_DT
      speed = wp.sqrt(vel[0] * vel[0] + vel[1] * vel[1])
    clearance += wp.abs(pos[2] - 0.1) * speed
    if pos[2] < 0.035:
      slip += speed * speed
    foot_pos_prev[world, f] = pos

  r = 2.0 * wp.exp(-lin_err / 0.25) + 2.0 * wp.exp(-ang_err / 0.5) + wp.exp(-tilt / 0.2)
  r += wp.exp(-pose_err / float(NJOINT))
  r -= limit_err + 0.1 * action_rate
  if moving:
    r -= 2.0 * clearance + 0.1 * slip
  reward[world] = r * POLICY_DT

  # The robot has fallen once the trunk tilts more than 70 degrees from upright.
  fell[world] = gravity[2] > -0.342


@wp.kernel
def reset_pose(
  done: wp.array(dtype=bool),
  default_qpos: wp.array(dtype=float),
  qpos: wp.array2d(dtype=float),
):
  world = wp.tid()
  if not done[world]:
    return
  qpos[world, 2] = 0.3
  for j in range(NJOINT):
    qpos[world, 7 + j] = default_qpos[j]


class Go1Env:
  """Thousands of Unitree Go1 quadrupeds simulated in parallel by MuJoCo Warp."""

  def __init__(self, nworld: int, resample_commands: bool = True):
    """Loads the Go1 model, creates the state of nworld robots, and allocates buffers."""
    self.nworld = nworld
    self.resample_commands = resample_commands
    self.mjm = load_go1_model()
    self.m = mjw.put_model(self.mjm)
    self.d = mjw.make_data(self.mjm, nworld=nworld, nconmax=32, njmax=128)

    # Per-joint constants, in qpos order (joint 0 is the floating base). The default pose is the
    # model's "home" keyframe. Calf joints may stray from it twice as far as hip and thigh joints.
    names = [self.mjm.joint(j).name for j in range(1, self.mjm.njnt)]
    default_qpos = self.mjm.key("home").qpos[7:]
    pose_std = [0.6 if "calf" in n else 0.3 for n in names]
    lo, hi = self.mjm.jnt_range[1:].T
    mid, half = (lo + hi) / 2, (hi - lo) / 2
    self.default_qpos_wp = wp.array(default_qpos, dtype=float)
    self.pose_std_wp = wp.array(pose_std, dtype=float)
    # Soft joint limits: the middle 90% of each joint's range.
    self.joint_lo_wp = wp.array(mid - 0.9 * half, dtype=float)
    self.joint_hi_wp = wp.array(mid + 0.9 * half, dtype=float)
    self.feet = wp.vec4i(*[self.mjm.site(name).id for name in ("FR", "FL", "RR", "RL")])

    self.obs_wp = wp.zeros((nworld, OBS_DIM), dtype=float)
    self.reward_wp = wp.zeros(nworld, dtype=float)
    self.fell_wp = wp.zeros(nworld, dtype=bool)
    self.done_wp = wp.ones(nworld, dtype=bool)
    self.action_wp = wp.zeros((nworld, ACT_DIM), dtype=float)
    self.last_action_wp = wp.zeros((nworld, ACT_DIM), dtype=float)
    self.command_wp = wp.zeros((nworld, 3), dtype=float)
    self.foot_pos_prev_wp = wp.zeros((nworld, NFOOT), dtype=wp.vec3)

    # Zero-copy PyTorch views of Warp and MuJoCo Warp memory.
    self.obs = wp.to_torch(self.obs_wp)
    self.reward = wp.to_torch(self.reward_wp)
    self.fell = wp.to_torch(self.fell_wp)
    self.done = wp.to_torch(self.done_wp)
    self.action = wp.to_torch(self.action_wp)
    self.last_action = wp.to_torch(self.last_action_wp)
    self.command = wp.to_torch(self.command_wp)
    self.ctrl = wp.to_torch(self.d.ctrl)
    self.default_ctrl = wp.to_torch(self.default_qpos_wp)

    device = self.obs.device
    self.episode_steps = torch.zeros(nworld, dtype=torch.int32, device=device)
    self.command_steps = torch.zeros(nworld, dtype=torch.int32, device=device)
    self.command_range = torch.tensor(COMMAND_RANGE, device=device)

  def _update_commands(self):
    """Gives a new random velocity command to worlds that were reset or whose command expired."""
    self.command_steps -= 1
    if not self.resample_commands:
      return
    expired = self.done | (self.command_steps <= 0)
    new = (2 * torch.rand_like(self.command) - 1) * self.command_range
    new *= (torch.rand(self.nworld, 1, device=new.device) > STANDING_FRACTION).float()
    self.command.copy_(torch.where(expired.unsqueeze(-1), new, self.command))
    # Each command lasts between 3 and 8 seconds.
    steps = torch.randint_like(self.command_steps, 150, 400)
    self.command_steps.copy_(torch.where(expired, steps, self.command_steps))

  def _reset_done_worlds(self):
    mjw.reset_data(self.m, self.d, self.done_wp)
    wp.launch(reset_pose, dim=self.nworld, inputs=[self.done_wp, self.default_qpos_wp, self.d.qpos])
    self.last_action.masked_fill_(self.done.unsqueeze(-1), 0.0)
    self.episode_steps.masked_fill_(self.done, 0)

  def _observe(self):
    wp.launch(
      observe,
      dim=self.nworld,
      inputs=[self.d.qpos, self.d.qvel, self.default_qpos_wp, self.last_action_wp, self.command_wp],
      outputs=[self.obs_wp],
    )

  def reset(self) -> torch.Tensor:
    """Resets every world to the standing pose with a new command and returns the observations."""
    self.done.fill_(True)
    self._reset_done_worlds()
    self._update_commands()
    self._observe()
    return self.obs

  def step(self, action: torch.Tensor):
    """Applies actions, steps physics, and resets finished worlds.

    Returns observations, rewards, done flags, and the lengths of the episodes that just finished.
    """
    # The actuators are ordered like the joints, so each action offsets its joint's position target.
    self.action.copy_(action)
    self.ctrl.copy_(self.default_ctrl + ACTION_SCALE * action)
    for _ in range(DECIMATION):
      mjw.step(self.m, self.d)

    wp.launch(
      compute_reward,
      dim=self.nworld,
      inputs=[
        self.d.qpos,
        self.d.qvel,
        self.d.site_xpos,
        self.feet,
        self.command_wp,
        self.action_wp,
        self.last_action_wp,
        self.default_qpos_wp,
        self.pose_std_wp,
        self.joint_lo_wp,
        self.joint_hi_wp,
        self.done_wp,
      ],
      outputs=[self.foot_pos_prev_wp, self.reward_wp, self.fell_wp],
    )
    self.last_action.copy_(action)
    self.episode_steps += 1
    torch.logical_or(self.fell, self.episode_steps >= EPISODE_STEPS, out=self.done)
    finished_lengths = self.episode_steps[self.done].clone()

    self._reset_done_worlds()
    self._update_commands()
    self._observe()
    return self.obs, self.reward, self.done, finished_lengths


class ObsNormalizer(nn.Module):
  """Normalizes observations with a running mean and variance."""

  def __init__(self, dim: int):
    """Starts with zero mean and unit variance."""
    super().__init__()
    self.register_buffer("mean", torch.zeros(dim))
    self.register_buffer("var", torch.ones(dim))
    self.register_buffer("count", torch.tensor(1e-4))

  def update(self, x: torch.Tensor):
    """Merges the statistics of a batch of observations into the running statistics."""
    batch_mean, batch_var, n = x.mean(0), x.var(0, unbiased=False), x.shape[0]
    total = self.count + n
    delta = batch_mean - self.mean
    self.mean += delta * n / total
    self.var = (self.var * self.count + batch_var * n + delta**2 * self.count * n / total) / total
    self.count = total

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Returns the normalized observations."""
    return (x - self.mean) / torch.sqrt(self.var + 1e-8)


def mlp(in_dim: int, out_dim: int) -> nn.Sequential:
  return nn.Sequential(
    nn.Linear(in_dim, 512), nn.ELU(), nn.Linear(512, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(), nn.Linear(128, out_dim)
  )


class ActorCritic(nn.Module):
  """Gaussian policy (actor) and state-value estimator (critic) with an observation normalizer."""

  def __init__(self):
    """Creates the normalizer, two MLPs, and a learnable action standard deviation."""
    super().__init__()
    self.normalizer = ObsNormalizer(OBS_DIM)
    self.actor = mlp(OBS_DIM, ACT_DIM)
    self.critic = mlp(OBS_DIM, 1)
    self.log_std = nn.Parameter(torch.zeros(ACT_DIM))

  def dist(self, obs):
    """Returns the action distribution for the given normalized observations."""
    return torch.distributions.Normal(self.actor(obs), self.log_std.exp())

  def value(self, obs):
    """Returns the estimated value of the given normalized observations."""
    return self.critic(obs).squeeze(-1)

  def act(self, raw_obs):
    """Returns the mean action for raw observations, for running the trained policy."""
    return self.actor(self.normalizer(raw_obs))


def train():
  torch.manual_seed(0)
  env = Go1Env(NWORLD)
  device = env.obs.device
  policy = ActorCritic().to(device)
  optimizer = torch.optim.Adam(policy.parameters(), lr=LR)
  lr = LR

  obs_buf = torch.zeros(HORIZON, NWORLD, OBS_DIM, device=device)
  act_buf = torch.zeros(HORIZON, NWORLD, ACT_DIM, device=device)
  mu_buf = torch.zeros(HORIZON, NWORLD, ACT_DIM, device=device)
  logp_buf = torch.zeros(HORIZON, NWORLD, device=device)
  val_buf = torch.zeros(HORIZON, NWORLD, device=device)
  rew_buf = torch.zeros(HORIZON, NWORLD, device=device)
  done_buf = torch.zeros(HORIZON, NWORLD, device=device)

  raw_obs = env.reset()
  for it in range(ITERATIONS):
    # 1. Collect a rollout from every world in parallel.
    lengths = []
    falls = 0
    vel_err = 0.0
    with torch.no_grad():
      for t in range(HORIZON):
        policy.normalizer.update(raw_obs)
        obs = policy.normalizer(raw_obs)
        dist = policy.dist(obs)
        action = dist.sample()
        obs_buf[t] = obs
        act_buf[t] = action
        mu_buf[t] = dist.mean
        logp_buf[t] = dist.log_prob(action).sum(-1)
        val_buf[t] = policy.value(obs)

        raw_obs, reward, done, finished = env.step(action)
        rew_buf[t] = reward
        done_buf[t] = done.float()
        lengths.append(finished)
        falls += env.fell.sum()
        vel_err += (env.command[:, :2] - raw_obs[:, :2]).norm(dim=-1).mean()

      # 2. Generalized advantage estimation.
      next_value = policy.value(policy.normalizer(raw_obs))
      adv_buf = torch.zeros_like(rew_buf)
      last_adv = torch.zeros(NWORLD, device=device)
      for t in reversed(range(HORIZON)):
        not_done = 1.0 - done_buf[t]
        delta = rew_buf[t] + GAMMA * next_value * not_done - val_buf[t]
        last_adv = delta + GAMMA * GAE_LAMBDA * not_done * last_adv
        adv_buf[t] = last_adv
        next_value = val_buf[t]
      ret_buf = adv_buf + val_buf

    # 3. PPO update, adapting the learning rate to keep each update's KL divergence near DESIRED_KL.
    b_obs, b_act, b_mu = obs_buf.reshape(-1, OBS_DIM), act_buf.reshape(-1, ACT_DIM), mu_buf.reshape(-1, ACT_DIM)
    b_logp, b_adv, b_ret = logp_buf.reshape(-1), adv_buf.reshape(-1), ret_buf.reshape(-1)
    b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)
    batch = b_obs.shape[0]
    for _ in range(EPOCHS):
      for idx in torch.randperm(batch, device=device).chunk(MINIBATCHES):
        old_std = policy.log_std.exp().detach()
        dist = policy.dist(b_obs[idx])
        with torch.no_grad():
          kl = torch.distributions.kl_divergence(torch.distributions.Normal(b_mu[idx], old_std), dist).sum(-1).mean()
          if kl > 2 * DESIRED_KL:
            lr = max(1e-5, lr / 1.5)
          elif kl < DESIRED_KL / 2:
            lr = min(1e-2, lr * 1.5)
          for group in optimizer.param_groups:
            group["lr"] = lr

        ratio = (dist.log_prob(b_act[idx]).sum(-1) - b_logp[idx]).exp()
        pg_loss = -torch.min(ratio * b_adv[idx], ratio.clamp(1 - CLIP, 1 + CLIP) * b_adv[idx]).mean()
        v_loss = (policy.value(b_obs[idx]) - b_ret[idx]).pow(2).mean()
        loss = pg_loss + v_loss - ENTROPY_COEF * dist.entropy().sum(-1).mean()
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

    finished = torch.cat(lengths)
    if (it + 1) % 50 == 0 or it == 0:
      mean_len = finished.float().mean().item() * POLICY_DT if finished.numel() else float("nan")
      print(
        f"iteration {it + 1:4d} | reward per second {rew_buf.mean().item() / POLICY_DT:5.2f} | "
        f"falls {int(falls):5d} | mean episode length {mean_len:5.1f} s | "
        f"velocity error {vel_err.item() / HORIZON:.2f} m/s",
        flush=True,
      )

  torch.save(policy.state_dict(), "go1_policy.pt")
  print("Saved policy to go1_policy.pt")


if __name__ == "__main__":
  wp.init()
  with wp.ScopedStream(wp.stream_from_torch()):
    train()
