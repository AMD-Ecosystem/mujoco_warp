.. meta::
  :description: What is MuJoCo Warp?
  :keywords: MuJoCo Warp, MJWarp, MuJoCo, Warp, documentation, physics, simulation, robotics, GPU, AMD, ROCm, overview, introduction

.. _what-is-mujoco-warp:

********************************************************************
What is MuJoCo Warp?
********************************************************************

`MuJoCo Warp (MJWarp) <https://mujoco.readthedocs.io/en/latest/mjwarp/index.html>`__ is a
GPU-accelerated version of the `MuJoCo <https://github.com/google-deepmind/mujoco>`__
(Multi-Joint dynamics with Contact) physics simulator. It is written in
`Warp <https://github.com/AMD-Ecosystem/warp>`__, a Python framework that just-in-time (JIT) compiles
Python functions into GPU kernels. MJWarp simulates many independent copies of a scene, called
*worlds*, in parallel on a single GPU, which makes it well suited to reinforcement learning
and other workloads that need millions of simulation steps per second.

MJWarp mirrors the MuJoCo data model. You load a model with the standard ``mujoco`` Python
bindings, copy it to the GPU with ``mujoco_warp.put_model`` and ``mujoco_warp.put_data``,
and advance all worlds at once with ``mujoco_warp.step``.

MuJoCo Warp on ROCm runs these kernels on AMD Instinct GPUs through the
`ROCm port of Warp <https://github.com/ROCm/warp>`__, which adds a HIP backend to Warp.
MJWarp itself is backend-agnostic, so the same Python code runs on AMD and NVIDIA GPUs.

Why MuJoCo Warp?
====================================================================

MuJoCo Warp is designed for GPU-accelerated robotics simulation for the following reasons:

- It batches thousands of simulation worlds on one GPU, delivering high throughput
  for data-hungry workloads such as reinforcement learning and policy evaluation.

- It reproduces MuJoCo's physics, including its soft-contact model, constraint solvers,
  and integrators, so results stay consistent with CPU MuJoCo.

- It keeps simulation state on the GPU as Warp arrays, avoiding host-device copies
  inside training loops and interoperating with PyTorch and JAX.

- The ROCm port adds optimizations tuned for AMD Instinct GPUs, letting you run
  existing MJWarp workloads on AMD hardware without code changes.

Features and use cases
====================================================================

MuJoCo Warp provides the following key features:

- **Batched rigid-body dynamics:** Simulates articulated rigid bodies across many
  parallel worlds with Euler, RK4, implicit, and implicit-fast integrators.

- **Contact-rich collision detection:** Supports primitive, convex mesh, heightfield,
  and signed distance field (SDF) collisions with configurable broadphase.

- **Constraint solvers:** Provides Newton and conjugate gradient (CG) solvers for
  contacts, joint limits, equality constraints, and friction.

- **Deformables (experimental):** Simulates flex objects such as cloth and soft bodies.

- **Sensors and actuators:** Supports the MuJoCo sensor and actuator models, including
  ray casting and rangefinder sensors.

- **Batch rendering:** Includes a GPU ray-tracing renderer for rendering many cameras
  across parallel worlds.


MuJoCo Warp is commonly used in the following scenarios:

- **Reinforcement learning:** Train locomotion and manipulation policies with
  frameworks such as `MuJoCo Playground <https://github.com/google-deepmind/mujoco_playground>`__
  and `mjlab <https://github.com/mujocolab/mjlab>`__.

- **Large-scale evaluation:** Evaluate controllers and policies across thousands of
  randomized scenes in parallel.

- **Motion planning and model-predictive control:** Roll out many candidate trajectories
  at once to search for good controls.

- **Simulation research:** Prototype new solvers, collision algorithms, and contact models
  on GPUs.
