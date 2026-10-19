.. meta::
  :description: MuJoCo Warp documentation
  :keywords: MuJoCo Warp, MJWarp, MuJoCo, Warp, ROCm, documentation, physics, simulation, robotics, GPU

.. _mujoco-warp-documentation-index:

********************************************************************
MuJoCo Warp on ROCm documentation
********************************************************************

Accelerate physics simulation for robotics research with MuJoCo Warp on ROCm
for AMD Instinct GPUs, enabling thousands of parallel, contact-rich simulation
worlds for reinforcement learning, motion planning, and large-scale evaluation.

`MuJoCo Warp (MJWarp) <https://mujoco.readthedocs.io/en/latest/mjwarp/index.html>`__ is a
GPU-accelerated version of the `MuJoCo <https://github.com/google-deepmind/mujoco>`__ physics
simulator, written in `Warp <https://github.com/AMD-Ecosystem/warp>`__. MJWarp delivers high-throughput,
accurate simulation of rigid bodies with contacts, soft bodies, cloth, signed distance fields,
and more, and is maintained upstream by Google DeepMind and NVIDIA as part of the
`Newton <https://github.com/newton-physics/newton>`__ project.

MuJoCo Warp on ROCm runs on AMD Instinct MI300X and MI325X GPUs through the
`ROCm port of Warp <https://github.com/AMD-Ecosystem/warp>`__, which compiles Warp kernels for the HIP backend.

MuJoCo Warp is part of the `AMD Simulation toolkit
<https://rocm.docs.amd.com/projects/rocm-simulation/en/latest/>`__.

The MuJoCo Warp public repository is located at `https://github.com/AMD-Ecosystem/mujoco_warp <https://github.com/AMD-Ecosystem/mujoco_warp>`__.

.. grid:: 2
  :gutter: 3

  .. grid-item-card:: Install

    * :doc:`Install MuJoCo Warp <install/mujoco-warp-install>`

  .. grid-item-card:: Examples

    * :doc:`Run a MuJoCo Warp example <examples/mujoco-warp-examples>`

  .. grid-item-card:: Reference

    * :doc:`ROCm configuration <reference/rocm-configuration>`
    * `API reference (upstream) <https://mujoco.readthedocs.io/en/latest/mjwarp/index.html>`__


You can find licensing information on the :doc:`Licensing <about/license>` page.
