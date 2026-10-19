.. meta::
  :description: MuJoCo Warp examples
  :keywords: MuJoCo Warp, MJWarp, programming, ROCm, example, sample, tutorial, physics, simulation

.. _run-a-mujoco-warp-example:

********************************************************************
Run a MuJoCo Warp example
********************************************************************

The `benchmarks <https://github.com/AMD-Ecosystem/mujoco_warp/tree/amd-integration/benchmarks>`__ and
`notebooks <https://github.com/AMD-Ecosystem/mujoco_warp/tree/amd-integration/notebooks>`__ folders in the
`https://github.com/AMD-Ecosystem/mujoco_warp <https://github.com/AMD-Ecosystem/mujoco_warp>`__ repository
have models and example code that you can use to run MuJoCo Warp.

Simulate many worlds in parallel
====================================================================

This example loads the humanoid model from the repository, copies it to the GPU,
and steps 4096 copies of the model in parallel.

1. Save the following code snippet to a Python script named ``mujoco_warp_example.py``
   in the root of the ``mujoco_warp`` repository.

   .. code-block:: python

      import time

      import mujoco
      import warp as wp

      import mujoco_warp as mjw

      # Load the model on the CPU with the MuJoCo Python bindings
      mjm = mujoco.MjModel.from_xml_path("benchmarks/humanoid/humanoid.xml")
      mjd = mujoco.MjData(mjm)
      mujoco.mj_forward(mjm, mjd)

      # Copy the model and data to the GPU, with 4096 parallel worlds
      nworld = 4096
      m = mjw.put_model(mjm)
      d = mjw.put_data(mjm, mjd, nworld=nworld)

      # The first steps compile the Warp kernels for the GPU
      for _ in range(10):
        mjw.step(m, d)
      wp.synchronize()

      # Time a rollout across all worlds
      nstep = 1000
      start = time.perf_counter()
      for _ in range(nstep):
        mjw.step(m, d)
      wp.synchronize()
      elapsed = time.perf_counter() - start

      print(f"device: {wp.get_device()}")
      print(f"steps per second: {nworld * nstep / elapsed:,.0f}")

      # Copy the first world back to the CPU
      mjw.get_data_into(mjd, mjm, d)
      print(f"world 0 torso height: {mjd.qpos[2]:.3f}")

2. Run the script.

   .. code-block:: bash

      python mujoco_warp_example.py


Benchmark a model
====================================================================

Use ``mjwarp-testspeed`` to measure simulation throughput for any MuJoCo XML model:

.. code-block:: bash

   mjwarp-testspeed benchmarks/humanoid/humanoid.xml --nworld=8192 --nstep=1000

To print a per-function timing breakdown, add ``--event_trace``. To see all options, run
``mjwarp-testspeed --help``.

To run the benchmark suite in the repository, use ``benchmarks/run.py``. The ``-f`` option
selects benchmarks with a regular expression:

.. code-block:: bash

   python benchmarks/run.py -f humanoid
