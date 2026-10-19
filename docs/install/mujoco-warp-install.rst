.. meta::
  :description: installing MuJoCo Warp for ROCm
  :keywords: installation instructions, Docker, AMD, ROCm, MuJoCo Warp, MJWarp, Warp

.. _mujoco-warp-on-rocm-installation:

********************************************************************
MuJoCo Warp on ROCm installation
********************************************************************

System requirements
====================================================================

To use MuJoCo Warp `3.8.1 <https://github.com/AMD-Ecosystem/mujoco_warp/tree/amd-integration>`__, you need the following prerequisites:

.. list-table::
   :stub-columns: 1
   :align: left

   * - GPU platform
     - AMD Instinct™ MI325X, MI355X
   * - ROCm version
     - 7.2.0 (MI325X), 10.1.0 (MI350X, MI355X)
   * - Operating system
     - Ubuntu 24.04
   * - Python
     - 3.12
   * - PyTorch
     - 2.9.1 (MI325X), 2.14.0 (MI350X/MI355X)

.. note::

  MuJoCo Warp does not declare ``warp-lang`` as a dependency. The ``warp-lang`` wheels on PyPI
  are built for NVIDIA CUDA only, so on ROCm you must build Warp from source first and then install
  MuJoCo Warp into the same Python environment.

Install MuJoCo Warp
================================================================================

To install MuJoCo Warp on ROCm, you must use the Dockerfile in the repository. This Dockerfile will build the image, copy the sources and
build the Warp library along with all the dependencies. 

The Dockerfile can be used for ROCm 7.2 for MI325x and ROCm 10.1.0 for MI350x or MI355x.

Using the Dockerfile
--------------------------------------------------------------------------------------

Docker is the recommended method to set up a Warp environment, as it avoids
dependency conflicts. The tested, prebuilt images include MuJoCo Warp, PyTorch,
ROCm, and all other requirements.

1. Clone the repository

   .. code-block:: bash

      git clone -b release/v3.8.1 https://github.com/AMD-Ecosystem/mujoco_warp.git
      cd mujoco_warp/

2. Build the Docker image for your GPU architecture.

   .. tab-set::

      .. tab-item:: MI325X (ROCm 7.2.0)

         .. code-block:: bash

            docker build --network=host -f docker/rocm/Dockerfile --target mujocowarp_gfx942 -t mujocowarp_gfx942:rocm_7.2_ubuntu24.04_py3.12_torch2.9.1 .

      .. tab-item:: MI350X/MI355X (ROCm 10.1.0)

         .. code-block:: bash

            docker build --network=host -f docker/rocm/Dockerfile --target mujocowarp_gfx950 -t mujocowarp_gfx950:rocm10.1.0_ubuntu24.04_py3.12_torch2.14.0 .

3. Start a Docker container using the image.

   .. tab-set::

      .. tab-item:: MI325X (ROCm 7.2.0)

         .. code-block:: bash

            docker run -it --rm \
            --privileged \
            --network=host --device=/dev/kfd \
            --device=/dev/dri --group-add video \
            --name=my_mujoco_warp --cap-add=SYS_PTRACE \
            --security-opt seccomp=unconfined \
            --ipc=host --shm-size 16G \
            mujocowarp:rocm_7.2_ubuntu24.04_py3.12_torch2.9.1

      .. tab-item:: MI350X/MI355X (ROCm 10.1.0)

         .. code-block:: bash

            docker run -it --rm \
            --privileged \
            --network=host --device=/dev/kfd \
            --device=/dev/dri --group-add video \
            --name=my_mujoco_warp --cap-add=SYS_PTRACE \
            --security-opt seccomp=unconfined \
            --ipc=host --shm-size 16G \
            mujocowarp:rocm10.1.0_ubuntu24.04_py3.12_torch2.14.0

3. The above step creates a Docker container with Warp pre-installed. Continue to
   :ref:`test-the-mujoco-warp-installation`.

.. _test-the-mujoco-warp-installation:

Test the MuJoCo Warp installation
======================================================================================

1. Verify that MuJoCo Warp detects your AMD GPUs:

   .. code-block:: bash

      python -c "import warp as wp; wp.init(); print(wp.get_devices())"

   AMD GPUs appear as ``cuda:N`` devices. The expected output on a system with one GPU is:

   .. code-block:: text

      [Device(name='cpu'), Device(name='cuda:0')]

2. Verify that MuJoCo Warp is installed:

   .. code-block:: bash

      python -c "import mujoco_warp; print(mujoco_warp.__version__)"

   Expected output:

   .. code-block:: text

      3.8.1

3. Optionally, run the unit tests.

   .. code-block:: bash

      pip install pytest pytest-xdist
      pytest mujoco_warp -n 8

If these commands succeed, MuJoCo Warp ``3.8.1`` has been installed successfully. You can now use
MuJoCo Warp in your projects.
