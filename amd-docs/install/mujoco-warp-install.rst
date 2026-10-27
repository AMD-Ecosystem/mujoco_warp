.. meta::
  :description: installing MuJoCo Warp for ROCm
  :keywords: installation instructions, Docker, AMD, ROCm, MuJoCo Warp, MJWarp, Warp, MI325X, MI350X/MI355X

.. _mujoco-warp-on-rocm-installation:

********************************************************************
MuJoCo Warp on ROCm installation
********************************************************************

System requirements
====================================================================

To use MuJoCo Warp `3.8.1 <https://github.com/AMD-Ecosystem/mujoco_warp/tree/release/v3.8.1>`__, you need the following prerequisites:

- **ROCm version:** `7.2.0 <https://rocm.docs.amd.com/en/docs-7.2.0/>`__ (MI325X), `10.1.0 <https://rocm.docs.amd.com/en/docs-10.1.0/>`__ (MI350X, MI355X)
- **Operating system:** Ubuntu 24.04
- **GPU platform:** AMD Instinct™ MI325X, MI350X, MI355X
- **PyTorch:** `2.9.1 <https://github.com/ROCm/pytorch/tree/release/2.9>`__ (MI325X), `2.14.0 <https://github.com/ROCm/pytorch/tree/release/2.14>`__ (MI350X, MI355X)
- **Python:** `3.12 <https://www.python.org/downloads/release/python-31213/>`__

.. note::

  MuJoCo Warp does not declare ``warp-lang`` as a dependency. The ``warp-lang`` wheels on PyPI
  are built for NVIDIA CUDA only, so on ROCm you must build Warp from source first and then install
  MuJoCo Warp into the same Python environment.

Install MuJoCo Warp
====================================================================

To install MuJoCo Warp on ROCm, use the Dockerfile in the repository. The Dockerfile
builds the image, copies the sources, and builds the Warp library along with all
dependencies.

The Dockerfile supports ROCm 7.2.0 for MI325X and ROCm 10.1.0 for MI350X and MI355X.

Using the Dockerfile
--------------------------------------------------------------------

Docker is the recommended method to set up a MuJoCo Warp environment, as it avoids
dependency conflicts. The built image includes MuJoCo Warp, Warp, PyTorch, ROCm,
and the other requirements.

1. Clone the repository.

   .. code-block:: bash

      git clone -b release/v3.8.1 https://github.com/AMD-Ecosystem/mujoco_warp.git
      cd mujoco_warp

2. Build the Docker image for your GPU architecture.

   .. note::

      To check which architecture your GPUs use, run ``rocm_agent_enumerator | sort -u``.
      ``gfx942`` corresponds to MI325X, and ``gfx950`` corresponds to MI350X and MI355X.

   .. tab-set::

      .. tab-item:: AMD Instinct MI325X (ROCm 7.2.0)

         .. code-block:: bash

            docker build --network=host -f docker/rocm/Dockerfile --target mujocowarp_gfx942 -t mujocowarp:rocm_7.2_ubuntu24.04_py3.12_torch2.9.1 .

      .. tab-item:: AMD Instinct MI350X/MI355X (ROCm 10.1.0)

         .. code-block:: bash

            docker build --network=host -f docker/rocm/Dockerfile --target mujocowarp_gfx950 -t mujocowarp:rocm10.1.0_ubuntu24.04_py3.12_torch2.14.0 .

   This builds an image with MuJoCo Warp 3.8.1 and the required dependencies, including
   PyTorch 2.9.1 (MI325X) or PyTorch 2.14.0 (MI350X, MI355X).

3. Run the Docker container.

   .. tab-set::

      .. tab-item:: AMD Instinct MI325X (ROCm 7.2.0)

         .. code-block:: bash

            docker run -it --rm \
               --name my_mujoco_warp \
               --device /dev/kfd \
               --device /dev/dri \
               --privileged \
               --network=host \
               --group-add video \
               --cap-add=SYS_PTRACE \
               --security-opt seccomp=unconfined \
               --ipc=host \
               --shm-size=16g \
               mujocowarp:rocm_7.2_ubuntu24.04_py3.12_torch2.9.1

      .. tab-item:: AMD Instinct MI350X/MI355X (ROCm 10.1.0)

         .. code-block:: bash

            docker run -it --rm \
               --name my_mujoco_warp \
               --device /dev/kfd \
               --device /dev/dri \
               --privileged \
               --network=host \
               --group-add video \
               --cap-add=SYS_PTRACE \
               --security-opt seccomp=unconfined \
               --ipc=host \
               --shm-size=16g \
               mujocowarp:rocm10.1.0_ubuntu24.04_py3.12_torch2.14.0

.. _test-the-mujoco-warp-installation:

Test the MuJoCo Warp installation
====================================================================

After starting the container, verify that MuJoCo Warp and ROCm are working correctly.
The container has Warp and MuJoCo Warp installed.

1. Verify that Warp detects your AMD GPUs.

   .. code-block:: bash

      python -c "import warp as wp; wp.init(); print(wp.get_devices())"

   AMD GPUs appear as ``cuda:N`` devices. The expected output on a system with one GPU is:

   .. code-block:: text

      [Device(name='cpu'), Device(name='cuda:0')]

2. Verify that MuJoCo Warp is installed.

   .. code-block:: bash

      python -c "import mujoco_warp; print(mujoco_warp.__version__)"

   Expected output:

   .. code-block:: text

      3.8.1

3. Optionally, run the unit tests.

   .. code-block:: bash

      pip install pytest pytest-xdist
      pytest mujoco_warp -n 8

If these commands succeed, MuJoCo Warp 3.8.1 has been installed successfully.
Proceed to :doc:`run a MuJoCo Warp example <../examples/mujoco-warp-examples>`.
