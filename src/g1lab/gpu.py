"""Pin every GPU use of this project to one physical GPU.

The host has two RTX 5090s; GPU0 (PCI 0000:01:00.0) runs other people's work and must not be touched.
The project GPU is identified by UUID/PCI address, not by index: CUDA's default device order
(FASTEST_FIRST) need not match nvidia-smi numbering.
Override for another host with G1LAB_GPU_UUID and G1LAB_GPU_PCI.

Call pin_cuda() before torch / Isaac Sim are imported, and pin_egl() before mujoco is imported.
"""

import os
from pathlib import Path

PROJECT_GPU_UUID = "GPU-20e5a706-6231-50a3-4c9b-8e8d4db3c470"  # nvidia-smi index 1
PROJECT_GPU_PCI = "0000:05:00.0"
# Isaac Lab's headless kit files enable multi-GPU rendering by default; keep Kit on one GPU. With multi-GPU off,
# Kit renders on Omniverse device `activeGpu` (Vulkan/PCI order, here 0 = bus 01 = GPU0, 1 = bus 05 = GPU1).
# GPU0 is additionally hidden from CUDA, so Kit skips it ("CUDA in bad state") and could not create it anyway;
# without activeGpu=1 Kit ends up with no render device at all (training works, cameras/video do not).
KIT_ARGS = (
    "--/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false "
    "--/renderer/multiGpu/maxGpuCount=1 --/renderer/activeGpu=1"
)


def project_gpu():
    return os.environ.get("G1LAB_GPU_UUID", PROJECT_GPU_UUID), os.environ.get(
        "G1LAB_GPU_PCI", PROJECT_GPU_PCI
    )


def pin_cuda():
    """Expose only the project GPU to CUDA; refuse a conflicting CUDA_VISIBLE_DEVICES."""
    uuid, _ = project_gpu()
    current = os.environ.get("CUDA_VISIBLE_DEVICES")
    if current not in (None, "", uuid):
        raise RuntimeError(
            f"CUDA_VISIBLE_DEVICES={current!r}, but this project may only use {uuid}. "
            "Unset it (the project pins the GPU itself) or set G1LAB_GPU_UUID deliberately."
        )
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = uuid
    return uuid


def _egl_device_pci(index_to_drm):
    result = {}
    for index, drm in index_to_drm.items():
        if not drm:
            continue
        device = Path("/sys/class/drm") / Path(drm).name / "device"
        try:
            result[index] = device.resolve().name
        except OSError:
            pass
    return result


def pin_egl():
    """Select the EGL device of the project GPU for MuJoCo rendering, without creating any context.

    Uses MuJoCo's own eglQueryDevicesEXT wrapper so indices match MUJOCO_EGL_DEVICE_ID exactly.
    """
    import ctypes

    _, pci = project_gpu()
    current = os.environ.get("MUJOCO_EGL_DEVICE_ID")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    if os.environ["PYOPENGL_PLATFORM"] != "egl":
        raise RuntimeError("PYOPENGL_PLATFORM must be egl for headless rendering")
    from mujoco.egl import egl_ext

    query = ctypes.CFUNCTYPE(ctypes.c_char_p, ctypes.c_void_p, ctypes.c_int32)(
        egl_ext.eglGetProcAddress("eglQueryDeviceStringEXT")
    )
    egl_drm_device_file_ext, egl_extensions = 0x3233, 0x3055
    drm, nvidia = {}, set()
    for i, device in enumerate(egl_ext.eglQueryDevicesEXT()):
        handle = ctypes.cast(device, ctypes.c_void_p)
        value = query(handle, egl_drm_device_file_ext)
        drm[i] = value.decode() if value else None
        # glvnd lists each card once per vendor library (NVIDIA and Mesa); keep the NVIDIA driver device.
        if b"EGL_NV_device_cuda" in (query(handle, egl_extensions) or b""):
            nvidia.add(i)
    matches = [i for i, bus in _egl_device_pci(drm).items() if bus == pci and i in nvidia]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one EGL device on {pci}, found {matches} in {drm}")
    if current not in (None, "", str(matches[0])):
        raise RuntimeError(
            f"MUJOCO_EGL_DEVICE_ID={current!r} is not the project GPU (EGL device {matches[0]})"
        )
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(matches[0])
    return matches[0]
