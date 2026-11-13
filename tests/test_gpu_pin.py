import pytest

from g1lab import gpu
from g1lab.fast_plan import command


def test_pin_cuda_sets_project_uuid_and_refuses_other_gpus(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("G1LAB_GPU_UUID", raising=False)
    assert gpu.pin_cuda() == gpu.PROJECT_GPU_UUID
    import os

    assert os.environ["CUDA_VISIBLE_DEVICES"] == gpu.PROJECT_GPU_UUID
    assert os.environ["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    for other in ["0", "1", "0,1", "GPU-4bb4ac0c-5f15-959c-dc6e-d6b9632cd45f"]:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", other)
        with pytest.raises(RuntimeError, match="may only use"):
            gpu.pin_cuda()


def test_isaac_command_disables_multi_gpu_rendering(tmp_path):
    args = command(tmp_path, "python", tmp_path / "robot.usd", 1.0)
    kit = args[args.index("--kit_args") + 1]
    assert "--/renderer/multiGpu/enabled=false" in kit
    assert "--/renderer/multiGpu/maxGpuCount=1" in kit
    assert "--/renderer/activeGpu=1" in kit
