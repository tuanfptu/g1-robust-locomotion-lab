from pathlib import Path

import numpy as np
import pytest

from g1lab.fast_plan import command, stage_settings
from g1lab.lab_policy import TermHistory


def test_term_major_history_order_scale_and_reset():
    history = TermHistory(
        {"a": {"history_length": 3, "scale": [2.0, 2.0]}, "b": {"history_length": 3, "scale": [1.0]}}
    )
    history.reset({"a": [1, 2], "b": [10]})
    obs = history.update({"a": [3, 4], "b": [20]})
    np.testing.assert_array_equal(obs, [2, 4, 2, 4, 6, 8, 10, 10, 20])
    history.reset({"a": [0, 0], "b": [0]})
    np.testing.assert_array_equal(history.update({"a": [0, 0], "b": [0]}), np.zeros(9))


def test_fast_stage_keeps_limits_and_does_not_resume_onnx(tmp_path):
    args = command(tmp_path, "python", tmp_path / "robot.usd", 2.0, checkpoint=tmp_path / "run/model.pt")
    assert "env.commands.base_velocity.ranges.lin_vel_x=[0.0,2.0]" in args
    assert "env.rewards.gait.params.threshold=0.45" in args
    assert "agent.algorithm.learning_rate=0.0001" in args
    assert "--resume" in args
    assert args[args.index("--checkpoint") + 1] == r"model\.pt"
    path_args = command(tmp_path, "python", tmp_path / "robot with spaces.usd", 1.0)
    assert any('spawn.usd_path="' in a and 'robot with spaces.usd"' in a for a in path_args)
    assert not any("effort_limit" in a or "velocity_limit_sim" in a for a in args)
    with pytest.raises(ValueError, match="ONNX"):
        command(tmp_path, "python", tmp_path / "robot.usd", 1.0, checkpoint=tmp_path / "policy.onnx")
    with pytest.raises(ValueError):
        stage_settings(10.0)


@pytest.mark.skipif(
    not Path("vendor/unitree_rl_lab").is_dir()
    or not Path("vendor/unitree_mujoco/unitree_robots/g1").is_dir(),
    reason="optional vendor assets are not installed",
)
def test_native_lab_policy_joint_mapping_and_repeatability():
    pytest.importorskip("onnxruntime")
    pytest.importorskip("yaml")
    from g1lab.lab_policy import LabReference

    policy = LabReference()
    assert len(set(policy.actuator_ids)) == 29
    assert policy.session.get_inputs()[0].shape == [1, 480]
    traces = []
    for _ in range(2):
        policy.reset(42)
        rows = [policy.step([0.3, 0, 0]) for _ in range(10)]
        traces.append(policy.data.qpos.copy())
        assert all(not r["fallen"] for r in rows)
    np.testing.assert_allclose(traces[0], traces[1], atol=1e-12)

    # Telemetry must not alter control/physics, and actuator utilization is bounded.
    policy.reset(42)
    diagnostic_rows = [policy.step([0.3, 0, 0], diagnostics=True) for _ in range(10)]
    np.testing.assert_allclose(traces[0], policy.data.qpos, atol=1e-12)
    assert all(0 <= r["requested_torque_clipped_fraction"] <= 1 for r in diagnostic_rows)
    assert all(0 <= r["peak_torque_utilization"] <= 1.00001 for r in diagnostic_rows)
    assert all(0 <= r["feet_in_contact"] <= 2 for r in diagnostic_rows)
