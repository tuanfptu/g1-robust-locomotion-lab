"""Serializable environment configuration, shared by train/eval/demo."""

from dataclasses import asdict, dataclass


@dataclass
class EnvConfig:
    task: str = "locomotion"
    recovery_pose: str = "supine"
    terrain: str = "flat"
    difficulty: float = 0.0
    episode_seconds: float = 20.0
    control_dt: float = 0.02
    physics_dt: float = 0.002
    command: tuple[float, float, float] = (0.4, 0.0, 0.0)
    random_commands: bool = True
    command_interval: float = 4.0
    domain_randomization: bool = False
    pushes: bool = False
    friction: float = 0.8
    mass_scale: float = 1.0
    push_impulse: float = 0.0
    push_angle: float = 0.0
    push_at: float = 5.0
    push_duration: float = 0.1
    tracking_sigma: float = 0.25
    linear_velocity_observation: bool = False
    action_scale: float = 0.35
    crouched_home: bool = False
    unitree_pd: bool = False
    phase_observation: bool = False
    leg_only: bool = False

    def __post_init__(self):
        if self.task not in {"locomotion", "recovery"}:
            raise ValueError("Unknown task")
        if self.recovery_pose not in {"supine", "prone", "side"}:
            raise ValueError("Unknown recovery pose")
        if self.task == "recovery" and (self.terrain != "flat" or self.leg_only):
            raise ValueError("Recovery requires flat terrain and full-body actions")
        if self.terrain not in {"flat", "slope", "rough", "stairs"}:
            raise ValueError(f"Unknown terrain: {self.terrain}")
        if not 0 <= self.difficulty <= 1:
            raise ValueError("difficulty must be in [0, 1]")
        if (
            min(
                self.episode_seconds,
                self.control_dt,
                self.physics_dt,
                self.command_interval,
                self.friction,
                self.mass_scale,
                self.push_duration,
                self.tracking_sigma,
                self.action_scale,
            )
            <= 0
        ):
            raise ValueError("Durations, friction and mass_scale must be positive")
        ratio = self.control_dt / self.physics_dt
        if abs(ratio - round(ratio)) > 1e-8 or ratio < 1:
            raise ValueError("control_dt must be an integer multiple of physics_dt")
        if self.push_impulse < 0 or len(self.command) != 3:
            raise ValueError("Invalid push impulse or command")

    def to_dict(self):
        return asdict(self)
