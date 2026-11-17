"""Unattended, offline sprint training loop on GPU1: train -> evaluate -> keep best -> recover/adapt -> report.

Each round resumes PPO from the current checkpoint for a fixed number of iterations on the Isaac sprint task,
carrying the speed-curriculum level over, then evaluates the final checkpoint on an Isaac speed ladder and
(asynchronously, on CPU) on the MuJoCo benchmark. Rules:

* best = highest Isaac max sustained speed (survival >= 90 %, mean vx >= 85 % of command, 10 s hold);
* regression (NaN, failed run, or max speed more than 1 m/s below best) -> roll back to best, halve LR;
* capability collapse (speed-only top speed >= `capability_drop` m/s below its best for 2 rounds in a row, which
  the yaw-inclusive metric does not see) -> roll back to the anchor = MuJoCo-validated best, if any;
* stall (no improvement for `stall_rounds`) -> apply the next knob from a fixed, logged list;
* a run that dies before its first iteration is a launch/config error, not a training regression: a rejected
  Hydra override reverts the knob that introduced it; otherwise retry with growing back-off and halt after
  `max_startup_failures` in a row (never loop on the same crash);
* safety: every Isaac/MuJoCo child runs without network (unshare -rn) and with CUDA pinned to GPU1; a guard
  stops everything if a process of this project ever appears on GPU0.

Physical limits (torques, joint velocity limits, PD gains, robot model) are never changed. The 11 m/s goal
is a ceiling for the curriculum, not an expectation. Uses only the Python standard library.
"""

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
GPU0_UUID = "GPU-4bb4ac0c-5f15-959c-dc6e-d6b9632cd45f"
# Knobs never raise heading/twist bonuses: in run 2026-11-16 doubling track_ang_vel_z made jogging straight pay more
# than running fast (rounds 18-30, top speed 5.4 -> 2.0 m/s). Speed tracking stays the dominant term.
KNOBS = [
    {"name": "track_weight_2.5", "overrides": {"env.rewards.track_lin_vel_xy.weight": 2.5}},
    {
        "name": "accel_3.0",
        "overrides": {
            "env.commands.base_velocity.accel_limit": 3.0,
            "env.commands.base_velocity.decel_limit": 3.5,
        },
    },
    {
        "name": "flight_weights_up",
        # reward v2 terms only (v1's swing_air_time no longer exists; Hydra rejects unknown keys)
        "overrides": {"env.rewards.flight_phase.weight": 1.5, "env.rewards.accel_push_off.weight": 2.0},
    },
    {"name": "lr_reset_3e-4", "lr": 3.0e-4},
]


class Autopilot:
    def __init__(self, args):
        self.args = args
        self.out = Path(args.output).resolve()
        self.state_path = self.out / "state.json"
        self.stop = threading.Event()
        self.child = None
        self.mujoco_jobs = []
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text())
            self.log(f"resuming from state round {self.state['round']}")
        else:
            self.out.mkdir(parents=True, exist_ok=False)
            self.state = {
                "started": datetime.now().isoformat(timespec="seconds"),
                "deadline": time.time() + args.hours * 3600 if args.hours > 0 else None,
                "round": 0,
                # "scratch": random initialisation (no checkpoint) for the first round.
                "latest_ckpt": None
                if args.init_checkpoint == "scratch"
                else str(Path(args.init_checkpoint).resolve()),
                "level": args.start_level,
                "lr": args.lr,
                "knobs_applied": [],
                "overrides": dict(item.split("=", 1) for item in args.override),
                "stall": 0,
                "consecutive_failures": 0,
                "best_isaac": None,
                "best_mujoco": None,
                "rounds": [],
                "events": [],
            }
            self.save()

    # ------------------------------------------------------------------ utils
    def log(self, message):
        line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
        print(line, flush=True)
        with (self.out / "autopilot.log").open("a") as handle:
            handle.write(line + "\n")

    def event(self, kind, **data):
        self.state["events"].append(
            {"time": datetime.now().isoformat(timespec="seconds"), "kind": kind, **data}
        )
        self.log(f"EVENT {kind} {data}")

    def save(self):
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=2))
        tmp.replace(self.state_path)
        self.write_report()

    def env(self):
        env = dict(os.environ)
        env.pop("CUDA_VISIBLE_DEVICES", None)  # g1lab.gpu pins GPU1 by UUID and rejects anything else
        env["OMNI_KIT_ACCEPT_EULA"] = "YES"
        return env

    def run(self, argv, log_path, timeout, cwd=ROOT):
        """Run a child offline (unshare -rn) in its own process group; return code or None on timeout/abort."""
        full = ["unshare", "-rn", *map(str, argv)]
        with open(log_path, "w") as log:
            self.child = subprocess.Popen(
                full, cwd=cwd, env=self.env(), stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            deadline = time.time() + timeout
            while self.child.poll() is None:
                if self.stop.is_set() or time.time() > deadline:
                    self.kill_child()
                    return None
                time.sleep(2)
            code = self.child.returncode
        self.child = None
        return code

    def kill_child(self):
        """SIGTERM the child's process group, then SIGKILL it: Isaac/Kit can ignore SIGTERM and survive as an
        orphan holding the GPU (seen once, ran 15 h)."""
        if not self.child:
            return
        pgid = self.child.pid
        for sig, wait in ((signal.SIGTERM, 20), (signal.SIGKILL, 15)):
            try:
                os.killpg(pgid, sig)
            except ProcessLookupError:
                break
            end = time.time() + wait
            while time.time() < end:
                try:
                    os.killpg(pgid, 0)
                except ProcessLookupError:
                    break
                time.sleep(1)
            else:
                continue
            break
        try:
            self.child.wait(5)
        except Exception:
            pass

    def guard(self):
        """Stop everything if any process of this workspace shows up on GPU0 (compute or graphics)."""
        while not self.stop.is_set():
            try:
                text = subprocess.run(
                    ["nvidia-smi", "-i", GPU0_UUID, "-q", "-d", "PIDS"],
                    capture_output=True,
                    text=True,
                    timeout=20,
                ).stdout
                for pid in re.findall(r"Process ID\s*:\s*(\d+)", text):
                    try:
                        cmd = (
                            Path(f"/proc/{pid}/cmdline")
                            .read_bytes()
                            .replace(b"\0", b" ")
                            .decode(errors="ignore")
                        )
                        cwd = os.readlink(f"/proc/{pid}/cwd")
                    except OSError:
                        continue
                    if str(WORKSPACE) in cmd or str(WORKSPACE) in cwd:
                        self.event("gpu0_violation_abort", pid=pid, cmd=cmd[:300])
                        self.stop.set()
                        self.kill_child()
                        self.save()
                        return
            except Exception as error:  # nvidia-smi hiccup: keep guarding
                self.log(f"guard warning: {error}")
            time.sleep(2)

    # ------------------------------------------------------------------ steps
    def train(self, round_dir, round_no):
        a = self.args
        overrides = {
            "env.commands.base_velocity.limit_ranges.lin_vel_x": f"[0.0,{a.speed_limit}]",
            "env.commands.base_velocity.ranges.lin_vel_x": f"[0.0,{self.state['level']:.2f}]",
            **self.state["overrides"],
        }
        argv = [
            a.isaac_python,
            "scripts/prepare_lab_fast.py",
            "--profile",
            "sprint",
            "--python",
            a.isaac_python,
            "--usd",
            a.usd,
            "--envs",
            a.envs,
            "--iterations",
            a.round_iterations,
            "--seed",
            a.seed + round_no,
            "--learning-rate",
            self.state["lr"],
            "--output",
            round_dir / "train",
            "--execute",
        ]
        if self.state["latest_ckpt"]:
            argv += ["--checkpoint", self.state["latest_ckpt"]]
        warm = round_no == 1 or self.state.pop("critic_warmup_next", False)  # e.g. after a reward change
        if warm and a.critic_warmup and self.state["latest_ckpt"]:
            argv += ["--critic-warmup", a.critic_warmup]
        for key, value in overrides.items():
            argv.append(f"--train-arg={key}={value}")
        code = self.run(argv, round_dir / "launcher.log", timeout=a.round_iterations * 12 + 900)
        meta_path = round_dir / "train" / "command.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        run_dirs = [Path(p) for p in meta.get("run_dirs", [])]
        ckpts = sorted(
            (c for d in run_dirs for c in d.glob("model_*.pt")), key=lambda c: int(c.stem.split("_")[1])
        )
        log_text = (
            (round_dir / "train" / "train.log").read_text(errors="ignore")
            if (round_dir / "train" / "train.log").exists()
            else ""
        )
        levels = re.findall(r"Curriculum/lin_vel_cmd_levels: ([\d.]+)", log_text)
        rewards = re.findall(r"Mean reward: (-?[\d.]+|nan|inf)", log_text)
        lengths = re.findall(r"Mean episode length: ([\d.]+)", log_text)
        return {
            "returncode": code,
            "checkpoint": str(ckpts[-1]) if ckpts else None,
            "run_dir": str(run_dirs[0]) if run_dirs else None,
            "level": float(levels[-1]) if levels else None,
            "last_reward": rewards[-1] if rewards else None,
            "last_episode_length": float(lengths[-1]) if lengths else None,
            "nan": bool(re.search(r"\bnan\b", " ".join(rewards), re.IGNORECASE)),
            "iterations_logged": len(re.findall(r"Learning iteration", log_text)),
        }

    def isaac_eval(self, ckpt, round_dir, top):
        """Two independent evaluations (seeds 0, 1; 32 robots per speed); scores are the worse of the two."""
        results = []
        for seed in (0, 1):
            ev = self.isaac_eval_once(ckpt, round_dir, top, seed)
            if ev is None:
                return None
            results.append(ev)
        merged = dict(results[0])
        for key in ("max_sustained_speed_mps", "max_speed_ignoring_yaw_mps", "max_mean_vx_mps"):
            merged[key] = min(r.get(key, 0.0) for r in results)
        merged["per_seed_max_sustained_mps"] = [r["max_sustained_speed_mps"] for r in results]
        return merged

    def isaac_eval_once(self, ckpt, round_dir, top, seed):
        speeds = [1.0]
        while speeds[-1] < top - 1e-9:
            speeds.append(round(speeds[-1] + 0.5, 2))
        out = round_dir / f"isaac_eval_seed{seed}.json"
        code = self.run(
            [
                self.args.isaac_python,
                "scripts/isaac_sprint_eval.py",
                "--policy",
                ckpt,
                "--usd",
                self.args.usd,
                "--speeds",
                ",".join(f"{s:g}" for s in speeds),
                "--envs_per_speed",
                32,
                "--seed",
                seed,
                "--output",
                out,
            ],
            round_dir / f"isaac_eval_seed{seed}.log",
            timeout=1800,
        )
        return json.loads(out.read_text()) if code == 0 and out.exists() else None

    def export(self, ckpt, round_dir):
        bundle = round_dir / "policy"
        code = self.run(
            [
                self.args.isaac_python,
                "scripts/export_rslrl_onnx.py",
                "--checkpoint",
                ckpt,
                "--output",
                bundle,
            ],
            round_dir / "export.log",
            timeout=600,
        )
        return bundle if code == 0 and (bundle / "exported/policy.onnx").exists() else None

    def start_mujoco(
        self,
        bundle,
        round_dir,
        top,
        round_no,
        seeds=2,
        hold=20.0,
        seed0=96000,
        name="mujoco_eval",
        kind="round",
    ):
        speeds, s = [], 1.0
        while s <= top + 1e-9:
            speeds.append(s)
            s = round(s + 0.5, 2)
        out = round_dir / f"{name}.json"
        argv = [
            "unshare",
            "-rn",
            self.args.eval_python,
            "scripts/mujoco_sprint_eval.py",
            "--policy-dir",
            bundle,
            "--speeds",
            ",".join(f"{v:g}" for v in speeds),
            "--seeds",
            seeds,
            "--seed0",
            seed0,
            "--hold",
            hold,
            "--workers",
            self.args.mujoco_workers,
            "--output",
            out,
        ]
        log = open(round_dir / f"{name}.log", "w")
        proc = subprocess.Popen(
            list(map(str, argv)),
            cwd=ROOT,
            env=self.env(),
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self.mujoco_jobs.append(
            {"proc": proc, "out": out, "round": round_no, "bundle": str(bundle), "log": log, "kind": kind}
        )

    def collect_mujoco(self, wait=False):
        for job in list(self.mujoco_jobs):
            if wait:
                try:
                    job["proc"].wait(timeout=7200)
                except subprocess.TimeoutExpired:
                    os.killpg(job["proc"].pid, signal.SIGKILL)
            if job["proc"].poll() is None:
                continue
            job["log"].close()
            self.mujoco_jobs.remove(job)
            if not job["out"].exists():
                self.event("mujoco_eval_failed", round=job["round"])
                continue
            result = json.loads(job["out"].read_text())
            speed = result["max_speed_all_seeds_passed_mps"]
            if job["kind"] == "validation_mujoco":
                best = self.state["best_mujoco"]
                if best and best["round"] == job["round"]:
                    best["validation_10seeds_60s_max_passed_mps"] = speed
                self.event("mujoco_best_validation_done", round=job["round"], max_passed=speed)
                continue
            if job["kind"] == "validation":
                summary = {
                    "round": job["round"],
                    "max_speed_all_10_seeds_passed_mps": speed,
                    "per_speed": [
                        {k: p[k] for k in ("speed_cmd_mps", "passed", "survived", "count", "flight_fraction")}
                        for p in result["per_speed"]
                    ],
                }
                self.state.setdefault("validations", []).append(summary)
                best = self.state["best_isaac"]
                if best and best["round"] == job["round"]:
                    best["mujoco_validation"] = summary
                self.event("mujoco_validation_done", round=job["round"], max_passed=speed)
                continue
            if job["kind"] == "final":
                continue
            entry = next((r for r in self.state["rounds"] if r["round"] == job["round"]), None)
            if entry is not None:
                entry["mujoco_max_passed_mps"] = speed
            best = self.state["best_mujoco"]
            if best is None or speed > best["speed"]:
                self.state["best_mujoco"] = {"speed": speed, "round": job["round"], "bundle": job["bundle"]}
                self.event("new_best_mujoco", speed=speed, round=job["round"])
                keep = self.out / "best_mujoco"
                shutil.rmtree(keep, ignore_errors=True)
                shutil.copytree(job["bundle"], keep / "policy")
                if speed > 0:
                    self.start_mujoco(
                        keep / "policy",
                        self.out / f"round_{job['round']:03d}",
                        speed,
                        job["round"],
                        seeds=10,
                        hold=60.0,
                        seed0=99000,
                        name="mujoco_validation_best_mujoco",
                        kind="validation_mujoco",
                    )
        self.save()

    # ------------------------------------------------------------------ loop
    def should_finish(self):
        """Deadline (None = run until stopped), control.json and free-disk checks, read before every round."""
        control = self.out / "control.json"
        if control.exists():
            try:
                c = json.loads(control.read_text())
            except ValueError:
                c = {}
            if c.get("stop"):
                self.event("stopped_by_control_file")
                return True
            if "hours_from_now" in c and not c.get("_applied"):
                self.state["deadline"] = time.time() + float(c["hours_from_now"]) * 3600
                c["_applied"] = True
                control.write_text(json.dumps(c))
                self.event("deadline_set_by_control_file", hours=c["hours_from_now"])
        free_gb = shutil.disk_usage(self.out).free / 2**30
        if free_gb < self.args.min_free_gb:
            self.event("low_disk_stop", free_gb=round(free_gb, 1))
            return True
        deadline = self.state["deadline"]
        return deadline is not None and time.time() >= deadline

    def loop(self):
        threading.Thread(target=self.guard, daemon=True).start()
        while not self.stop.is_set() and not self.should_finish():
            self.state["round"] += 1
            n = self.state["round"]
            round_dir = self.out / f"round_{n:03d}"
            round_dir.mkdir(exist_ok=True)
            self.log(
                f"round {n}: level={self.state['level']:.2f} lr={self.state['lr']:.2e} ckpt={self.state['latest_ckpt']}"
            )
            started = time.time()
            train = self.train(round_dir, n)
            entry = {
                "round": n,
                "train": train,
                "level_in": self.state["level"],
                "lr": self.state["lr"],
                "knobs": list(self.state["knobs_applied"]),
                "train_minutes": (time.time() - started) / 60,
            }
            self.collect_mujoco()
            if self.stop.is_set():
                break
            ok = train["returncode"] == 0 and train["checkpoint"] and not train["nan"]
            if not ok and train["iterations_logged"] == 0:
                self.state["rounds"].append(entry)
                if not self.startup_failure(n, round_dir):
                    self.save()
                    break
                self.save()
                continue
            self.state["startup_failures"] = 0
            if not ok:
                self.state["consecutive_failures"] += 1
                self.event(
                    "train_failed",
                    round=n,
                    returncode=train["returncode"],
                    nan=train["nan"],
                    partial_ckpt=train["checkpoint"],
                )
                if train["checkpoint"] and not train["nan"]:
                    self.state["latest_ckpt"] = train["checkpoint"]
                if self.state["consecutive_failures"] >= 2 or train["nan"]:
                    self.rollback("repeated_failure_or_nan")
                self.state["rounds"].append(entry)
                self.save()
                continue
            self.state["consecutive_failures"] = 0
            if train["level"] is not None:
                self.state["level"] = max(self.state["level"], train["level"])
            top = min(self.args.speed_limit, max(2.0, self.state["level"] + 1.0))
            ev = self.isaac_eval(train["checkpoint"], round_dir, top)
            entry["isaac_eval"] = (
                None
                if ev is None
                else {
                    "max_sustained_speed_mps": ev["max_sustained_speed_mps"],
                    "max_speed_ignoring_yaw_mps": ev.get("max_speed_ignoring_yaw_mps"),
                    "per_seed_max_sustained_mps": ev.get("per_seed_max_sustained_mps"),
                    "max_mean_vx_mps": ev["max_mean_vx_mps"],
                    "groups": ev["groups"],
                }
            )
            bundle = self.export(train["checkpoint"], round_dir)
            if bundle and ev:
                top_mj = max(ev["max_sustained_speed_mps"], ev.get("max_speed_ignoring_yaw_mps", 0.0))
                self.start_mujoco(bundle, round_dir, max(1.5, top_mj + 0.5), n)
            if ev is None:
                self.event("isaac_eval_failed", round=n)
                self.state["latest_ckpt"] = train["checkpoint"]
            else:
                self.decide(n, train["checkpoint"], ev, bundle)
                # Curriculum follows running capability (speed criteria only); best selection uses all criteria.
                self.push_curriculum(
                    entry["level_in"], train["level"], ev.get("max_speed_ignoring_yaw_mps", 0.0)
                )
            entry["level_out"] = self.state["level"]
            self.state["rounds"].append(entry)
            self.save()
        self.finish()

    def keep_capability(self, n, ckpt, ev, bundle):
        """Also keep the fastest policy by speed criteria alone (yaw not yet fixed), within the actuator spec."""
        cap = ev.get("max_speed_ignoring_yaw_mps", 0.0)
        group = next((g for g in ev["groups"] if g["speed_cmd_mps"] == cap), {})
        best = self.state.get("best_capability")
        if best is None or cap > best["speed"]:
            keep = self.out / "best_capability"
            keep.mkdir(exist_ok=True)
            shutil.copy2(ckpt, keep / "model.pt")
            if bundle:
                shutil.rmtree(keep / "policy", ignore_errors=True)
                shutil.copytree(bundle, keep / "policy")
            self.state["best_capability"] = {
                "speed": cap,
                "round": n,
                "yaw_rate_rms": group.get("yaw_rate_rms"),
                "flight_fraction": group.get("flight_fraction"),
                "ckpt": str(keep / "model.pt"),
            }
            self.event("new_best_capability", round=n, speed=cap, yaw=group.get("yaw_rate_rms"))

    def decide(self, n, ckpt, ev, bundle):
        prev_cap = (self.state.get("best_capability") or {}).get("speed", 0.0)
        self.keep_capability(n, ckpt, ev, bundle)
        cap = ev.get("max_speed_ignoring_yaw_mps", 0.0)
        if prev_cap >= 3.0 and cap <= prev_cap - self.args.capability_drop:
            self.state["capability_drop_rounds"] = self.state.get("capability_drop_rounds", 0) + 1
        else:
            self.state["capability_drop_rounds"] = 0
        if self.state["capability_drop_rounds"] >= 2:
            self.state["capability_drop_rounds"] = 0
            self.state["capability_rollbacks"] = self.state.get("capability_rollbacks", 0) + 1
            self.rollback(f"capability collapse {cap} <= best {prev_cap} - {self.args.capability_drop}")
            if self.state["capability_rollbacks"] % 2 == 0:
                self.apply_knob()  # the same recipe collapsed twice from the anchor: change something
            return
        speed = ev["max_sustained_speed_mps"]
        group = next((g for g in ev["groups"] if g["speed_cmd_mps"] == speed), {})
        vx = group.get("mean_vx_mps") or 0.0
        best = self.state["best_isaac"]
        improved = (
            best is None or speed > best["speed"] or (speed == best["speed"] and vx > best["vx"] + 0.02)
        )
        if improved:
            keep = self.out / "best_isaac"
            keep.mkdir(exist_ok=True)
            shutil.copy2(ckpt, keep / "model.pt")
            if bundle:
                shutil.rmtree(keep / "policy", ignore_errors=True)
                shutil.copytree(bundle, keep / "policy")
            (keep / "isaac_eval.json").write_text(json.dumps(ev, indent=2))
            self.state["best_isaac"] = {
                "speed": speed,
                "vx": vx,
                "round": n,
                "ckpt": str(keep / "model.pt"),
                "source_ckpt": ckpt,
                "flight_fraction": group.get("flight_fraction"),
            }
            self.state["stall"] = 0
            self.state["rollback_streak"] = 0
            self.event("new_best_isaac", round=n, speed=speed, vx=vx, flight=group.get("flight_fraction"))
            if (keep / "policy").exists():
                # Held-out MuJoCo check of every new best: 10 new seeds, 60 s hold (background, CPU).
                self.start_mujoco(
                    keep / "policy",
                    self.out / f"round_{n:03d}",
                    max(1.5, speed),
                    n,
                    seeds=10,
                    hold=60.0,
                    seed0=99000,
                    name="mujoco_validation_10seeds",
                    kind="validation",
                )
            self.state["latest_ckpt"] = ckpt
            return
        self.state["stall"] += 1
        cooldown = self.state.get("rollback_cooldown", 0)
        if best["speed"] >= 1.5 and speed < best["speed"] - 1.0:
            if cooldown > 0:
                self.state["rollback_cooldown"] = cooldown - 1
            elif self.state.get("rollback_streak", 0) >= 2:
                # Repeated rollbacks to the same best (the old run looped ~160 times): keep training forward.
                self.state["rollback_streak"] = 0
                self.state["rollback_cooldown"] = 4
                self.event("rollback_suppressed", round=n, speed=speed, best=best["speed"], cooldown_rounds=4)
            else:
                self.state["rollback_streak"] = self.state.get("rollback_streak", 0) + 1
                self.rollback(f"regression {speed} < best {best['speed']} - 1.0")
                return
        else:
            self.state["rollback_streak"] = 0
            self.state["rollback_cooldown"] = max(0, cooldown - 1)
        self.state["latest_ckpt"] = ckpt
        if self.state["stall"] >= self.args.stall_rounds:
            self.apply_knob()

    def push_curriculum(self, level_in, level_trained, max_speed):
        """The Unitree curriculum only widens the command range when tracking reward > 80 % of its maximum,
        which starting every episode from rest makes marginal. If it did not move but the policy already holds
        speeds near the top of the trained range, widen the range by 0.5 m/s ourselves."""
        moved = level_trained is not None and level_trained > level_in + 1e-6
        if (
            not moved
            and max_speed >= self.state["level"] - 0.25
            and self.state["level"] < self.args.speed_limit
        ):
            self.state["level"] = min(self.args.speed_limit, self.state["level"] + 0.5)
            self.event("curriculum_push", level=self.state["level"], max_sustained=max_speed)

    def anchor(self):
        """(checkpoint, speed) to roll back to: the MuJoCo-validated best if known, else the Isaac best."""
        mj = self.state.get("best_mujoco")
        if mj and mj.get("speed", 0) > 0:
            entry = next((r for r in self.state["rounds"] if r["round"] == mj["round"]), None)
            ckpt = entry and entry["train"].get("checkpoint")
            if ckpt and Path(ckpt).exists():
                return ckpt, mj.get("validation_10seeds_60s_max_passed_mps") or mj["speed"]
        best = self.state["best_isaac"]
        if best:
            return best["ckpt"], best["speed"]
        return self.state["latest_ckpt"], None  # never roll back to a random policy

    def rollback(self, reason):
        target, speed = self.anchor()
        self.state["latest_ckpt"] = target
        self.state["lr"] = max(self.state["lr"] * 0.5, 1.0e-5)
        if speed is not None:
            self.state["level"] = min(self.state["level"], speed + 0.5)
        self.state["consecutive_failures"] = 0
        self.event("rollback", reason=reason, to=target, lr=self.state["lr"], level=self.state["level"])

    def startup_failure(self, n, round_dir):
        """Run died before training started. Returns False when the loop must halt."""
        log = round_dir / "train" / "train.log"
        text = log.read_text(errors="ignore") if log.exists() else ""
        bad_keys = re.findall(r"Could not override '([^']+)'", text)
        self.state["startup_failures"] = self.state.get("startup_failures", 0) + 1
        self.event("startup_failure", round=n, count=self.state["startup_failures"], rejected_keys=bad_keys)
        if bad_keys and self.revert_last_knob(bad_keys):
            return True
        if self.state["startup_failures"] >= self.args.max_startup_failures:
            self.event("halted", reason="repeated_startup_failures", count=self.state["startup_failures"])
            return False
        wait = min(1800, 60 * 2 ** (self.state["startup_failures"] - 1))
        self.log(f"startup failure; retrying in {wait}s")
        self.stop.wait(wait)
        return True

    def revert_last_knob(self, bad_keys):
        """Undo the most recent knob if it introduced a rejected override (restores previous values)."""
        last = self.state.get("last_knob")
        if not last or not set(bad_keys) & set(last["previous"]):
            for key in bad_keys:  # an override not from a knob (e.g. --override): drop it, it can never work
                self.state["overrides"].pop(key, None)
            self.event("override_dropped", keys=bad_keys)
            return bool(bad_keys)
        for key, old in last["previous"].items():
            if old is None:
                self.state["overrides"].pop(key, None)
            else:
                self.state["overrides"][key] = old
        if last.get("lr_before") is not None:
            self.state["lr"] = last["lr_before"]
        self.state["last_knob"] = None
        self.event("knob_reverted", knob=last["name"], rejected_keys=bad_keys)
        return True

    def apply_knob(self):
        done = {k for k in self.state["knobs_applied"]}
        for knob in KNOBS:
            if knob["name"] in done:
                continue
            overrides = knob.get("overrides", {})
            self.state["last_knob"] = {
                "name": knob["name"],
                "previous": {k: self.state["overrides"].get(k) for k in overrides},
                "lr_before": self.state["lr"] if "lr" in knob else None,
            }
            self.state["overrides"].update(dict(overrides))
            if "lr" in knob:
                self.state["lr"] = knob["lr"]
            self.state["knobs_applied"].append(knob["name"])
            self.state["stall"] = 0
            self.event("knob_applied", knob=knob["name"])
            return
        if self.state["lr"] < 1.0e-4:
            self.state["lr"] = 1.0e-4  # after repeated rollbacks, do not stay frozen at the LR floor
        self.event("no_knobs_left", lr=self.state["lr"])
        self.state["stall"] = 0

    def finish(self):
        self.log("deadline reached or stopped; final evaluation")
        self.collect_mujoco(wait=True)
        if not self.stop.is_set():
            final = self.out / "final"
            final.mkdir(exist_ok=True)
            best = self.state["best_isaac"]
            if best:
                full = self.isaac_eval(best["ckpt"], final, self.args.speed_limit)
                if full:
                    self.state["final_isaac_ladder"] = {
                        "max_sustained_speed_mps": full["max_sustained_speed_mps"],
                        "max_mean_vx_mps": full["max_mean_vx_mps"],
                    }
                bundle = self.out / "best_isaac" / "policy"
                if bundle.exists():
                    # Held-out validation: 10 new seeds, 60 s hold, MuJoCo.
                    self.start_mujoco(
                        bundle,
                        final,
                        max(1.5, best["speed"]),
                        best["round"],
                        seeds=10,
                        hold=60.0,
                        seed0=99000,
                        name="mujoco_validation_10seeds",
                        kind="final",
                    )
                    self.collect_mujoco(wait=True)
                    path = final / "mujoco_validation_10seeds.json"
                    if path.exists():
                        val = json.loads(path.read_text())
                        self.state["final_mujoco_validation"] = {
                            "max_speed_all_10_seeds_passed_mps": val["max_speed_all_seeds_passed_mps"],
                            "per_speed": [
                                {k: p[k] for k in ("speed_cmd_mps", "passed", "survived", "count")}
                                for p in val["per_speed"]
                            ],
                        }
        self.state["finished"] = datetime.now().isoformat(timespec="seconds")
        self.save()
        self.log("autopilot finished")

    # ------------------------------------------------------------------ report
    def write_report(self):
        s = self.state
        lines = [
            "# Autopilot sprint — báo cáo tự động",
            "",
            f"Bắt đầu: {s['started']} · Hạn chót: "
            f"{'không, chạy đến khi dừng' if s['deadline'] is None else datetime.fromtimestamp(s['deadline']).isoformat(timespec='seconds')}"
            f" · Kết thúc: {s.get('finished', 'đang chạy')}",
            "",
            "Số đo Isaac là trên simulator dùng để train (PhysX). MuJoCo là simulator độc lập để kiểm chứng.",
            "Cả hai dùng chung bảng thông số động cơ (g1lab.actuator_spec): đường cong mô-men-tốc độ Unitree, giới hạn",
            "chặt nhất giữa các nguồn, không khớp nào vượt tốc độ định mức. Train có domain randomization; đánh giá ở",
            "điều kiện danh định, giữ hướng (gain 1, ±0,2 rad/s).",
            "Isaac đạt = sống ≥ 90%, vx ≥ 85% lệnh, RMS quay < 0,25 rad/s, lệch hướng < 30° (2 seed × 32 robot, lấy kém hơn); MuJoCo: mọi seed qua",
            "tiêu chí cũ (không ngã, vx ≥ 85%, RMSE, quay, hướng, dừng). Giới hạn vật lý không bị thay đổi.",
            "",
            f"**Tốt nhất (Isaac):** {s['best_isaac']}",
            "",
            f"**Tốt nhất (MuJoCo, 2 seed, giữ 20 s):** {s['best_mujoco']}",
            "",
            f"**Năng lực tốc độ cao nhất (Isaac, chỉ xét tốc độ, đúng thông số):** {s.get('best_capability')}",
            "",
        ]
        if "final_isaac_ladder" in s:
            lines += [
                f"**Đánh giá cuối Isaac (thang 1–{self.args.speed_limit} m/s):** {s['final_isaac_ladder']}",
                "",
            ]
        if "final_mujoco_validation" in s:
            lines += [
                f"**Kiểm chứng cuối MuJoCo (10 seed mới, giữ 60 s):** {s['final_mujoco_validation']}",
                "",
            ]
        for v in s.get("validations", [])[-3:]:
            lines += [
                f"**Kiểm chứng MuJoCo bản tốt nhất vòng {v['round']} (10 seed mới, giữ 60 s):** tốc độ cao nhất mọi"
                f" seed đạt = {v['max_speed_all_10_seeds_passed_mps']} m/s; {v['per_speed']}",
                "",
            ]
        lines += [
            "| Vòng | Phút | Level curriculum | LR | Isaac đạt đủ tiêu chí (m/s) | Isaac chỉ xét tốc độ | RMS quay ở mức đó |"
            " Bay ở mức đó | MuJoCo đạt (spec) | Ghi chú |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for r in s["rounds"]:
            ev = r.get("isaac_eval") or {}
            mx = ev.get("max_sustained_speed_mps")
            cap = ev.get("max_speed_ignoring_yaw_mps")
            flight = yaw = None
            for g in ev.get("groups", []):
                if g["speed_cmd_mps"] == (cap or mx):
                    flight, yaw = g.get("flight_fraction"), g.get("yaw_rate_rms")
            note = "" if r["train"]["returncode"] == 0 else f"train rc={r['train']['returncode']}"
            lines.append(
                f"| {r['round']} | {r['train_minutes']:.0f} | {r['level_in']:.1f}→{r.get('level_out', r['level_in']):.1f} | "
                f"{r['lr']:.1e} | {mx} | {cap} | {yaw if yaw is None else round(yaw, 2)} | "
                f"{flight if flight is None else round(flight, 3)} | {r.get('mujoco_max_passed_mps')} | {note} |"
            )
        lines += ["", "## Sự kiện", ""] + [
            f"- {e['time']} {e['kind']} {json.dumps({k: v for k, v in e.items() if k not in ('time', 'kind')})}"
            for e in s["events"]
        ]
        (self.out / "REPORT.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--usd", required=True)
    parser.add_argument("--isaac-python", default=str(WORKSPACE / "envs/isaac51/bin/python"))
    parser.add_argument("--eval-python", default=str(ROOT / ".venv-eval/bin/python"))
    parser.add_argument("--hours", type=float, default=10.0)
    parser.add_argument("--envs", type=int, default=4096)
    parser.add_argument("--round-iterations", type=int, default=600)
    parser.add_argument("--critic-warmup", type=int, default=20)
    parser.add_argument("--speed-limit", type=float, default=11.0)
    parser.add_argument("--start-level", type=float, default=1.5)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--stall-rounds", type=int, default=3)
    parser.add_argument("--mujoco-workers", type=int, default=4)
    parser.add_argument("--min-free-gb", type=float, default=50.0)
    parser.add_argument("--max-startup-failures", type=int, default=5)
    parser.add_argument("--capability-drop", type=float, default=1.5)
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Hydra override applied to every round, e.g. agent.algorithm.entropy_coef=0.001 (repeatable)",
    )
    args = parser.parse_args()
    pilot = Autopilot(args)
    signal.signal(signal.SIGTERM, lambda *_: (pilot.stop.set(), pilot.kill_child()))
    pilot.loop()


if __name__ == "__main__":
    main()
