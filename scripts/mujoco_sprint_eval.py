"""MuJoCo speed-ladder benchmark of an exported policy bundle with sprint gait metrics (CPU, offline).

Same pass criteria as the controller benchmark (no fall, vx >= 85% command, planar RMSE, yaw RMS, heading,
stop), heading hold on. Ramp time scales with speed (speed / accel, at least 3 s). Run with .venv-eval.
"""

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def run_one(job):
    from g1lab.lab_evaluation import trial
    from g1lab.lab_policy import LabReference

    policy_dir, speed, seed, hold, accel = job
    env = LabReference(policy_dir=policy_dir)
    ramp = max(3.0, speed / accel)
    result, _ = trial(
        env, speed, seed, hold=hold, heading_hold=True, heading_gain=1.0, ramp=ramp, diagnostics=True
    )
    result["ramp_s"] = ramp
    return result


def main(args):
    speeds = [float(s) for s in args.speeds.split(",")]
    seeds = [args.seed0 + i for i in range(args.seeds)]
    jobs = [(args.policy_dir, v, s, args.hold, args.accel) for v in speeds for s in seeds]
    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(run_one, jobs))
    per_speed = []
    for v in speeds:
        rows = [r for r in results if r["command_mps"] == v]
        per_speed.append(
            {
                "speed_cmd_mps": v,
                "passed": sum(r["passed"] for r in rows),
                "survived": sum(r["survived"] for r in rows),
                "count": len(rows),
                "mean_vx_mps": [r["steady_vx_mps"] for r in rows],
                "flight_fraction": [r.get("gait_flight_fraction") for r in rows],
                "median_stance_s": [r.get("gait_median_stance_time_s") for r in rows],
                "forefoot_strike_fraction": [r.get("gait_forefoot_strike_fraction") for r in rows],
            }
        )
    all_pass = [p["speed_cmd_mps"] for p in per_speed if p["passed"] == p["count"]]
    out = {
        "policy_dir": str(Path(args.policy_dir).resolve()),
        "physics": "MuJoCo (independent of the PhysX training simulator)",
        "seeds": seeds,
        "hold_s": args.hold,
        "max_speed_all_seeds_passed_mps": max(all_pass, default=0.0),
        "wall_time_s": time.time() - started,
        "per_speed": per_speed,
        "episodes": results,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps({k: out[k] for k in ("max_speed_all_seeds_passed_mps", "wall_time_s")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-dir", required=True)
    parser.add_argument("--speeds", required=True)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--seed0", type=int, default=96000)
    parser.add_argument("--hold", type=float, default=20.0)
    parser.add_argument("--accel", type=float, default=2.0)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--output", required=True)
    main(parser.parse_args())
