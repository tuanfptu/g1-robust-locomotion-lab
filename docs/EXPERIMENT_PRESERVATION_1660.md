# Six-task skill-preservation pilot

Follow-up to the turning regression in [the terrain-only pilot](EXPERIMENT_MIXED_1660.md). One shared policy is trained with PPO and behavioral replay from the previous student. The student itself originated from a Unitree teacher; this is not from-scratch learning.

Selected source: `runs/preservation-1660-v2/round-0/final`. Additional PPO transitions: 36,864. Measured CPU training throughput: 757–788 transitions/s.

## Paired held-out results

Seeds 50000–50002; 30 seconds per episode. All forward terrain commands are 0.3 m/s. Lateral command: 0.2 m/s; turn: 0.5 rad/s. Three episodes and one training trajectory are insufficient to claim statistical superiority or deployment reliability.

| Task | Previous survival | Selected survival | Previous velocity RMSE | Selected velocity RMSE | Previous yaw RMSE | Selected yaw RMSE |
|---|---:|---:|---:|---:|---:|---:|
| flat | 3/3 | 3/3 | 0.0863 | 0.0862 | 0.1237 | 0.1235 |
| slope | 3/3 | 3/3 | 0.1331 | 0.1327 | 0.1213 | 0.1218 |
| rough | 3/3 | 3/3 | 0.1048 | 0.1052 | 0.1292 | 0.1277 |
| stand | 3/3 | 3/3 | 0.0521 | 0.0521 | 0.0786 | 0.0781 |
| lateral | 3/3 | 3/3 | 0.1428 | 0.1531 | 0.1626 | 0.1696 |
| turn | 3/3 | 2/3 | 0.0606 | 0.2392 | 0.1736 | 0.4725 |

## Scope

The candidate completed only 2/3 turning episodes versus 3/3 for the prior policy. It is NOT promoted. The short 15-second validation gate missed a later instability. The script now defaults to two 30-second validation episodes for future campaigns; this stronger setting has not been retrospectively presented as part of the completed pilot.

The broader 22-scenario paired audit in the previous report applies to the previous policy and the first terrain-only candidate; it does not certify this follow-up. Steeper slopes, taller roughness, stairs, arbitrary command sequences and new push conditions remain outside this six-task test. The previous default checkpoint is retained.

Validation seed 7000 was used for model selection. Test seeds 50000–50002 were used only after selection. No further policy training was performed using these results. See [the local protocol](LOCAL_TRAINING.md) for replay collection, acceptance rules, and reproduction commands.
