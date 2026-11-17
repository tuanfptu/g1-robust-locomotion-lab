# g1-sprint-v2-r11 — policy sprint tốt nhất (17/11/2026)

Policy chạy có pha bay cho Unitree G1 29 khớp, train từ đầu trong Isaac Lab (task
`Unitree-G1-29dof-Sprint-G1Lab`, reward v2, động cơ theo thông số, domain randomization), vòng 11 của lượt
autopilot `autopilot-v2-scratch-20261116`, iteration 6589.

| File | Nội dung |
|---|---|
| `exported/policy.onnx` | actor (input `obs` 480, output `actions` 29), sha256 `104673f8…d231` |
| `params/deploy.yaml` | gain PD, thứ tự khớp, scale — đi cùng ONNX |
| `model.pt` | checkpoint RSL-RL đầy đủ (actor, critic, optimizer) để train tiếp, sha256 `288990f7…4dbc` |

Kiểm chứng MuJoCo (10 seed mới 99000–99009, đứng → tăng tốc 2 m/s² → giữ 60 s → giảm tốc → dừng, cùng mô hình
động cơ): đạt 10/10 ở mọi lệnh 1,0–5,5 m/s. Ở lệnh 5,5 m/s: thực 4,94 m/s, pha bay 46 %, chạm đất 0,13 s,
khớp ≤ 72 % tốc độ định mức. Tiếp đất bằng gót. Số liệu:
[best_round11_mujoco_10seeds_60s.json](../../docs/experiment-data/sprint-v2/best_round11_mujoco_10seeds_60s.json).

```bash
.venv-eval/bin/python scripts/mujoco_sprint_eval.py --policy-dir models/g1-sprint-v2-r11 --speeds 1,3,5.5 \
  --seeds 10 --seed0 99000 --hold 60 --output artifacts/check.json
```

Chỉ dùng cho mô phỏng. Chưa kiểm tra mô-men liên tục, nhiệt động cơ hay robot thật.
