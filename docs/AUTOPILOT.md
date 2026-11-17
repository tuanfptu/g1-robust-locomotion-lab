# Autopilot sprint (train qua đêm, không cần mạng)

[scripts/autopilot_sprint.py](../scripts/autopilot_sprint.py) — chỉ dùng thư viện chuẩn Python, chạy bằng
`.venv-eval`, gọi Isaac (train, thang tốc độ) và MuJoCo (chấm) làm tiến trình con. Lịch sử các lượt, sự cố và
bài học: [G1_SPRINT_REPORT.md](G1_SPRINT_REPORT.md) mục 7.

## Khởi động

```bash
USD=.../G1/29dof/usd/g1_29dof_rev_1_0/g1_29dof_rev_1_0.usd

# Từ đầu
setsid nohup .venv-eval/bin/python scripts/autopilot_sprint.py --output artifacts/autopilot-<tên> \
  --init-checkpoint scratch --usd "$USD" --hours 0 --envs 4096 --round-iterations 600 \
  --lr 1e-3 --start-level 1.0 --speed-limit 11.0 > artifacts/autopilot-<tên>.out 2>&1 &

# Train tiếp từ policy đã kiểm chứng (policy này là mốc rollback cho tới khi bị vượt)
… --init-checkpoint models/g1-sprint-v2-r11/model.pt --init-validated-speed 5.5 --start-level 5.5
```

`--hours 0` = chạy tới khi có lệnh dừng. Chạy lại đúng lệnh với cùng `--output` sẽ tiếp tục từ `state.json`.
`--override KEY=VALUE` (lặp được) thêm Hydra override cho mọi vòng.

## Theo dõi (offline)

```bash
python3 scripts/autopilot_dashboard.py --run artifacts/autopilot-<tên> --watch 60 &   # ghi dashboard.html mỗi phút
python3 scripts/autopilot_dashboard.py --run artifacts/autopilot-<tên> --terminal --watch 60
cat artifacts/autopilot-<tên>/REPORT.md
```

Trong thư mục lượt chạy: `round_NNN/` (log train, đánh giá Isaac, bundle ONNX, đánh giá MuJoCo),
`best_isaac/`, `best_capability/` (nhanh nhất chỉ xét tốc độ), `best_mujoco/` (kiểm chứng 10 seed × 60 s),
`final/`, `state.json`, `autopilot.log`.

## Dừng

```bash
echo '{"stop": true}' > artifacts/autopilot-<tên>/control.json       # dừng êm sau vòng hiện tại
echo '{"hours_from_now": 2}' > artifacts/autopilot-<tên>/control.json  # đặt hạn chót
```

Tự dừng khi ổ đĩa còn < 50 GB, hoặc sau 5 lần train chết ngay khi khởi động.

## Mỗi vòng

1. Train PPO `--round-iterations` iteration trên `Unitree-G1-29dof-Sprint-G1Lab`, tiếp từ checkpoint hiện tại,
   mang theo mức curriculum (vòng đầu từ checkpoint: warm-up critic 20 iteration).
2. Thang tốc độ Isaac 1 → (mức + 1) m/s, 2 seed × 32 robot, lấy kết quả kém hơn; xuất phát từ đứng yên, giữ
   10 s. "Đủ tiêu chí": sống ≥ 90 %, vx ≥ 85 % lệnh, quay < 0,25 rad/s, lệch hướng < 30°. "Chỉ xét tốc độ": bỏ
   hai điều kiện quay.
3. Export ONNX; MuJoCo nhanh chạy nền (CPU). Bản tốt nhất mới → MuJoCo 10 seed mới × 60 s.

## Quy tắc tự xử lý

| Tình huống | Xử lý |
|---|---|
| NaN, hoặc std hành động PPO âm/NaN | rollback về mốc, LR × 0,5; không train tiếp từ checkpoint đó |
| Train lỗi 2 lần liên tiếp | rollback về mốc, LR × 0,5 |
| "Chỉ xét tốc độ" về 0 m/s trong khi mốc chạy được | rollback ngay |
| "Chỉ xét tốc độ" thấp hơn kỷ lục ≥ 1,5 m/s hai vòng liền | rollback; lần thứ hai cùng kiểu → thêm nút chỉnh |
| "Đủ tiêu chí" tụt > 1 m/s so với bản tốt nhất | rollback (tối đa 2 lần liên tiếp, sau đó chờ 4 vòng) |
| Chững 3 vòng | nút chỉnh kế tiếp: bám tốc độ 2,5 → gia tốc lệnh 3 m/s² → tăng thưởng pha bay/đẩy → LR 3e-4 |
| Train chết trước iteration đầu, Hydra từ chối khoá | gỡ nút chỉnh gây lỗi, train tiếp |
| Train chết trước iteration đầu, lý do khác | chờ 1, 2, 4 … phút; dừng sau 5 lần |
| Curriculum không tự tăng nhưng policy giữ được gần đỉnh dải | tự nâng 0,5 m/s |

**Mốc rollback** = policy nhanh nhất đã kiểm chứng MuJoCo (kể cả policy khởi tạo nếu có
`--init-validated-speed`); nếu chưa có thì bản tốt nhất Isaac còn chạy được; không bao giờ là policy ngẫu nhiên.
Không nút chỉnh nào được tăng thưởng giữ hướng/xoắn thân (nguyên nhân đợt sụp vòng 18–30 của lượt v2).

## An toàn

- Chỉ GPU dự án: CUDA ghim theo UUID; luồng canh GPU0 mỗi 2 s, thấy tiến trình của dự án là dừng toàn bộ.
- Mọi tiến trình con chạy trong `unshare -rn` (không mạng); mặt đất sinh tại chỗ.
- Dừng tiến trình con: SIGTERM, sau đó SIGKILL cả nhóm (Isaac có thể bỏ qua SIGTERM).
- Không đổi giới hạn mô-men, tốc độ khớp, gain PD hay mô hình robot.
- Tắt máy hoặc cho máy ngủ là dừng training; tắt màn hình và tắt mạng thì không sao.

## Kỳ vọng

11 m/s là trần curriculum, không phải dự báo. SPRINT (arXiv 2605.28549) công bố ~6 m/s trên G1.
