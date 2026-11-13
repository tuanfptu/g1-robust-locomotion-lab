# Autopilot sprint (chạy qua đêm, không cần mạng)

Lượt hiện tại: `artifacts/autopilot-continuous-20261113/` (bắt đầu 13/11/2026 ~22:23), **không hạn chót**:
chạy cho đến khi có lệnh dừng. (`autopilot-night-20261113/` là lượt 10,5 h bị dừng ở vòng 1 để chuyển sang chạy
không giới hạn; `autopilot-test-*` là lượt thử nhỏ.) Mã: [scripts/autopilot_sprint.py](../scripts/autopilot_sprint.py).

## Sáng mai xem gì

```bash
cd ~/TUAN__HUMAN_NOID_G1_RUNNER/g1-5090-handoff/g1-robust-locomotion-lab
cat artifacts/autopilot-continuous-20261113/REPORT.md        # bảng từng vòng + tốt nhất + kiểm chứng cuối
tail -30 artifacts/autopilot-continuous-20261113/autopilot.log
ps -eo pid,etime,args | grep "[a]utopilot_sprint"       # còn chạy không
```

- `best_isaac/model.pt`, `best_isaac/policy/` (ONNX + deploy.yaml), `best_isaac/isaac_eval.json`: checkpoint có
  tốc độ giữ được cao nhất trong Isaac.
- `round_NNN/`: log train, đánh giá Isaac (`isaac_eval.json`), bundle ONNX, đánh giá MuJoCo (`mujoco_eval.json`).
- `final/`: thang 1–11 m/s trong Isaac và **kiểm chứng MuJoCo 10 seed mới, giữ 60 s** cho checkpoint tốt nhất.
- `state.json`: toàn bộ trạng thái, có thể chạy tiếp bằng đúng lệnh khởi động với cùng `--output`.

## Dừng

Dừng êm sau vòng đang chạy (rồi tự đánh giá cuối: thang Isaac 1–11 m/s + MuJoCo 10 seed, giữ 60 s):

```bash
echo '{"stop": true}' > artifacts/autopilot-continuous-20261113/control.json
```

Đặt hạn chót tính từ bây giờ: `echo '{"hours_from_now": 2}' > .../control.json`.
Dừng ngay (bỏ dở vòng hiện tại): `kill -TERM $(pgrep -f "scripts/autopilot_sprint.py --output")`.
Tự dừng khi ổ đĩa còn < 50 GB. Mỗi bản tốt nhất mới tự được kiểm chứng MuJoCo 10 seed mới / 60 s.

## Vòng lặp làm gì

1. Train PPO 600 iteration, 4096 env, task `Unitree-G1-29dof-Sprint-G1Lab`, tiếp từ checkpoint hiện tại
   (vòng 1: actor ONNX ghép + warm-up critic 20 iteration). Mức curriculum được mang sang vòng sau.
2. Đánh giá Isaac theo thang 1 → (mức + 1) m/s, bước 0,5, 16 robot/tốc độ, xuất phát từ đứng yên, giữ 10 s.
   Đạt = sống ≥ 90% và vx ≥ 85% lệnh.
3. Export ONNX; đánh giá MuJoCo (2 seed, giữ 20 s) chạy nền trên CPU.
4. Quyết định: tốt hơn → lưu `best_isaac`; tụt > 1 m/s, NaN hoặc lỗi 2 lần → quay về best, giảm LR một nửa;
   3 vòng không tiến → bật nút chỉnh kế tiếp (trọng số bám tốc độ 1,5; độ rộng 0,75; gia tốc lệnh 3 m/s²;
   tăng thưởng pha bay; LR 3e-4); curriculum Unitree không tăng mà policy đã giữ gần đỉnh dải → tự nâng 0,5 m/s.

## An toàn

- Chỉ GPU1: CUDA ghim theo UUID; một luồng canh GPU0 mỗi 2 s, thấy tiến trình của dự án là dừng toàn bộ.
- Mọi tiến trình Isaac/MuJoCo chạy trong `unshare -rn` (không mạng); mặt đất sinh tại chỗ, không tải asset.
  Đã kiểm thử train + đánh giá khi không có mạng.
- Không đổi giới hạn mô-men, giới hạn vận tốc khớp, gain PD hay mô hình robot.
- **Không tắt máy RTX 5090**: tắt máy là dừng training.

## Kỳ vọng trung thực

11 m/s là trần curriculum, không phải dự báo. SPRINT công bố ~6 m/s trên G1 với prior chuyển động và
6,5 h trên RTX 4090. Mô hình actuator ở đây chỉ có giới hạn mô-men/vận tốc phẳng, không có đường cong mô-men
theo tốc độ, nên tốc độ đạt được trong sim có thể lạc quan so với robot thật. Kết quả sáng mai cần đọc theo
cả Isaac lẫn MuJoCo, kèm tỷ lệ pha bay.
