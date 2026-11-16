# Autopilot sprint (chạy qua đêm, không cần mạng)

Lượt hiện tại: `artifacts/autopilot-spec-20261116/` (bắt đầu 16/11/2026 09:34, **không hạn chót**), xuất phát từ
bản tốt nhất của lượt trước (`checkpoints/round7-best/model.pt`). Mã: [scripts/autopilot_sprint.py](../scripts/autopilot_sprint.py).

### Lượt trước `autopilot-continuous-20261113` (13–16/11, 170 vòng) — đã dừng

- 3,5 h đầu: Isaac từ 1,5 → 5,5 m/s, chạy có pha bay (~59–68% thời gian), chạm đất ~0,09 s, tiếp đất mũi ở
  3–4,5 m/s; MuJoCo giữ được tới 5,5 m/s (thực 5,14 m/s) nhưng **trượt mọi seed vì thân xoay** (RMS quay 0,6–0,9 rad/s).
- Sau đó ~2 ngày kẹt vòng lặp quay về bản tốt nhất; một tiến trình train treo 15 h vì Isaac bỏ qua SIGTERM.
- Động cơ train khi đó **mạnh hơn thông số** (88/139 Nm phẳng, không đường cong mô-men–tốc độ).
  Chấm lại theo thông số: MuJoCo sống tới 5,0 m/s nhưng khớp vượt tốc độ định mức 10%; Isaac chỉ còn
  4,5 m/s (xét tốc độ) và 1,0 m/s (đủ tiêu chí).

### Thay đổi cho lượt hiện tại

- **Thông số động cơ chung** ([actuator_spec.py](../src/g1lab/actuator_spec.py)) cho cả Isaac và MuJoCo: đường cong
  mô-men–tốc độ N7520 của Unitree + ma sát động cơ, giới hạn chặt nhất giữa các nguồn; MuJoCo trượt nếu khớp vượt
  tốc độ định mức.
- **Domain randomization:** ma sát 0,3–1,0; khối lượng khâu ±10%; torso +[−1, 3] kg; trọng tâm torso ±2–3 cm;
  gain PD (kp ±10%, kd ±20%); quán tính khớp ±20%; độ trễ lệnh 0–20 ms (động cơ chân); đẩy ngẫu nhiên 3–7 s;
  nhiễu quan sát.
- **Sửa lỗi quay:** train với lệnh giữ hướng; phạt sai lệch vận tốc quay khi chạy; tiêu chí Isaac thêm RMS quay
  < 0,25 rad/s và lệch hướng < 30°.
- **Autopilot:** đánh giá 2 seed × 32 robot, lấy kết quả kém hơn; ngừng quay về sau 2 lần liên tiếp (chờ 4 vòng);
  SIGKILL cả nhóm tiến trình nếu SIGTERM không có tác dụng; lưu thêm `best_capability/` (nhanh nhất chỉ xét tốc độ)
  và `best_mujoco/` (có kiểm chứng 10 seed, 60 s).

## Sáng mai xem gì

```bash
cd ~/TUAN__HUMAN_NOID_G1_RUNNER/g1-5090-handoff/g1-robust-locomotion-lab
cat artifacts/autopilot-spec-20261116/REPORT.md        # bảng từng vòng + tốt nhất + kiểm chứng cuối
tail -30 artifacts/autopilot-spec-20261116/autopilot.log
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
echo '{"stop": true}' > artifacts/autopilot-spec-20261116/control.json
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
