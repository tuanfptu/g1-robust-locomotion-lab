# Task Sprint G1 (chạy bứt tốc có pha bay) — thiết kế và kiểm chứng ban đầu

Task Isaac: `Unitree-G1-29dof-Sprint-G1Lab` ([isaac_sprint.py](../src/g1lab/isaac_sprint.py)), kế thừa task gain v0.
Quan sát/action không đổi (480 → 29) để vẫn ghép được actor ONNX. **Chưa train dài; mọi trọng số là giả thuyết.**

## Mục tiêu kỹ thuật (yêu cầu của chủ dự án)

Pha bay là yếu tố quyết định: thời gian bay dài, thời gian chạm đất cực ngắn, lực đẩy ngang cực đại, tiếp đất
bằng mũi bàn chân, nâng cao đầu gối, gập cổ chân, đánh tay mạnh và dứt khoát, đổ người theo giai đoạn.

## Cơ chế

- **Cổng tốc độ** s(v) = clip((v_cmd − 1,6)/(2,2 − 1,6), 0, 1): dưới 1,6 m/s reward giống hệt task đi bộ mà
  ONNX được train; trên 2,2 m/s các term chỉ dành cho đi bộ tắt hẳn (vận tốc đứng, giữ thân phẳng, chiều cao
  thân, đồng hồ dáng đi, giữ tay), các term sprint bật hết.
- **Lệnh tốc độ** tăng dần về đích với gia tốc ≤ 2,0 m/s², giảm tốc ≤ 2,5 m/s²; đích lấy mẫu mỗi 4–8 s, 10%
  env lệnh dừng; mỗi tập bắt đầu từ đứng yên. Curriculum Unitree mở dải 0–1,5 → tối đa 3,5 m/s (+0,1 khi bám tốt).
- **Pha bước không dùng đồng hồ** (policy không quan sát pha): cảm biến bàn chân riêng ngưỡng 20 N, khử nhiễu —
  chạm đất ngay, nhấc chân sau 40 ms không tiếp xúc. Probe cho thấy không khử nhiễu thì một lần chạm bị tách
  thành nhiều sự kiện.
- **Lực đẩy ngang** = Σ mᵢaᵢ theo hướng chạy (Newton); cảm biến Isaac chỉ cho lực pháp tuyến.

| Term (chỉ khi chạy) | Trọng số | Định nghĩa |
|---|---|---|
| flight_phase | +2,0 | cả hai chân rời đất |
| double_support | −1,0 | cả hai chân chạm |
| stance_too_long | −10 | lúc nhấc chân: phần chạm đất vượt t_max = clip(0,36 − 0,06v; 0,15; 0,36) s |
| swing_air_time | +2,0 | lúc chạm: (thời gian bay của chân − 0,30 s), trần 0,55 s |
| push_off_force | +0,5 | lực ngang về trước khi có chân chạm, theo lần trọng lượng |
| forefoot_touchdown | +1,0 | lúc chạm: mũi thấp hơn gót ≥ 2 cm → +1, gót trước → −1 (tư thế ngay trước chạm) |
| knee_lift | +0,5 | chân đang vung: gối từ sâu 0,44 m (đứng) lên 0,22 m dưới hông |
| swing_dorsiflexion | +0,3 | chân vung cao > 8 cm: cổ chân gập mũi lên so với mặc định (0,3 rad = tối đa) |
| arm_leg_counter_swing | +0,5 | tanh(−Σ q̇_vai · q̇_hông cùng bên / 20): tay ngược pha chân cùng bên, mạnh |
| staged_lean | −20 | (|pitch thân − (0,05 + 0,04v + 0,06a)| − 0,05)², a = gia tốc lệnh |
| torso_roll | −5 | nghiêng ngang thân |
| run_joint_deviation_arms | −0,1 | giữ khuỷu/cổ tay/vai roll-yaw gần mặc định; vai pitch tự do |

Giới hạn mô-men, gain PD, mô hình robot: **không đổi**.

## Kiểm chứng đã chạy (GPU1)

**Dấu và độ lớn (probe PhysX, actor ONNX, không đẩy/nhiễu):** gối sâu 0,44 m lúc đứng (giả định ban đầu 0,30 m
sai, đã sửa); thân ngả 0,3° ở 1,0 m/s và 3,2° ở 1,5 m/s; ở 1,5 m/s chạm đất 0,30 s, vung 0,22 s, 22–30% hai
chân cùng chạm, 0% bay; lực đẩy trung bình 0,057 lần trọng lượng.

**Số đo kỹ thuật MuJoCo của baseline** (ONNX + giữ hướng, lệnh 1,5 m/s, seed 95000, 1,39 m/s):

| Chỉ số | Baseline |
|---|---|
| Tỷ lệ thời gian bay | 0% |
| Hai chân cùng chạm | 18,5% |
| Chạm đất / vung (trung vị) | 0,31 s / 0,21 s |
| Cadence | 3,8 bước/s |
| Tiếp đất mũi chân / gót | 0% / 100% |
| Lực ngang đỉnh mỗi lần chạm | 0,11 lần trọng lượng |
| Gối cao nhất khi vung (sâu dưới hông) | 0,40 m (đứng 0,44 m) |
| Gập cổ chân tối đa khi vung | −0,25 rad |
| Biên độ vai pitch | 0,17 / 0,14 rad; tương quan vai–hông cùng bên −0,98 / −0,96 |
| Thân ngả | 3,6° |

Số đo MuJoCo dùng cùng quy tắc tiếp xúc (20 N, khử nhiễu 40 ms) và khớp probe Isaac (0,31 vs 0,30 s chạm,
0,21 vs 0,22 s vung, 3,8 bước/s). Thêm telemetry không đổi vật lý: baseline vẫn bit-exact.

**Smoke train Sprint** (4096 env, ghép actor ONNX, std 0,1, warm-up critic):
- Dải mặc định 0–1,5 m/s, 40 iteration: ngã ~0,1%, reward 0,7 → 23,7, không NaN; term sprint = 0 đúng thiết kế
  (chưa vượt ngưỡng); curriculum chưa tăng trong 40 iteration.
- Ép dải 0–2,5 m/s, 20 iteration: cả 12 term sprint hữu hạn, khác 0, đúng dấu; ngã 36% (policy chưa biết chạy).
  Tổng term sprint (~0,03/tập-giây) nhỏ so với bám vận tốc (~0,30) — có thể phải tăng sau ablation.
- 4096 env: VRAM 6,6 GB, ~39k bước/s khi CPU đang bị job khác chia sẻ.

## Xem Isaac Sim

Quay video cần renderer RTX của Kit. Với GPU0 ẩn khỏi CUDA và `activeGpu=1`, renderer crash trong
`rtx.scenedb` lúc khởi tạo. Thử các cấu hình renderer khác đã bị chặn vì một cấu hình bật multi-GPU mặc định
có rủi ro chạm GPU0. Training không camera không bị ảnh hưởng. Chờ quyết định — xem báo cáo.

## Tiêu chí chấp nhận đề xuất (chờ chốt)

Benchmark MuJoCo 10 seed mới: không ngã, giữ 60 s, giảm tốc và dừng, cùng các ngưỡng tốc độ/quay/hướng cũ;
thêm cho pha chạy ở mốc 2,0 / 2,5 / 3,0 m/s: tỷ lệ bay ≥ X%, chạm đất trung vị ≤ Y s, tiếp đất mũi ≥ Z%,
hai chân cùng chạm ≈ 0, mô-men bị cắt và mức sử dụng đỉnh được báo cáo, không nới giới hạn actuator.
