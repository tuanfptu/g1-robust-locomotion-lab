# SPRINT: hiện trạng tài nguyên và đề xuất tích hợp

Paper: *SPRINT: Efficient Spectral Priors for Humanoid Athletic Sprints*, arXiv:2605.28549 (27/05/2026).
Kiểm tra ngày 13/11/2026. Chưa có dòng code SPRINT nào trong repo này; đây là đề xuất, chưa phải kết quả.

## Code, data, checkpoint chính thức: chưa có

- Trang dự án `anonymous.4open.science/w/SPRINT-138A/` chỉ có `index.html` và video. API repo ẩn danh
  liệt kê thư mục `SPRINT` nhưng truy cập trả `{"error":"not_connected"}` (nguồn không còn liên kết).
- GitHub search "SPRINT spectral humanoid" / "spectral priors humanoid sprint": 0 kết quả.
- Không có link Hugging Face, dataset hay checkpoint trong paper/trang dự án.

Vì vậy việc "port Isaac Gym → Isaac Lab" thực chất là **tái hiện từ paper** trên Isaac Lab, không phải
chuyển code. Cần kiểm tra lại định kỳ; nếu tác giả công bố code thì đối chiếu trước khi tự viết tiếp.

## Paper mô tả gì (đủ để triển khai)

1. **Thư viện tham chiếu**: 5 chuỗi LAFAN1 (đi, chạy chậm, chạy; tổng 428 khung, 14 s) retarget sang G1
   bằng GMR; giữ 10 khớp (hip pitch, knee, ankle pitch, shoulder pitch, elbow × 2 bên); mỗi chu kỳ chuẩn
   hóa 10 s, nội suy Fourier 30→60 Hz, lọc Savitzky–Golay. Cặp vận tốc/tần số:
   {0,66; 1,10; 2,29; 2,87; 3,40} m/s ↔ {0,68; 0,86; 1,25; 1,36; 1,58} Hz.
2. **Spectral prior**: VAE mã hóa tần số f → z; vector điều kiện c=[z,f]; decoder MLP với FiLM
   (γ = 1 + 0,1 tanh(·), β = 0,1 tanh(·)) nhận K = 6 hài sin/cos của pha φ = 2πft; loss MSE + β_KL·KL.
3. **Policy**: q_target = q_ref + α·a_res; PPO; actor proprioceptive (vận tốc góc, trọng lực, lệnh, khớp,
   residual trước đó), critic thêm vận tốc tuyến tính và q_ref. MLP 256 + LSTM 64; 200 Hz/50 Hz;
   curriculum v ∈ [0, 6] m/s (f ∈ [0,6; 2,3] Hz), ω ∈ [−0,7; 0,7] rad/s.
4. **Reward**: bám lệnh đã lọc giới hạn gia tốc; prior Σ exp(−‖q_j − q_ref‖²/σ_q); thời gian bay
   t_target = 0,4/f_cmd; nghiêng thân theo tốc độ g_f = 0,1 + 0,05·v; bảng phạt (chân chạm, trượt,
   mô-men, gia tốc khớp, giới hạn khớp, chiều cao thân…).
5. Kết quả công bố: 5,96 m/s sim / 6 m/s thực trên G1, hội tụ 6,5 h trên một RTX 4090 (Isaac Gym).
   Ablation: bỏ curriculum chỉ còn 4,48 m/s; bỏ AAC 5,46 m/s.

**Thiếu hoặc mơ hồ** — phải tự chọn và ghi vào protocol: hàm ánh xạ v_cmd → f ngoài 5 điểm; α; mọi
σ và trọng số w_v, w_ω, w_p, w_a, w_t; kích thước VAE, β_KL; cách chọn z lúc chạy (mean hay sample);
xử lý pha khi f đổi (công thức φ = 2πft gây nhảy pha — cần tích lũy φ += 2πf·dt); policy có thấy pha
hay không; khớp ngoài 10 khớp lấy tham chiếu gì; gain PD, action scale, dải domain randomization,
lịch curriculum; clip LAFAN1 cụ thể. Hai bảng reward trong paper có trọng số nhưng không có σ.

## Đánh giá chuyển sang Isaac Lab (stack hiện tại: Sim 5.1 / Lab 2.3.2 / rsl-rl 3.1.2)

| Thành phần | Trong Isaac Lab | Công sức |
|---|---|---|
| Prior (VAE + FiLM) | Module PyTorch độc lập, train offline, đóng băng; không phụ thuộc simulator | Thấp |
| q_ref theo pha | `ActionTerm` mới kế thừa `JointPositionAction`: buffer pha từng env, tích lũy theo f(v_cmd), reset ngẫu nhiên pha | Trung bình |
| Residual action | Cùng ActionTerm: target = q_ref (10 khớp) / default (19 khớp) + α·a | Thấp |
| Quan sát | Policy giữ 480 chiều hiện tại (last_action thành residual); critic thêm q_ref, pha | Thấp |
| Reward | Hàm reward manager mới: prior, air-time theo f, tilt, lệnh lọc gia tốc | Trung bình |
| LSTM | Có `RslRlPpoActorCriticRecurrentCfg`; nhưng export ONNX và eval MuJoCo hiện chỉ hỗ trợ MLP + history | Trung bình–cao |
| Sim 200/50 Hz | Đã trùng (dt 0,005, decimation 4) | Không |
| Vật lý | PhysX khác bản Isaac Gym; hyperparameter không chuyển nguyên được | Rủi ro, không phải code |

Eval MuJoCo hiện tại phải mở rộng: bộ sinh pha/q_ref phải chạy bên ngoài ONNX (hoặc export prior thành
ONNX thứ hai) để `benchmark_lab_policy.py` chấm cùng tiêu chí.

## Đề xuất tích hợp theo giai đoạn

**Giai đoạn 0 — quyết định dữ liệu (cần anh/chị chọn).** LAFAN1 theo CC BY-NC-ND 4.0: chỉ phi thương
mại; dữ liệu retarget có thể bị coi là phái sinh, không nên công bố/đẩy lên repo. GMR là MIT và hỗ trợ
sẵn LAFAN1 → `unitree_g1` 29 DoF. Phương án thay thế không vướng license: tự trích chu kỳ khớp từ policy
hiện có (chỉ có dáng đi ≤1,5 m/s, không có pha bay — chỉ hợp để thử bộ máy, không đủ cho sprint).

**Giai đoạn 1 — prior offline (không cần GPU dài).** Retarget 5 clip, cắt chu kỳ, FFT kiểm tra tần số
có khớp bảng paper; train VAE+FiLM; báo L_rec, E_BA, hình so trajectory tại 5 tần số và 0,6/2,3 Hz.
Đạt khi tái tạo đúng 5 điểm và biên độ ngoại suy bị chặn, kiểm bằng giới hạn khớp G1.

**Giai đoạn 2 — residual PPO trên task gain v0.** ActionTerm + reward; train PPO từ đầu theo paper
(residual không dùng chung được actor ONNX vì ONNX xuất target tuyệt đối). Bắt đầu MLP + history 5
để giữ đường export/eval; LSTM là ablation sau. Curriculum 1 → 1,5 → 2 → 2,5 → 3 m/s; chỉ tăng khi đạt.

**Giai đoạn 3 — so sánh công bằng.** Cùng ngân sách bước mô phỏng và cùng benchmark MuJoCo 10 seed mới
(giữ 60 s, giảm tốc, dừng):
(a) baseline ghép actor ONNX + PPO + curriculum tốc độ (đã có pipeline),
(b) SPRINT residual + prior,
(c) SPRINT bỏ r_prior (ablation).
Thông lượng đo được: ~64k steps/s ở 4096 env; chưa có thời gian hội tụ trên máy này nên không ước
lượng số giờ.

Không có kế hoạch nào ở đây đảm bảo đạt 6 m/s trên mô hình MuJoCo/actuator hiện tại; mục tiêu gần vẫn là
2–3 m/s thực, giữ thăng bằng, dừng được.
