# G1 toàn thân từ Unitree RL Lab

Đã chạy policy ONNX công khai của [Unitree RL Lab](https://github.com/unitreerobotics/unitree_rl_lab)
trên mô hình G1 gốc của Unitree MuJoCo. Với vòng giữ hướng bên ngoài policy, tốc độ tiến thực tế
trung bình **1,394 m/s (5,02 km/h)** ở lệnh 1,5 m/s; **5/5 seed** hoàn thành giữ tốc độ 60 giây,
giảm tốc rồi dừng. Mỗi lượt dài 73 giây mô phỏng. Đây là mức cao nhất vượt bộ tiêu chí trong
các mức đã thử, chưa phải cực đại toàn cục hoặc thành tích chạy nhanh.

Không có cập nhật trọng số RL trong đợt này. Policy mới điều khiển **29 khớp**, khác teacher
12 khớp chân trong `UNITREE_SPEED.md`; hai benchmark khác mô hình và tiêu chí, không phải
so sánh thuật toán có kiểm soát. Chưa đánh giá địa hình, lực đẩy, độ trễ hay sai lệch mô hình.

## Đã xây dựng

- Adapter ONNX Runtime CPU: 480 quan sát, lịch sử 5 bước theo từng nhóm quan sát, 29 action;
  ánh xạ đúng thứ tự khớp và actuator, dùng PD gains từ cấu hình deploy.
- MuJoCo bước vật lý 0,002 giây, policy 0,02 giây; giữ giới hạn mô-men actuator trong XML.
- Benchmark tăng tốc, giữ tốc độ, giảm tốc, dừng; lưu từng trajectory, seed, hash policy và protocol.
- Giữ hướng bằng `yaw_command = clip(-2 * heading, -0.2, 0.2)` rad/s. Đây là bộ điều khiển
  ngoài policy, không phải kỹ năng mới đã học.
- Bộ chuẩn bị lệnh PPO Isaac Lab theo các mức vận tốc; trạng thái hiện tại chỉ là chuẩn bị.

## Kết quả và cách đo

Lịch mỗi lượt: đứng 2 giây → tăng tốc 3 giây → giữ 30 hoặc 60 giây → giảm tốc 3 giây → dừng 5 giây.
Vận tốc báo cáo là vận tốc pelvis trong hệ trục thân, lấy trung bình từ giây 10 đến hết đoạn giữ.
Các seed chỉ thay đổi nhiễu khớp ban đầu nhỏ; đây chưa phải kiểm chứng robustness diện rộng.

Một lượt đạt khi: không ngã; vận tốc tiến ≥85% lệnh; RMSE vận tốc phẳng ≤max(0,15 m/s,
20% lệnh); tốc độ dừng trung bình 2 giây cuối <0,15 m/s; RMS vận tốc quay <0,25 rad/s;
góc lệch hướng tuyệt đối <30°. Ngã nếu pelvis thấp hơn 0,45 m, độ thẳng đứng <0,45 hoặc trạng thái không hữu hạn.

| Cấu hình / lệnh | Kết quả | Diễn giải |
|---|---|---|
| ONNX nguyên bản, 0,75 m/s | 5/5 lượt giữ 60 giây đạt | Vận tốc thực tế khoảng 0,733 m/s |
| ONNX nguyên bản, 1 m/s | Sống sót 3/3, không đạt | Trôi hướng vượt 30° |
| ONNX + giữ hướng, 1,5 m/s | 5/5 lượt giữ 60 giây đạt | Trung bình 1,394 m/s; lệch hướng tối đa 3,76° |
| ONNX + giữ hướng, 1,6 m/s | Sống sót 3/3 lượt giữ 30 giây, không đạt | RMS quay 0,319–0,330 rad/s |
| ONNX + giữ hướng, 1,65 m/s | Sống sót 3/3 lượt giữ 30 giây, không đạt | RMS quay 0,481–0,491 rad/s |
| ONNX + giữ hướng, 1,7 m/s | Ngã 3/3 | Ngã sau khoảng 12–14,5 giây |
| ONNX + giữ hướng, 1,75; 2; 2,5; 3 m/s | Ngã ở mọi lượt đã thử | Không dùng tốc độ tức thời trước ngã làm thành tích |

Ở năm lượt validation 1,5 m/s: RMSE phẳng 0,123–0,127 m/s; tốc độ dừng
0,0011–0,0026 m/s; RMS quay 0,172–0,173 rad/s. Snapshot số liệu nằm trong
[experiment-data](experiment-data/unitree-lab-29dof-heading-v2/summary.json), gồm cả cấu hình
[không giữ hướng](experiment-data/unitree-lab-29dof-baseline-v1/summary.json) và
[các mức bổ sung bị loại](experiment-data/unitree-lab-29dof-refined-v3/episodes.csv).
Video local: `artifacts/unitree-lab-29dof-heading-v2/lab-policy.mp4` (lượt minh họa giữ 30 giây,
khác lượt validation 60 giây). Video và dữ liệu trajectory lớn không đưa lên Git.

## Tái lập trên máy hiện tại

```bash
bash scripts/fetch_lab.sh
.venv/bin/python -m pip install -e '.[lab]'
.venv/bin/python scripts/benchmark_lab_policy.py \
  --heading-hold --output artifacts/lab-repeat
# Có EGL hoạt động thì thêm --video, với MUJOCO_GL=egl EGL_PLATFORM=surfaceless.
```

Thư mục output phải chưa tồn tại. Bỏ `--heading-hold` để đo ONNX nguyên bản. Có thể dùng
`--policy-dir` trỏ tới bundle chứa `params/deploy.yaml` và `exported/policy.onnx` cùng hợp đồng
khớp/quan sát. Phép thử bổ sung 1,6/1,65/1,7 gọi `trial` (hiện ở `g1lab.lab_evaluation`), seed 92000–92002,
`hold=30, heading_hold=True`; protocol và bảng kết quả được lưu riêng trong snapshot v3.

Nguồn được ghim:

- Unitree RL Lab: `4960b84732b0c2ec593dccbfe963fda1bcd7b1e3`.
- Unitree MuJoCo: `1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d`.

MuJoCo và PhysX/Isaac khác nhau; PD deployment cũng khác actuator ngầm khi training.
Kết quả ở đây là đo trên adapter MuJoCo, không phải xác nhận hiệu suất training PhysX.

## Huấn luyện tiếp khi có máy phù hợp

Pipeline upstream dùng Isaac Sim/Isaac Lab, PPO của RSL-RL. GTX 1660 Ti không có RT cores
và không đáp ứng pipeline Isaac Sim này theo
[yêu cầu NVIDIA](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html).
Máy hiện tại cũng chưa có runtime Isaac. Vì chưa có máy RTX để truy cập, **chưa chạy training
upstream**, chưa kiểm chứng tương thích phiên bản Isaac/RSL-RL hoặc các Hydra override lúc chạy.

Lộ trình thử nghiệm: mặt phẳng, lệnh tối đa 1 → 1,5 → 2 → 2,5 → 3 m/s; chỉ lên mức khi qua
validation nhiều seed, giữ tốc độ và dừng. Gait period/stance fraction trong cấu hình là giả thuyết
cần thử, không phải thông số đã chứng minh. Không tăng giả giới hạn mô-men hay tốc độ khớp.

```bash
.venv/bin/python scripts/prepare_lab_fast.py \
  --usd /actual/path/to/g1_29dof_rev_1_0.usd \
  --cap 1 --envs 512 --iterations 1000 --output artifacts/prepare-example
```

Lệnh trên chỉ tạo `command.json`. Trên host RTX đã cài runtime tương thích, chạy launcher
bằng chính Python Isaac, cung cấp `--python` cùng interpreter và `--execute` để bắt đầu.
Cần tải/cấu hình USD chính thức và kiểm tra runtime trước; snapshot `lab-fast-prepared` hiện
chứa đường dẫn USD placeholder, không thể chạy nguyên trạng.

Repo công khai có ONNX export, không có checkpoint PPO đầy đủ để resume. ONNX thiếu critic
và optimizer; không truyền nó như checkpoint huấn luyện. `--checkpoint` chỉ nhận checkpoint
RSL-RL `.pt` đầy đủ, tương thích kiến trúc. Không có checkpoint đó thì cấu hình chuẩn bị là
PPO từ đầu. Khởi tạo actor bằng imitation/BC từ ONNX là hướng tiếp theo, **chưa triển khai**.
