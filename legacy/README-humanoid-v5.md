# Humanoid Lab — RL trong MuJoCo

Mục tiêu: xây dựng humanoid học vận động bằng RL, có thí nghiệm tái lập và demo trực quan.
Bản đầu dùng Humanoid-v5 của Gymnasium và PPO của Stable-Baselines3.
Đây là humanoid benchmark, chưa phải mô hình robot thương mại hay hệ thống sim-to-real.

## Chạy bản nền

Python 3.10 trở lên. Tạo môi trường riêng rồi cài dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python humanoid_lab.py train --steps 3000000 --envs 4
python humanoid_lab.py eval --episodes 10
python humanoid_lab.py eval --episodes 3 --video
tensorboard --logdir runs
```

Máy Linux không có màn hình có thể cần `MUJOCO_GL=egl` và driver EGL để xuất video.
Huấn luyện không yêu cầu render. Các môi trường hiện chạy tuần tự trong DummyVecEnv;
`--envs` tăng số trajectory mỗi rollout, không tạo worker CPU song song.

Kiểm tra nhanh pipeline bằng `--steps 2048 --envs 1 --run runs/smoke`.
Đây chỉ là kiểm tra chạy, không đủ để robot học đi. PPO làm tròn số bước theo rollout.
Mặc định dùng CPU; thời gian và chất lượng học phụ thuộc phần cứng, seed và số bước.
Ba triệu bước là ngân sách khởi đầu, không bảo đảm hội tụ.

Mỗi run chứa config, monitor CSV, TensorBoard, policy và thống kê chuẩn hóa observation/reward.
Checkpoint định kỳ cũng lưu thống kê chuẩn hóa; phải dùng đúng cặp khi tải.
Lệnh eval dùng policy cuối và vecnormalize.pkl, tắt cập nhật thống kê, báo reward gốc.
Kết quả ghi vào evaluation.json; video vào videos/. Chạy eval lại sẽ ghi đè báo cáo.

## Lộ trình nâng cấp

1. Baseline: train/eval/video; so sánh ít nhất 3 seed với cùng ngân sách bước.
2. Command-conditioned locomotion: thêm lệnh vận tốc tiến/ngang/quay vào observation;
   reward bám lệnh, phạt trượt chân, năng lượng và thay đổi action.
3. Curriculum: tăng dần vận tốc, độ khó địa hình và nhiễu lực đẩy.
4. Robustness: randomize ma sát, khối lượng, actuator strength; đánh giá riêng ngoài phân phối train.
5. Demo tương tác: điều khiển vận tốc, chủ động đẩy robot và hiển thị metric.

Các mục 2–5 là kế hoạch, chưa được triển khai. Chỉ chuyển sang robot thật sau một quy trình
kiểm chứng phần cứng và điều khiển riêng; benchmark này không xác nhận khả năng chuyển giao.

Đánh giá nghiên cứu: return, thời gian đứng/đi, vận tốc thực tế, tracking error khi có lệnh,
năng lượng, tỷ lệ ngã và độ phục hồi sau lực đẩy. Báo trung bình và độ phân tán trên nhiều seed;
không dùng một video đẹp để kết luận policy tốt.

## Tài liệu

- https://gymnasium.farama.org/environments/mujoco/humanoid/
- https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
