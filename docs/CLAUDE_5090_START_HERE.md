# Bàn giao cho Claude trên RTX 5090 — đọc trước khi thực hiện

Bạn đang tiếp quản dự án `g1-robust-locomotion-lab` của tuanfptu. Làm việc trực tiếp trên nhánh
`main`, theo yêu cầu người dùng chỉ giữ một nhánh. Mục tiêu: humanoid G1 chạy nhanh, giữ thăng
bằng, giảm tốc và dừng được, hoàn toàn mô phỏng. Mục tiêu 11 m/s là hướng nghiên cứu dài hạn,
chưa chứng minh khả thi với mô hình/giới hạn actuator này; không hứa thời gian hội tụ.

## 1. Xác minh gói và phục hồi Git

Gói chứa source, vendor public ghim revision, kết quả gần nhất, video và hai bundle SB3 cũ.
Không chứa môi trường Python cũ, thông tin đăng nhập, Isaac runtime hoặc USD training.

Chạy trên máy đích, trước khi sửa hoặc chạy những lệnh có thể cập nhật Git index:

```bash
sha256sum -c g1-5090-handoff.tar.gz.sha256
tar -xzf g1-5090-handoff.tar.gz
cd g1-robust-locomotion-lab
sha256sum -c TRANSFER_SHA256SUMS > /tmp/g1-transfer-verify.log
cat TRANSFER_MANIFEST.json

git init -b main
git fetch artifacts/transfer/project.bundle main
git reset --mixed FETCH_HEAD
git remote add origin https://github.com/tuanfptu/g1-robust-locomotion-lab.git
git status --short
```

`TRANSFER_*` là metadata local được Git ignore. Không cần đưa chúng lên repo. `project.bundle` giữ lịch sử main, không chứa cấu hình credentials của máy cũ.
GitHub authentication phải đăng nhập riêng trên máy mới nếu cần pull/push repo private.
Vendor đã được chuyển kèm Git metadata tối thiểu và public remote sạch, không cần fetch lại để
chạy baseline. Khi có mạng vẫn có thể dùng `scripts/fetch_lab.sh` để kiểm tra/tải đúng revision.

## 2. Trạng thái đã xác nhận

- Máy cũ: GTX 1660 Ti 6 GB, RAM 16 GB; MuJoCo CPU và ONNX Runtime CPU, GPU render EGL.
- Policy mới là ONNX Unitree RL Lab 29 action/480 observation/history 5, không phải teacher
  12 khớp chân của pipeline SB3 cũ.
- Vòng giữ hướng: `yaw = clip(-gain * heading, -0.2, 0.2)` rad/s.
- Baseline gain 2/ramp 3 s: khoảng 1,394 m/s, 5/5 seed giữ 60 s rồi dừng.
- Quét 46 lượt: gain 1/ramp 3 s đạt khoảng 1,391 m/s, 10/10 seed riêng giữ 60 s rồi dừng.
  Lệnh 1,6 và 1,65 m/s sống sót nhưng dao động quay vượt ngưỡng. Chưa tăng mốc tốc độ đạt chuẩn.
- Không có cập nhật RL mới cho policy 29 khớp trong các benchmark trên. 27 tests và lint/format
  đã đạt trên máy cũ; refactor khớp số đo cũ. Đọc `docs/CLEANUP_1660_REPORT.md`.
- `runs/autonomous-1660/selected*` là SB3 legacy dựa trên teacher chân, **không phải** full
  RSL-RL checkpoint cho pipeline 29 khớp; chúng được mang theo để lưu giữ/tái lập, không để resume Isaac.
- Không làm tiếp pilot tự đứng dậy ở giai đoạn này.

## 3. Việc đầu tiên trên host mới

Kiểm tra `nvidia-smi`, hệ điều hành, RAM, dung lượng ổ đĩa và driver. Ngân sách đề xuất:
RAM 64 GB và khoảng 150 GB trống trở lên cho runtime/assets/logs, không phải cam kết tối thiểu.

Tạo **hai môi trường riêng**:

**A. MuJoCo baseline/evaluation** — có thể giữ Torch CPU, vì physics native MuJoCo chạy CPU.
Dùng Python 3.10 để tái hiện stack cũ (cài Python/venv phù hợp nếu host chưa có):

```bash
python3.10 -m venv .venv-eval
.venv-eval/bin/python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
.venv-eval/bin/python -m pip install -e '.[dev,lab]'
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv-eval/bin/python -m pytest -q
.venv-eval/bin/python scripts/tune_lab_controller.py --output artifacts/5090-baseline-check
```

Nếu chỉ cần kiểm tra import, policy và ánh xạ khớp trước: chạy
`.venv-eval/bin/python -m pytest tests/test_lab_contract.py -q`.
Muốn xem video đã có: `artifacts/controller-1660-render/g1-1660.mp4`.

**B. Isaac training** — tạo env/container riêng theo hướng dẫn chính thức, **không** dùng
`scripts/install_cpu.sh` hoặc `requirements-lock.txt` của evaluation trong env này.

- Chọn cặp Isaac Sim / Isaac Lab và CUDA PyTorch tương thích RTX 5090 (Blackwell).
- README upstream ghim hiện quảng bá Sim 5.1/Lab 2.3. Đây là điểm đối chiếu API, chưa phải
  stack được kiểm chứng trên 5090. Tài liệu NVIDIA hiện ghi 5.1 hết hỗ trợ. Nếu dùng bản mới,
  port API có kiểm thử, ghi rõ version/commit; không nâng tất cả package mù quáng.
- Cài RSL-RL theo stack Isaac đã chọn; kiểm tra API PPO config/runner upstream.
- Cài môi trường Unitree bằng Python Isaac:
  `python -m pip install -e vendor/unitree_rl_lab/source/unitree_rl_lab`.
- Để launcher local có thể import g1lab mà không kéo Torch CPU/Gymnasium cũ vào env Isaac:
  `python -m pip install --no-deps -e .`. Các script preflight/prepare chỉ cần Python chuẩn +
  Torch sẵn có; evaluation/render chạy bằng `.venv-eval` riêng.
- Kiểm tra CUDA bằng phép toán tensor thật, không chỉ `torch.cuda.is_available()`.

Nguồn chính thức để chọn đúng stack lúc cài:

- https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html
- https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html
- `vendor/unitree_rl_lab/README.md`

## 4. Asset và smoke training

Tải G1 USD chính thức, giữ nguyên cấu trúc tham chiếu từ
https://huggingface.co/datasets/unitreerobotics/unitree_model/tree/main .
Asset MuJoCo đã kèm **không thay thế** USD cho Isaac. Kiểm tra path `UNITREE_MODEL_DIR` và
các asset được import khi đăng ký task; ưu tiên override có ghi nhận, tránh sửa vendor không lưu diff.

Trong env Isaac, với file USD thật:

```bash
python scripts/check_training_host.py \
  --usd /actual/path/g1_29dof_rev_1_0.usd \
  --output artifacts/5090-host.json --require-training

python scripts/prepare_lab_fast.py \
  --python "$(command -v python)" --usd /actual/path/g1_29dof_rev_1_0.usd \
  --cap 1 --envs 32 --iterations 2 --output artifacts/5090-smoke --execute
```

Preflight pass chưa đủ: smoke phải khởi tạo được simulator, task và PPO, tạo log/checkpoint,
không NaN. Hydra overrides và runtime Isaac **chưa được chạy trên máy cũ**. Nếu lỗi thì sửa
compatibility trước, không tăng quy mô. Output phải chưa tồn tại; dùng tên mới mỗi lần thử.
Nếu thay revision vendor, launcher hiện cố ý chặn revision khác; cập nhật pin có lý do, ghi diff,
và kiểm tra lại hợp đồng observation/action thay vì bỏ kiểm tra.

## 5. Chọn cách huấn luyện, rồi mới chạy dài

Repo public có ONNX export, không có full PPO `.pt` đã xác nhận để tiếp tục. ONNX không chứa
critic/optimizer; không dùng nó như checkpoint resume. BC/distillation actor 29 khớp **chưa triển khai**.

Sau smoke, đánh giá 3 lựa chọn: full checkpoint tương thích nếu có; triển khai BC từ ONNX rồi
PPO; hoặc PPO từ đầu. Mọi lệnh không có `--checkpoint` hiện là PPO từ đầu. Chốt lựa chọn và ghi
vào protocol trước khi chạy dài. BC từ teacher đi chậm chỉ khởi tạo kỹ năng, không tự tạo kỹ năng sprint.

Sau đó đo 512 env: VRAM, steps/s, NaN, log reward/episode. Chỉ tăng 1024/2048/4096 khi số đo
cho phép. PPO + curriculum tốc độ là baseline; motion prior/AMP có thể thử sau, có ablation.
Mục tiêu gần: 2–3 m/s thực tế, giữ 60 s, 10 seed mới, giảm tốc/dừng; sau đó mở rộng.
Giữ giới hạn vật lý, không dùng đỉnh tốc độ trước ngã làm thành tích.

Khi export policy mới: dùng `scripts/benchmark_lab_policy.py --policy-dir ...` trong env eval;
bundle phải có `params/deploy.yaml` và `exported/policy.onnx` đúng contract. Không mặc định
PhysX và MuJoCo cho cùng kết quả. Lưu checkpoint + config + normalization + export + protocol,
commit code lên main và báo cáo rõ phần thực chạy, phần chưa xác nhận, các lượt bị loại.
