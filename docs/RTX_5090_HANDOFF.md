# Chuyển dự án sang RTX 5090

Mục tiêu đợt chuyển máy: xác nhận runtime, chạy PPO smoke nhỏ, sau đó mới tăng số môi trường.
Chưa có truy cập máy RTX nên chưa xác nhận thời gian train, thông lượng hay tương thích runtime.

## Mang theo gì?

- Clone repo và dùng nhánh `main`; mã và kết quả bàn giao đã được hợp nhất vào nhánh này.
- Chạy `bash scripts/fetch_lab.sh` để tải lại đúng revision policy và mô hình Unitree.
- Mang theo `artifacts/controller-1660-v1/` và `artifacts/controller-1660-render/` nếu muốn xem toàn bộ
  trajectory/video. Các bảng kết quả nhỏ đã nằm trong Git; không cần chuyển `.venv`, cache hay token.
- Policy ONNX được fetch từ upstream; checkpoint SB3 cũ không dùng để resume RSL-RL.

## Chuẩn bị host

Dùng máy RTX 5090 với driver, hệ điều hành, CUDA PyTorch và Isaac tương thích nhau. Khuyến nghị
cho dự án: Linux, RAM 64 GB, SSD còn trống khoảng 150 GB trở lên cho runtime/assets/logs;
đây là ngân sách vận hành đề xuất, không phải thông số tối thiểu của NVIDIA.

Đối chiếu [yêu cầu Isaac Sim hiện hành](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html)
và [hướng dẫn Isaac Lab](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).
Revision Unitree được ghim quảng bá Isaac Sim 5.1 / Isaac Lab 2.3 trong README; cần thử cặp này
hoặc thực hiện port có kiểm thử sang phiên bản được hỗ trợ. Tài liệu NVIDIA hiện đánh dấu
[Isaac Sim 5.1 đã hết hỗ trợ](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html).
Vì vậy chưa coi cấu hình version cũ là stack production đã xác nhận cho máy mới.

**Tạo môi trường Isaac riêng.** Không chạy `scripts/install_cpu.sh` trong môi trường Isaac vì script
đó cài Torch CPU. Không mang nguyên `.venv` Python 3.10 của 1660 sang. Cài Torch CUDA theo
stack Isaac đã chọn và xác nhận có thể chạy tensor trên GPU, không chỉ nhìn `nvidia-smi`.
Cài project và upstream bằng chính Python đó; khi cài project local cần kiểm tra dependency
MuJoCo/Gymnasium/SB3 có xung đột với stack Isaac không. Launcher không cần import SB3.

Tải USD G1 29 khớp và giữ đầy đủ cấu trúc tham chiếu từ
[Unitree model](https://huggingface.co/datasets/unitreerobotics/unitree_model/tree/main).
Launcher hỗ trợ override đường dẫn USD; chưa thay đổi file asset upstream. Cần kiểm tra mọi asset
khác được import khi đăng ký task trước khi bắt đầu train.

## Trình tự kiểm tra

Trong các lệnh dưới, `python` phải là interpreter Isaac đã cấu hình; thay đường dẫn USD bằng file thật.

```bash
python scripts/check_training_host.py \
  --usd /actual/path/g1_29dof_rev_1_0.usd \
  --output artifacts/5090-host.json --require-training

python scripts/prepare_lab_fast.py \
  --python "$(command -v python)" --usd /actual/path/g1_29dof_rev_1_0.usd \
  --cap 1 --envs 32 --iterations 2 --output artifacts/5090-smoke --execute
```

Preflight chỉ kiểm tra package/CUDA/đường dẫn, **không chứng nhận Isaac chạy được**. Smoke 32 env
phải khởi tạo scene thành công, không NaN, tạo log và checkpoint. Khi smoke lỗi, xem `train.log`,
kiểm tra API Isaac/RSL-RL và Hydra override trước khi tăng quy mô.

Sau smoke đạt: thử 512 env, đo VRAM và số bước/giây; cân nhắc tăng 1024/2048/4096 theo số đo.
Không có cam kết 4096 env phù hợp hoặc nhanh nhất trước khi đo. Lưu driver, package versions,
source commit, seed, cấu hình và TensorBoard cho mỗi run.

## Học tiếp bằng gì?

ONNX public không có critic/optimizer. Hiện chưa triển khai BC khởi tạo actor 29 khớp; do đó lệnh
training không checkpoint là **PPO từ đầu**, không phải fine-tune policy Unitree.
Nếu có full checkpoint RSL-RL tương thích, dùng `--checkpoint /path/run/model_N.pt`.
Khi tiếp tục run, giữ cả cấu hình và observation normalization của run gốc; chỉ file `.pt` đúng
đuôi chưa đủ chứng minh kiến trúc tương thích.

Thứ tự phát triển sau smoke:

1. Chọn có chủ đích: lấy full checkpoint hợp lệ, triển khai BC → PPO, hoặc train PPO từ đầu.
2. Tái hiện đứng/đi/dừng ở mức 1 m/s trước khi tăng mục tiêu.
3. Tăng dải tốc độ 1 → 1,5 → 2 → 2,5 → 3 m/s; các gait override hiện tại chỉ là giả thuyết.
4. Export ONNX/deploy YAML, dùng benchmark MuJoCo cùng tiêu chí, seed mới; sau đó thử lực đẩy,
   ma sát và độ trễ. Chỉ tăng cấp khi tốc độ thực tế và ổn định cùng đạt.

Không tiếp tục mục tiêu đứng dậy trong đợt này. Giữ checkpoint đi tốt làm baseline riêng;
không thay mặc định chỉ vì reward training cao hơn.
