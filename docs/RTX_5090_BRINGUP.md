# Bring-up trên RTX 5090 (13/11/2026)

Đợt này dựng môi trường, tái hiện baseline, chạy smoke/đo quy mô Isaac và chốt cách khởi tạo PPO.
**Chưa có training dài, chưa có policy mới vượt baseline.** Checkpoint duy nhất được export là lượt
kiểm chứng pipeline 100 iteration, không dùng làm kết quả.

Số liệu nhỏ được lưu trong [experiment-data/5090-bringup](experiment-data/5090-bringup/); log thô,
checkpoint và video nằm trong `artifacts/`, `vendor/unitree_rl_lab/logs/` (Git ignore).

## Host

| Hạng mục | Giá trị |
|---|---|
| GPU | 2× RTX 5090 32 GB, compute capability 12.0 |
| Driver | 595.99.02 (CUDA 13.2); NVIDIA chỉ ghi Isaac Sim 5.1 đã test với 580.65 |
| OS / CPU / RAM | Ubuntu 24.04.5, kernel 6.14, Ryzen 9 9950X, 123 GiB |
| Disk trống | ~580 GiB |

**Dự án chỉ chạy trên GPU1** (`nvidia-smi` index 1, PCI `0000:05:00.0`,
UUID `GPU-20e5a706-6231-50a3-4c9b-8e8d4db3c470`). GPU0 dành cho tiến trình khác và không được động tới.
GPU1 báo **PCIe Gen1 x1** khi rảnh (tối đa Gen4 x16; GPU0 x16/Gen5). Chưa đo ảnh hưởng; pipeline Isaac
giữ dữ liệu trên GPU nên có thể nhỏ, nhưng nên kiểm tra khe cắm/riser.

### Cơ chế ghim GPU1

Ghim theo UUID/PCI, không theo số thứ tự: CUDA mặc định sắp `FASTEST_FIRST`, không chắc trùng `nvidia-smi`.

1. [`g1lab/gpu.py`](../src/g1lab/gpu.py): `pin_cuda()` đặt `CUDA_VISIBLE_DEVICES=<UUID GPU1>` và
   `CUDA_DEVICE_ORDER=PCI_BUS_ID`, **từ chối** mọi giá trị khác (kể cả `1`, `0`, `0,1`). Được gọi trước khi
   import Torch/Isaac trong `prepare_lab_fast.py`, `check_training_host.py`, `isaac_train.py`,
   `isaac_eval_policy.py`. `pin_egl()` chọn thiết bị EGL của driver NVIDIA trên PCI `05:00.0` (glvnd liệt kê
   mỗi card hai lần: NVIDIA và Mesa) mà không tạo context; gọi trong `render_lab_result.py` và khi
   `lab_evaluation.trial` quay video.
2. Kit: cấu hình headless của Isaac Lab bật `renderer.multiGpu.enabled/autoEnable`. Launcher và eval truyền
   `--kit_args` tắt multi-GPU, `maxGpuCount=1`.
3. File `g1lab_gpu_pin.pth` trong site-packages của `envs/isaac51` và `.venv-eval` đặt mặc định UUID
   (và `MUJOCO_GL=egl`, `MUJOCO_EGL_DEVICE_ID=1` cho eval) cho cả lệnh Python ad-hoc. Đây là mặc định,
   không phải kiểm chứng; nếu môi trường được tạo lại thì phải tạo lại hai file này.
4. `~/TUAN__HUMAN_NOID_G1_RUNNER/tools/measure_run.sh` quét tiến trình compute **và** đồ hoạ trên GPU0
   mỗi 0,5 s, trả lỗi nếu thấy tiến trình có cmdline/cwd thuộc dự án.

Kiểm chứng: smoke 512 env và eval PhysX chạy không đặt biến môi trường thủ công; GPU0 không có tiến
trình nào của dự án, còn quét đối chứng trên GPU1 bắt được tiến trình eval; VRAM 3,9 GB nằm trên GPU1.
Bảng thiết bị của Kit không còn GPU nào ở trạng thái Active ngoài thiết bị CUDA nhìn thấy.
Các lượt trước khi ghim (đặt tay `CUDA_VISIBLE_DEVICES=1`) cũng chỉ kích hoạt GPU bus 05.

## Môi trường

**A. Eval MuJoCo** — `.venv-eval` (Python 3.10, Torch 2.6.0+cpu, MuJoCo 3.3.7, ONNX Runtime 1.22.0),
cài theo `requirements-lock.txt`. Torch CPU ở đây là cố ý: physics và ONNX chạy CPU.

**B. Training Isaac** — `envs/isaac51` ngoài repo (Python 3.11):
Isaac Sim 5.1.0, Isaac Lab v2.3.2 (`37ddf62`), Torch **2.7.0+cu128** (có `sm_120`), rsl-rl-lib 3.1.2,
unitree_rl_lab `4960b84` (editable), project cài `--no-deps`. Không có Torch CPU, MuJoCo hay SB3 trong env này.

- Chọn cặp Sim 5.1/Lab 2.3.2 vì vendor Unitree viết cho Lab 2.3; Lab 3.0 đổi backend (Newton mặc định)
  cần port API. Sim 5.1 đã hết hỗ trợ theo NVIDIA — đánh đổi có chủ đích, xem phần còn lại.
- `isaaclab.sh --install` dùng `uv` và **đã định nâng Torch lên 2.15.0/CUDA 13**; lỗi mạng chặn lại.
  Cài thủ công bằng pip với [constraints](experiment-data/5090-bringup/isaac51-constraints.txt):
  ghim torch/torchvision, `setuptools<81` (build `flatdict`), `psutil==5.9.8` (Isaac Sim),
  `starlette==0.49.1` (Isaac Lab ghim, ghi đè metadata fastapi của Isaac Sim).
- CUDA kiểm bằng phép toán thật: matmul 8192² FP32 ~68 TFLOPS, sai số so FP64 8e-4; MLP fwd/bwd hữu hạn.
- Đã chấp nhận EULA Omniverse qua `OMNI_KIT_ACCEPT_EULA=YES` để chạy headless.

Asset: 4 file USD G1 29 DoF từ `unitreerobotics/unitree_model` revision `323e350`, giữ nguyên cấu trúc,
tại `~/TUAN__HUMAN_NOID_G1_RUNNER/assets/unitree_model` ([checksum](experiment-data/5090-bringup/g1-usd-assets.txt)).
Đường dẫn truyền qua Hydra override, không sửa `UNITREE_MODEL_DIR` của vendor.

## Baseline MuJoCo — tái hiện chính xác

- 27/27 test (nay 29/29 sau bổ sung), `ruff check` đạt. `ruff format` báo 1 file trong `legacy/`, chưa sửa.
- `tune_lab_controller.py`: 46/46 lượt khớp `episodes.csv` của 1660 tới **8,9e-16**. Validation 10/10 đạt,
  1,3909 m/s, RMS quay 0,168 rad/s, lệch hướng tối đa 5,54°, dừng 0,00105 m/s.
- `benchmark_lab_policy.py --heading-hold`: `episodes.csv` **giống từng byte** gói chuyển; 5/5 seed,
  1,394 m/s. Lệnh 2,0/2,5/3,0 m/s: 0/3 sống sót.

## Isaac: smoke và quy mô

Smoke 32 env/2 iteration đạt: scene + PPO khởi tạo, obs policy 480/critic 495, log TensorBoard,
`model_0/1.pt`, không NaN, cùng seed cho trọng số giống hệt. Resume từ checkpoint đã kiểm (smoke-03).

Lỗi đã sửa: `train.py` vendor **bỏ qua `--experiment_name`**, nên checkpoint resume stage vào
`logs/rsl_rl/g1_fast` không bao giờ được tìm thấy. Launcher thêm override `agent.experiment_name=g1_fast`.
Launcher cũng ghi provenance (version, driver, GPU, commit, thư mục run, wall time) vào `command.json`.

Quy mô (task vendor gain gốc, flat plane, PPO từ đầu, cap 1 m/s, seed 42; chỉ đo hiệu năng):

| Env | VRAM đỉnh | steps/s (median nửa sau) | s/iteration | GPU util |
|---|---|---|---|---|
| 512 | 3,96 GB | 20,0k | 0,65 | 53% |
| 1024 | 4,35 GB | 32,7k | 0,75 | 51% |
| 2048 | 5,09 GB | 49,7k | 0,99 | 55% |
| 4096 | 6,63 GB | 64,6k | 1,53 | 58% |

Đo lại trên task gain v0, 20 iteration, **khi máy đang có job khác chiếm ~10 lõi CPU** (job GPU0, nice 10):

| Env | VRAM đỉnh | steps/s | s/iteration | GPU util |
|---|---|---|---|---|
| 2048 | 5,02 GB | 24,2k | 2,05 | 22% |
| 4096 | 6,56 GB | 37,2k | 2,69 | 25% |
| 8192 | 9,35 GB | 55,4k | 3,66 | 27% |
| 16384 | 14,60 GB | 70,4k | 5,69 | 26% |

Cùng 4096 env, thông lượng giảm 64,6k → 37,2k khi CPU bị chia sẻ: pipeline bị giới hạn bởi CPU/Python,
không phải GPU. VRAM tăng ~0,65 GB/1000 env; 16384 env không có lỗi PhysX/NaN.

GPU chưa bão hòa (≤148 W/600 W). 4096 env dùng chưa tới 7 GB; có dư địa thử 8192 hoặc 2 run song song.
PPO từ đầu trong 30–50 iteration học "ngã sớm" (độ dài tập 45→7 bước, 100% kết thúc do nghiêng) vì
reward/bước âm; quá ngắn để kết luận, nhưng là tín hiệu nên khởi tạo từ policy đã biết đi.

## Phát hiện: gain PD của ONNX công khai khác vendor ghim

`deploy.yaml` đi kèm ONNX v0 dùng kp=200 ở waist roll/pitch và kd=10 ở 14 khớp tay; robot config của
revision ghim dùng 40 và 1. Thứ tự khớp, quan sát, action scale/offset trùng khít. Actor ONNX chạy tất
định trong PhysX (64 env, 20 s, không giữ hướng, không nhiễu/đẩy):

| Lệnh (m/s) | Task gain v0 | Task gain vendor |
|---|---|---|
| 0,5 | 100% sống, 0,486 m/s | 31% sống, quay loạn |
| 1,0 | 100%, 0,960 m/s | 0% |
| 1,5 | 100%, 1,373 m/s | 0% |
| 2,0 | 0% | 0% |

Tốc độ PhysX với gain v0 gần MuJoCo (0,48 / 0,95 / 1,39). Do đó thêm task
`Unitree-G1-29dof-Velocity-G1LabV0Gains` ([isaac_tasks.py](../src/g1lab/isaac_tasks.py),
[isaac_v0_cfg.py](../src/g1lab/isaac_v0_cfg.py)) — đăng ký phía project, không sửa vendor — và đặt
làm mặc định của launcher. Test xác nhận gain khớp `deploy.yaml` v0.

## Quyết định khởi tạo: ghép actor ONNX + warm-up critic

ONNX công khai **chính là actor RSL-RL**: tham số `actor.{0,2,4,6}`, MLP 480→512→256→128→29 ELU,
không chuẩn hóa quan sát. Ba lựa chọn trong handoff được thay bằng lựa chọn chính xác hơn:

| Lựa chọn | Đánh giá |
|---|---|
| Full checkpoint chính thức | Không có trong repo/HF đã kiểm; không dùng `.pt` lạ không rõ config |
| BC/distillation | Không cần: chép trọng số cho actor giống hệt, không có sai số chưng cất |
| PPO từ đầu | Khả thi về lâu dài nhưng mất bước đi đã có; thử ngắn cho thấy học ngã sớm |
| **Ghép actor + warm-up critic** | **Chọn.** Actor đúng từng tham số; critic khởi tạo seed, học trước khi cập nhật actor |

Thực hiện: [onnx_to_rslrl.py](../scripts/onnx_to_rslrl.py) tạo checkpoint (critic seed 0, optimizer rỗng,
iter 0); [isaac_train.py](../scripts/isaac_train.py) `--critic_warmup N` đóng băng actor và std trong N
iteration đầu (KL = 0 nên LR adaptive không đổi). **std = 0,1**: thử nhiễu action trên actor ONNX ở PhysX,
std 0,1 giữ 100% sống ở 1,0 và 1,5 m/s; 0,2 rớt còn 50% ở 1,5 m/s; 0,5 ngã hết.

Kiểm chứng (1024 env, 50 warm-up + 50 PPO, cap 1, LR 1e-4, seed 42, 74 s wall): tập đạt giới hạn
1000 bước, 0% kết thúc do nghiêng, reward 0,6 → 52,8. Checkpoint cuối trong PhysX: 1,0 → 0,967 m/s,
1,5 → 1,420 m/s (100% sống). Export [export_rslrl_onnx.py](../scripts/export_rslrl_onnx.py) rồi chấm
MuJoCo với tiêu chí cũ: đạt 1,25 m/s (5/5, 1,192 m/s); ở 1,5 m/s sống 3/3, vận tốc 1,45 nhưng RMS quay
0,294 > 0,25 nên **không đạt**. Đây là kiểm chứng pipeline Isaac → ONNX → MuJoCo, không phải cải tiến.

## Lệnh tái lập

```bash
unset CUDA_VISIBLE_DEVICES   # scripts pin GPU1 by UUID themselves and refuse other values
export OMNI_KIT_ACCEPT_EULA=YES
source ~/TUAN__HUMAN_NOID_G1_RUNNER/envs/isaac51/bin/activate
USD=~/TUAN__HUMAN_NOID_G1_RUNNER/assets/unitree_model/G1/29dof/usd/g1_29dof_rev_1_0/g1_29dof_rev_1_0.usd
python scripts/onnx_to_rslrl.py --std 0.1 --seed 0 \
  --onnx vendor/unitree_rl_lab/deploy/robots/g1_29dof/config/policy/velocity/v0/exported/policy.onnx \
  --template <smoke_run>/model_0.pt --output <new_dir>/model_0.pt
python scripts/prepare_lab_fast.py --usd "$USD" --cap 1 --envs 1024 --iterations 100 \
  --checkpoint <new_dir>/model_0.pt --critic-warmup 50 --output artifacts/<new_name> --execute
python scripts/isaac_eval_policy.py --policy <run>/model_99.pt --usd "$USD" --speed 1.0 --output <json>
python scripts/export_rslrl_onnx.py --checkpoint <run>/model_99.pt --output artifacts/policies/<name>
.venv-eval/bin/python scripts/benchmark_lab_policy.py --heading-hold --policy-dir artifacts/policies/<name> \
  --output artifacts/<bench_name>
```

## Còn lại

- Chưa training dài; chưa có số đo 2–3 m/s. Lệnh ≥2 m/s hiện ngã ở cả MuJoCo và PhysX.
- Các override dáng đi theo cap (chu kỳ/stance) vẫn là giả thuyết chưa kiểm.
- Isaac Sim 5.1 EOL, driver 595 ngoài danh sách đã test; nếu cần stack được hỗ trợ: port lên Lab 3.x.
- `isaac_eval_policy.py` chưa có vòng giữ hướng và pha dừng; tiêu chí chấp nhận vẫn là benchmark MuJoCo.
- Isaac cảnh báo `Overriding environment ... already in registry` khi import vendor — vô hại.
- Code đợt này chưa commit; `command.json` ghi `project_dirty: true`.
