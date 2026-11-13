# G1 Robust Locomotion Lab

**Dự án RL cho humanoid Unitree G1 29 khớp, chạy hoàn toàn trong mô phỏng.**
Hướng hiện tại: dạy G1 **chạy bứt tốc có pha bay**, giữ thăng bằng, giảm tốc và dừng được.

Isaac Sim 5.1 / Isaac Lab 2.3.2 (PhysX, train) • RSL-RL PPO • MuJoCo (đánh giá độc lập) • RTX 5090

## Trạng thái hiện tại (13/11/2026)

| Hạng mục | Trạng thái |
|---|---|
| Baseline tốt nhất đã kiểm chứng | Policy ONNX công khai của Unitree + vòng giữ hướng: **1,39 m/s**, 10/10 seed giữ 60 s rồi dừng (MuJoCo). Dáng **đi bộ**: 0% pha bay, tiếp đất bằng gót. Từ 2 m/s ngã. |
| Tái hiện trên RTX 5090 | Khớp kết quả máy cũ tới 1e-15 (46/46 lượt), `episodes.csv` giống từng byte |
| Pipeline train Isaac | Chạy được: smoke, resume, 512–16384 env, export ONNX → chấm MuJoCo |
| Khởi tạo PPO | Ghép **chính xác** actor ONNX vào checkpoint RSL-RL + warm-up critic (ONNX chính là actor RSL-RL) |
| Task Sprint | Đã dựng và kiểm chứng ở mức smoke: 12 term reward kỹ thuật chạy hoạt động, đúng dấu |
| Autopilot | Vòng lặp train → đánh giá → giữ bản tốt nhất → tự xử lý → báo cáo, chạy không cần mạng; một lượt chạy không giới hạn đã khởi động 13/11/2026 |
| **Chưa có** | Policy chạy nhanh hơn baseline đã kiểm chứng. Kết quả autopilot (checkpoint, báo cáo) nằm trong `artifacts/` trên máy train, không có trong Git |

Mục tiêu gần: 2–3 m/s thực tế, giữ 60 s, có pha bay, giảm tốc và dừng, kiểm chứng trên 10 seed mới.
11 m/s chỉ là trần của curriculum để nghiên cứu, **không phải cam kết**: bài báo SPRINT công bố khoảng
6 m/s trên G1. Giới hạn mô-men, giới hạn vận tốc khớp, gain PD và mô hình robot không bị nới để làm đẹp
kết quả.

## Luồng làm việc

```text
Isaac Lab (PhysX, GPU1)              export                 MuJoCo (CPU, độc lập)
task Sprint, PPO 4096 env  ──►  policy.onnx + deploy.yaml  ──►  thang tốc độ, 10 seed mới,
đánh giá thang tốc độ 1→11 m/s                                 giữ 60 s, dừng, số đo pha bay
```

Isaac là simulator để **train**; MuJoCo là simulator **chấm kết quả**. Hai simulator không được mặc định
cho cùng kết quả, nên mọi con số đều ghi rõ đo bằng simulator nào.

## Phát hiện quan trọng

- **Gain PD.** `deploy.yaml` đi kèm ONNX v0 dùng kp = 200 ở eo roll/pitch và kd = 10 ở 14 khớp tay; robot
  config của revision Unitree đang ghim dùng 40 và 1. Với gain v0, actor ONNX đi được trong PhysX (100% sống
  ở 1,5 m/s, 1,37 m/s, gần MuJoCo); với gain vendor nó ngã hết từ 1 m/s. Mọi task của dự án dùng gain v0
  (`Unitree-G1-29dof-Velocity-G1LabV0Gains`), đăng ký phía project, không sửa vendor.
- **Resume.** `train.py` của vendor bỏ qua `--experiment_name`; launcher dùng Hydra override
  `agent.experiment_name` để resume tìm đúng checkpoint.
- **Lực ngang.** Cảm biến tiếp xúc Isaac chỉ cho lực pháp tuyến; lực đẩy ngang được tính từ Σ mᵢaᵢ.
- **Chạy offline.** Mặt đất "plane" mặc định tải từ cloud NVIDIA; task Sprint dùng mặt đất sinh tại chỗ.

## Task Sprint

`Unitree-G1-29dof-Sprint-G1Lab` ([src/g1lab/isaac_sprint.py](src/g1lab/isaac_sprint.py)) giữ nguyên quan sát
và action (480 → 29), nên vẫn ghép được actor ONNX.

- **Cổng tốc độ:** dưới 1,6 m/s, reward giống task đi bộ mà ONNX được train. Từ 1,6 đến 2,2 m/s, reward
  chuyển dần sang bộ reward chạy nước rút.
- **Reward kỹ thuật chạy:** pha bay; phạt hai chân cùng chạm; chạm đất ngắn (trần thời gian giảm theo tốc
  độ); thời gian bay của chân; lực đẩy ngang; tiếp đất bằng mũi chân; nâng gối; gập cổ chân khi vung; tay
  vung ngược pha chân cùng bên; đổ người theo tốc độ **và** gia tốc.
- **Pha bước không dùng đồng hồ:** cảm biến bàn chân ngưỡng 20 N, khử nhiễu 40 ms.
- **Lệnh tốc độ có giới hạn gia tốc:** mỗi tập xuất phát từ đứng yên, có lệnh giảm tốc và dừng. Curriculum
  mở dải tốc độ tới trần 3,5 m/s; autopilot nâng trần lên 11 m/s.

Chi tiết trọng số và số liệu kiểm chứng: [docs/SPRINT_TASK.md](docs/SPRINT_TASK.md).

## Cài đặt

Dùng **hai môi trường riêng**; không cài Torch CPU vào môi trường train.

**A. Đánh giá MuJoCo** (Python 3.10, Torch CPU là cố ý):

```bash
python3.10 -m venv .venv-eval
.venv-eval/bin/python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
.venv-eval/bin/python -m pip install -e '.[dev,lab]' -c requirements-lock.txt
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv-eval/bin/python -m pytest -q
```

**B. Train Isaac** (Python 3.11): Isaac Sim 5.1.0 (pip), Torch 2.7.0+cu128, Isaac Lab v2.3.2 (source),
RSL-RL 3.1.2, `vendor/unitree_rl_lab` (editable), project cài `--no-deps`. Luôn cài với
[file constraints](docs/experiment-data/5090-bringup/isaac51-constraints.txt): `isaaclab.sh --install` dùng
`uv` và có thể tự nâng Torch lên bản CUDA 13. Từng bước và lý do: [docs/RTX_5090_BRINGUP.md](docs/RTX_5090_BRINGUP.md).

Asset USD G1 29 khớp lấy từ [unitreerobotics/unitree_model](https://huggingface.co/datasets/unitreerobotics/unitree_model),
giữ nguyên cấu trúc thư mục; đường dẫn truyền qua `--usd`.

### Chỉ dùng một GPU

[src/g1lab/gpu.py](src/g1lab/gpu.py) ghim CUDA theo **UUID** của GPU dành cho dự án, đặt
`CUDA_DEVICE_ORDER=PCI_BUS_ID`, và **từ chối** mọi `CUDA_VISIBLE_DEVICES` khác. Render MuJoCo/EGL cũng ghim
theo địa chỉ PCI. Kit chạy với multi-GPU tắt. UUID, PCI và `activeGpu` hiện đặt cho máy phát triển; trên máy
khác, đặt `G1LAB_GPU_UUID`/`G1LAB_GPU_PCI` và kiểm tra lại `KIT_ARGS`. Đừng tự đặt `CUDA_VISIBLE_DEVICES`.

## Chạy

Mỗi thư mục output phải mới. `python` là interpreter Isaac; `$USD` là đường dẫn
`.../G1/29dof/usd/g1_29dof_rev_1_0/g1_29dof_rev_1_0.usd`.

```bash
# Tái hiện baseline (MuJoCo)
.venv-eval/bin/python scripts/tune_lab_controller.py --output artifacts/baseline-check
.venv-eval/bin/python scripts/benchmark_lab_policy.py --heading-hold --output artifacts/heading-check

# Preflight và smoke Isaac
python scripts/check_training_host.py --usd "$USD" --output artifacts/host.json --require-training
python scripts/prepare_lab_fast.py --usd "$USD" --cap 1 --envs 32 --iterations 2 --output artifacts/smoke --execute

# Checkpoint khởi tạo từ actor ONNX (template = checkpoint bất kỳ cùng kiến trúc, ví dụ từ smoke)
python scripts/onnx_to_rslrl.py --std 0.1 --seed 0 \
  --onnx vendor/unitree_rl_lab/deploy/robots/g1_29dof/config/policy/velocity/v0/exported/policy.onnx \
  --template <smoke_run>/model_0.pt --output checkpoints/onnx-v0-std0.1/model_0.pt

# Một lượt train task Sprint
python scripts/prepare_lab_fast.py --profile sprint --usd "$USD" --envs 4096 --iterations 600 \
  --checkpoint checkpoints/onnx-v0-std0.1/model_0.pt --critic-warmup 20 --output artifacts/sprint-01 --execute

# Đánh giá: thang tốc độ Isaac, export, thang tốc độ MuJoCo kèm số đo pha bay
python scripts/isaac_sprint_eval.py --policy <run>/model_N.pt --usd "$USD" --speeds 1,2,3,4,5 --output <json>
python scripts/export_rslrl_onnx.py --checkpoint <run>/model_N.pt --output artifacts/policies/<name>
.venv-eval/bin/python scripts/mujoco_sprint_eval.py --policy-dir artifacts/policies/<name> --speeds 1,1.5,2 --output <json>
```

### Autopilot (train không người trông, không cần mạng)

```bash
setsid nohup .venv-eval/bin/python scripts/autopilot_sprint.py --output artifacts/autopilot-<tên> \
  --init-checkpoint checkpoints/onnx-v0-std0.1/model_0.pt --usd "$USD" --hours 0 \
  --envs 4096 --round-iterations 600 --speed-limit 11.0 > artifacts/autopilot-<tên>.out 2>&1 &

cat artifacts/autopilot-<tên>/REPORT.md                                # tiến độ, bản tốt nhất, kiểm chứng
echo '{"stop": true}' > artifacts/autopilot-<tên>/control.json        # dừng êm sau vòng hiện tại
```

`--hours 0` nghĩa là chạy đến khi có lệnh dừng. Mỗi vòng: train 600 iteration → đánh giá Isaac → export →
MuJoCo chạy nền. Quy tắc tự xử lý:
- Lưu bản tốt nhất theo tốc độ giữ được trong Isaac.
- Tụt hơn 1 m/s so với bản tốt nhất, hoặc NaN, hoặc train lỗi → quay về bản tốt nhất và giảm LR một nửa.
- Chững 3 vòng → bật nút chỉnh kế tiếp.
- Nếu curriculum không tự tăng mà policy đã giữ được tốc độ gần đỉnh dải → tự nâng thêm 0,5 m/s.

Mỗi bản tốt nhất mới được kiểm chứng MuJoCo 10 seed mới, giữ 60 s. Mọi tiến trình con chạy trong
`unshare -rn` (không mạng). Một luồng canh sẽ dừng toàn bộ nếu tiến trình của dự án xuất hiện trên GPU khác.
Tự dừng khi ổ đĩa còn dưới 50 GB. Chi tiết: [docs/AUTOPILOT.md](docs/AUTOPILOT.md).

## Tiêu chí đánh giá

| Simulator | Đạt |
|---|---|
| Isaac (thang tốc độ) | Xuất phát từ đứng yên, tăng tốc 2 m/s², giữ 10 s: ≥ 90% robot sống và vx trung bình ≥ 85% lệnh |
| MuJoCo (chấm cuối) | Đứng → tăng tốc → giữ 60 s → giảm tốc → dừng: không ngã, vx ≥ 85% lệnh, RMSE phẳng ≤ max(0,15; 20% lệnh), RMS quay < 0,25 rad/s, lệch hướng < 30°, tốc độ cuối < 0,15 m/s |

Số đo kỹ thuật chạy được báo cáo kèm (MuJoCo 500 Hz, ngưỡng 20 N, khử nhiễu 40 ms): tỷ lệ pha bay, hai chân
cùng chạm, thời gian chạm/vung, cadence, tỷ lệ tiếp đất mũi/gót, lực ngang đỉnh, độ nâng gối, độ gập cổ chân,
biên độ và tương quan tay–chân, góc ngả thân. Ngưỡng bắt buộc cho pha bay chưa được chốt.

## Cấu trúc

```text
src/g1lab/isaac_sprint.py     Task Sprint: lệnh tốc độ giới hạn gia tốc, theo dõi pha bước, reward, terrain offline
src/g1lab/isaac_v0_cfg.py     Task velocity với gain PD của ONNX v0
src/g1lab/isaac_tasks.py      Đăng ký task (entry point dạng chuỗi, không cần Isaac khi import)
src/g1lab/gpu.py              Ghim GPU (CUDA theo UUID, EGL theo PCI, tham số Kit)
src/g1lab/fast_plan.py        Lệnh train Isaac (speed-stage, sprint)
src/g1lab/gait_metrics.py     Số đo kỹ thuật chạy (thuần numpy)
src/g1lab/lab_policy.py       Chạy policy ONNX 29 khớp trong MuJoCo (+ telemetry chỉ đọc)
src/g1lab/lab_evaluation.py   Kịch bản đứng/tăng tốc/giữ/giảm tốc/dừng và tiêu chí đạt
scripts/autopilot_sprint.py   Vòng lặp train/đánh giá/tự xử lý không người trông
scripts/isaac_train.py        Wrapper train.py của vendor: đăng ký task, warm-up critic
scripts/prepare_lab_fast.py   Launcher train Isaac + provenance
scripts/onnx_to_rslrl.py      ONNX → checkpoint RSL-RL
scripts/export_rslrl_onnx.py  Checkpoint → policy.onnx + deploy.yaml
scripts/isaac_sprint_eval.py  Thang tốc độ trong Isaac
scripts/mujoco_sprint_eval.py Thang tốc độ trong MuJoCo kèm số đo pha bay
scripts/isaac_sprint_probe.py Kiểm tra dấu/độ lớn các term reward
vendor/                       unitree_rl_lab, unitree_mujoco (revision ghim)
docs/experiment-data/         Số liệu nhỏ đi kèm các báo cáo
legacy/, src/g1lab/env.py ... Pipeline MuJoCo/SB3 thời 1660 (xem bên dưới)
```

## Lịch sử: pipeline MuJoCo/SB3 trên GTX 1660 Ti

Trước RTX 5090, dự án chạy trên máy GTX 1660 Ti 6 GB với môi trường MuJoCo CPU riêng và PPO
Stable-Baselines3 (CLI `g1lab`: train/eval/curriculum/report/demo, địa hình phẳng/dốc/gồ ghề/cầu thang).
Pipeline này vẫn còn trong repo và test vẫn chạy, nhưng không còn là hướng chính. Checkpoint SB3 dùng
teacher chân 12 khớp, không resume được bằng RSL-RL.

- [Báo cáo dọn mã và 46 lượt thử bộ giữ hướng](docs/CLEANUP_1660_REPORT.md)
- [Tích hợp Unitree RL Lab 29 khớp](docs/UNITREE_RL_LAB.md) · [Đo tốc độ policy Unitree gốc](docs/UNITREE_SPEED.md)
- [Thực nghiệm 1660](docs/EXPERIMENT_1660.md) · [trộn địa hình](docs/EXPERIMENT_MIXED_1660.md) ·
  [giữ kỹ năng bằng replay](docs/EXPERIMENT_PRESERVATION_1660.md) · [chạy tiếp tại máy](docs/LOCAL_TRAINING.md)
- [Giao thức đánh giá](docs/EVALUATION.md) · [Demo 16 môi trường](docs/SWARM_DEMO.md)
- [Pilot tự đứng dậy](docs/RECOVERY.md) — tạm dừng, không nằm trong phạm vi hiện tại

## Tài liệu

- [Bring-up RTX 5090](docs/RTX_5090_BRINGUP.md): môi trường, số đo quy mô env, gain v0, khởi tạo
- [Task Sprint](docs/SPRINT_TASK.md) · [Autopilot](docs/AUTOPILOT.md)
- [Đánh giá SPRINT (arXiv 2605.28549) và kế hoạch tích hợp](docs/SPRINT_INTEGRATION.md): chưa có code/data công khai
- [Bàn giao 5090](docs/CLAUDE_5090_START_HERE.md) · [Trạng thái kiểm chứng](docs/VALIDATION.md)
- Provenance và giấy phép: [THIRD_PARTY.md](THIRD_PARTY.md)

Dự án chỉ chạy trong mô phỏng; không có mã điều khiển robot thật. Smoke test hoặc lượt train thành công
không có nghĩa là policy đã học chạy. Chỉ benchmark MuJoCo trên seed mới mới được dùng để kết luận.
