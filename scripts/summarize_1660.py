"""Publish small experiment summaries; raw trajectories and policy weights remain local."""

import csv
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("artifacts/autonomous-1660")
DOCS = Path("docs")


def summary(name):
    return json.loads((ROOT / name / "summary.json").read_text())


def table(data):
    rows = [
        "| Scenario | Không ngã / 3 | Thời gian (s) | RMSE vận tốc (m/s) | RMSE quay (rad/s) |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, item in data.items():
        rows.append(
            f"| {name} | {round(item['survival_rate'] * item['episodes'])}/{item['episodes']} | "
            f"{item['duration_s']['mean']:.1f} | {item['velocity_rmse_mps']['mean']:.3f} | "
            f"{item['yaw_rmse_radps']['mean']:.3f} |"
        )
    return rows


def main():
    destination = DOCS / "experiment-data"
    destination.mkdir(exist_ok=True)
    for folder in ROOT.iterdir():
        if folder.is_dir() and (folder / "summary.json").exists():
            target = destination / folder.name
            target.mkdir(exist_ok=True)
            for name in ("summary.json", "episodes.csv", "protocol.json", "environments.json"):
                if (folder / name).exists():
                    shutil.copyfile(folder / name, target / name)
    training = {}
    for path in Path("runs/autonomous-1660").rglob("training.json"):
        training[str(path.parent)] = json.loads(path.read_text())
    (destination / "training.json").write_text(json.dumps(training, indent=2))
    for name, folder in (
        ("ppo-plain", "ppo-campaign"),
        ("ppo-anchored", "ppo-anchored"),
        ("dagger", "dagger-42"),
        ("dagger-slope", "dagger-slope"),
    ):
        path = Path("runs/autonomous-1660") / folder / "progress.json"
        if path.exists():
            shutil.copyfile(path, destination / f"{name}.json")
    manifest = {}
    for folder in ("selected", "selected-terrain"):
        bundle = Path("runs/autonomous-1660") / folder
        manifest[folder] = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in bundle.iterdir()
            if path.is_file()
        }
    (destination / "checkpoint-sha256.json").write_text(json.dumps(manifest, indent=2))
    before = summary("dagger-walk-20000")["configured"]
    after = summary("final-walk-20000")["configured"]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), layout="constrained")
    for ax, key, title in zip(
        axes,
        ["velocity_rmse_mps", "yaw_rmse_radps", "mean_power_w"],
        ["Velocity RMSE (m/s)", "Yaw RMSE (rad/s)", "Mechanical power proxy (W)"],
    ):
        ax.bar(
            ["DAgger", "PPO + BC"],
            [before[key]["mean"], after[key]["mean"]],
            yerr=[before[key]["std"], after[key]["std"]],
            color=["#3a7188", "#d39c49"],
        )
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
    fig.savefig(DOCS / "images" / "paired-walk.png", dpi=160)
    plt.close(fig)
    episodes = list(csv.DictReader((ROOT / "final-walk-20000" / "episodes.csv").open()))
    distance = sum(float(r["x_progress_m"]) for r in episodes) / len(episodes)
    text = [
        "# Kết quả thực nghiệm trên máy 1660 Ti — 2026-11-11",
        "",
        "Đã thực hiện toàn bộ chuỗi kiểm tra mô phỏng, học đứng, học đi, điều khiển, chống nhiễu, "
        "dốc và rough terrain. Các bài không đạt được giữ nguyên trong báo cáo.",
        "",
        "## Nguồn policy và phạm vi",
        "",
        "Policy sử dụng được hiện tại có nguồn **Unitree pretrained → imitation → DAgger → PPO + BC anchoring**. "
        "Không phải kết quả PPO tự học locomotion từ đầu. G1 có 29 khớp vật lý; policy xuất 29 action, "
        "17 action thân trên được mask về 0, 12 khớp chân điều khiển chuyển động. Thân trên giữ bằng PD. "
        "Phase 0.8 s được cung cấp cho actor. Chỉ chạy trong MuJoCo.",
        "",
        "Nguồn teacher: https://github.com/unitreerobotics/unitree_rl_gym tại commit "
        "`276801e46c5d433564f24658bac64f254b7d2d4b`. Policy/models upstream nằm trong `vendor/` không được commit.",
        "",
        "## Máy và quá trình chạy",
        "",
        "Ryzen 5 4600H, RAM 16 GB, GTX 1660 Ti Mobile 6 GB, Linux, Python 3.10, MuJoCo 3.3.7, "
        "PyTorch 2.6 CPU, SB3 2.7. Physics và PPO dùng CPU; EGL render xác nhận GPU NVIDIA 1660 Ti. "
        "GUI đã mở trực tiếp, tự chạy chu kỳ đứng–đi–quay nhẹ–dừng.",
        "",
        "- Baseline PD và policy đứng: 3/3 episode đủ 30 s, seed 10000–10002.",
        "- PPO đứng: 32,768 bước. Pilot đi chậm dừng sau 48,796 bước thu thập vì đứng tại chỗ.",
        "- Pilot PPO thêm vận tốc thân và reward chặt hơn: 131,072 bước; vẫn ngã khi cố tiến lên.",
        "- Imitation: 16,000 mẫu teacher trên mô hình G1 của dự án. MSE nhỏ chưa bảo đảm rollout ổn định.",
        "- DAgger: 10 vòng × 2,048 mẫu, chọn round 7 từ validation seed 5000 trước đánh giá 30 s.",
        "- PPO thuần fine-tune: 4 giai đoạn × 16,384 bước, đều bị guard loại do mất khả năng đi thẳng.",
        "- PPO + BC: 4 giai đoạn × 8,192 bước, có warm-up critic 4,096 bước/giai đoạn; đều qua guard đi thẳng.",
        "- Bổ sung DAgger cho dốc 2°: 8 vòng × 1,024 mẫu, checkpoint round 4 được kiểm tra riêng.",
        "",
        "Guard chỉ đo đi thẳng 10 s trên seed 5000, không chứng nhận khả năng đi terrain hay mọi lệnh. "
        "Các run là thí nghiệm phát triển với nhiều cấu hình thay đổi, không phải so sánh thuật toán công bằng.",
        "",
        "## Kết quả đi thẳng có ghép cặp",
        "",
        "Lệnh 0.3 m/s, 30 s, cùng seed 20000–20002. Cả hai checkpoint sống sót 3/3.",
        "",
        f"Checkpoint PPO + BC tiến trung bình {distance:.2f} m; {episodes[0]['validated_footsteps']} lần "
        "nhấc–đặt chân hợp lệ trong episode đầu. Vận tốc dao động theo chu kỳ bước, không phải trượt đứng yên.",
        "",
        "![Paired walking metrics](images/paired-walk.png)",
        "",
        "Trong bộ test nhỏ này, PPO + BC giảm sai số quay và công suất cơ học proxy, nhưng sai số vận tốc "
        "tăng so với DAgger. Không kết luận cải thiện toàn diện. Error bar là SD qua episode, không phải CI qua seed train.",
        "",
        "## Benchmark checkpoint PPO + BC",
        "",
        "`runs/autonomous-1660/selected`, seed 20000–20002, mỗi bài 30 s. Forward/full-suite dùng 0.4 m/s; "
        "khác bài đi 0.3 m/s ở trên. Dốc 0.2/0.5 tương ứng 2°/5°; rough 0.2/0.5 tối đa 1.2/3 cm.",
        "",
        *table(summary("final-20000")),
        "",
        "Lực đẩy áp vào pelvis theo +X/+Y thế giới tại giây thứ 5 trong 0.1 s; báo xung lực N·s. "
        "Không suy ra ngưỡng lực đẩy tối đa từ ba mức 2/5/10 đã thử.",
        "",
        "## Checkpoint bổ sung cho dốc nhẹ",
        "",
        "`runs/autonomous-1660/dagger-slope/round-4`, lệnh 0.3 m/s, seed 30000–30002, 30 s.",
        "",
        "Dốc 2°:",
        "",
        *table(summary("slope-specialist-30000")),
        "",
        "Kiểm tra lại sàn phẳng cùng checkpoint:",
        "",
        *table(summary("slope-flat-30000")),
        "",
    ]
    if (ROOT / "terrain-final-30000" / "summary.json").exists():
        text += [
            "### Benchmark mở rộng checkpoint có dốc",
            "",
            "Đây là checkpoint khác bảng PPO + BC ở trên. Dùng full suite, seed 30000–30002, 30 s.",
            "",
            *table(summary("terrain-final-30000")),
            "",
        ]
    recovery_path = ROOT / "recovery-verified" / "episodes.csv"
    if recovery_path.exists():
        recovery_rows = list(csv.DictReader(recovery_path.open()))
        times = [float(r["recovery_time_s"]) for r in recovery_rows if r["recovery_time_s"]]
        if times:
            text += [
                "### Phục hồi đo lại bằng schema đã sửa",
                "",
                f"Đẩy ngang 10 N·s trên checkpoint terrain: {len(times)}/3 lần có phục hồi đo được, "
                f"trung bình {sum(times) / len(times):.2f} s sau khi xung lực kết thúc. "
                "Điều kiện: sai số vận tốc <0.15 m/s và upright vật lý >0.9 liên tục ít nhất 0.5 s. "
                "Xem `experiment-data/recovery-verified/episodes.csv`.",
                "",
            ]
    text += [
        "## Sửa lỗi chỉ số tư thế",
        "",
        "Trong log đầu chiến dịch, key `upright` bị reward cùng tên ghi đè thành 0.5 × giá trị tư thế thật. "
        "Đã sửa code để reward dùng tiền tố `reward_` và thêm kiểm thử. Dynamics, reward tổng, survival, "
        "vận tốc và năng lượng không bị đổi. Biểu đồ cũ dựng lại upright vật lý bằng 2 × giá trị log; raw CSV "
        "giữ nguyên. `recovery_time_s` trong episode CSV cũ không dùng để kết luận. "
        "Bài `recovery-verified` được chạy lại bằng schema đã sửa.",
        "",
        "## Giới hạn và cách xem lại",
        "",
        "- Chỉ một seed huấn luyện chính (42); ba seed đánh giá chủ yếu thay đổi tư thế ban đầu và rough map. "
        "Chưa có kết luận thống kê qua nhiều policy train độc lập.",
        "- Bộ seed 10000+ đã dùng khi phát triển; seed 20000+ và 30000+ dùng sau chọn checkpoint tương ứng.",
        "- Đi ngang, quay tại chỗ, dốc lớn và cầu thang phải xem tỷ lệ thất bại trong bảng; không được "
        "coi mọi năng lực đã đạt chỉ vì demo đi thẳng thành công.",
        "- Công suất là proxy cơ học từ actuator, không phải điện năng tiêu thụ robot hay GPU.",
        "- Terrain scan là hình học chính xác từ simulator. Không có camera perception, AMP hay robot thật.",
        "- Kiểm thử code: 14 pytest đạt, ruff đạt; GUI được mở và trạng thái mô phỏng được theo dõi qua log. "
        "Phiên computer-use không cung cấp quan sát cửa sổ native, nên kiểm tra hình ảnh dùng render/video.",
        "",
        "```bash",
        "source .venv/bin/activate",
        "g1lab demo --bundle runs/autonomous-1660/selected-terrain --auto",
        "g1lab demo --bundle runs/autonomous-1660/selected --terrain rough --difficulty 0.2 --auto",
        "g1lab demo --bundle runs/autonomous-1660/selected-terrain --terrain slope --difficulty 0.2 --auto",
        "```",
        "",
        "Video local: `artifacts/autonomous-1660/final-video/configured-20000.mp4` và "
        "`artifacts/autonomous-1660/slope-video/configured-30000.mp4`. Raw trajectory CSV, log và checkpoint "
        "giữ tại máy. Các summary/protocol/episode CSV nhỏ được lưu ở [experiment-data](experiment-data/). "
        "SHA256 checkpoint nằm trong `experiment-data/checkpoint-sha256.json`. Các run được thực hiện "
        "trong working tree phát triển; metadata git_revision cũ không tự mô tả mọi thay đổi chưa commit lúc đó.",
        "",
    ]
    (DOCS / "EXPERIMENT_1660.md").write_text("\n".join(text))
    print(DOCS / "EXPERIMENT_1660.md")


if __name__ == "__main__":
    main()
