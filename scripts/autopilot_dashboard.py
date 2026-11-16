"""Offline dashboard for an autopilot_sprint.py run: a self-contained HTML page and/or a terminal summary.

Standard library only, no network, no GPU. Reads <run>/state.json, the current round's train.log and nvidia-smi.

  python3 scripts/autopilot_dashboard.py --run artifacts/autopilot-v2-scratch-20261116            # write dashboard.html
  python3 scripts/autopilot_dashboard.py --run artifacts/autopilot-v2-scratch-20261116 --watch 60 # keep it updated
  python3 scripts/autopilot_dashboard.py --run artifacts/autopilot-v2-scratch-20261116 --terminal # text summary
  python3 scripts/autopilot_dashboard.py --run ... --terminal --watch 60                          # live text view

Open <run>/dashboard.html in a browser; the page reloads itself every minute.
"""

import argparse
import html
import json
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

GPU1_UUID = "GPU-20e5a706-6231-50a3-4c9b-8e8d4db3c470"


# --------------------------------------------------------------------------------------------- data
def autopilot_pid(run):
    name = Path(run).name
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            cmd = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="ignore")
        except OSError:
            continue
        if "autopilot_sprint.py" in cmd and name in cmd and "python" in cmd.split(" ")[0]:
            return int(proc.name)
    return None


def gpu_status():
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                GPU1_UUID,
                "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        used, total, util, temp = [x.strip() for x in out.split(",")]
        return {
            "mem_used_gb": int(used) / 1024,
            "mem_total_gb": int(total) / 1024,
            "util": int(util),
            "temp": int(temp),
        }
    except Exception:
        return None


def current_training(run, rounds_done):
    dirs = sorted(Path(run).glob("round_*"))
    if not dirs:
        return None
    log = dirs[-1] / "train" / "train.log"
    info = {"round": int(dirs[-1].name.split("_")[1]), "phase": "khởi động"}
    if not log.exists():
        return info
    text = log.read_text(errors="ignore")
    its = re.findall(r"Learning iteration (\d+)/(\d+)", text)
    series = {
        key: [float(x) for x in re.findall(re.escape(label) + r": (-?[\d.]+)", text)]
        for key, label in [
            ("reward", "Mean reward"),
            ("length", "Mean episode length"),
            ("level", "Curriculum/lin_vel_cmd_levels"),
            ("falls", "Episode_Termination/bad_orientation"),
        ]
    }
    info.update(series)
    if its:
        first = int(its[0][0])
        cur, total = int(its[-1][0]), int(its[-1][1])
        info.update({"iteration": cur - first + 1, "iterations": total - first, "phase": "đang train"})
    if info["round"] <= rounds_done:
        info["phase"] = "đánh giá / export"
    return info


def load(run):
    state = json.loads((Path(run) / "state.json").read_text())
    rounds = state.get("rounds", [])
    return {
        "state": state,
        "rounds": rounds,
        "pid": autopilot_pid(run),
        "gpu": gpu_status(),
        "current": current_training(run, len(rounds)),
        "generated": datetime.now().isoformat(timespec="seconds"),
        "run": str(Path(run).resolve()),
    }


def per_round(rounds):
    rows = []
    for r in rounds:
        ev = r.get("isaac_eval") or {}
        cap = ev.get("max_speed_ignoring_yaw_mps")
        strict = ev.get("max_sustained_speed_mps")
        group = next((g for g in ev.get("groups", []) if g["speed_cmd_mps"] == cap), {})
        rows.append(
            {
                "round": r["round"],
                "minutes": r.get("train_minutes"),
                "level_in": r.get("level_in"),
                "level_out": r.get("level_out", r.get("level_in")),
                "lr": r.get("lr"),
                "strict": strict,
                "capability": cap,
                "yaw": group.get("yaw_rate_rms"),
                "flight": group.get("flight_fraction"),
                "vx": group.get("mean_vx_mps"),
                "mujoco": r.get("mujoco_max_passed_mps"),
                "ok": (r.get("train") or {}).get("returncode") == 0,
            }
        )
    return rows


# ------------------------------------------------------------------------------------------- charts
PALETTE = ["#2f6fdf", "#e0782f", "#2a9d6a", "#b8459c"]


def svg_chart(title, series, y_label="", width=560, height=230, y_min=None):
    """series: list of (name, [(x, y or None), ...]). Plain inline SVG line chart."""
    points = [(x, y) for _, data in series for x, y in data if y is not None]
    if not points:
        return f'<figure class="chart"><figcaption>{html.escape(title)}</figcaption><p class="muted">Chưa có dữ liệu</p></figure>'
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x0, x1 = min(xs), max(xs)
    y0 = min(ys) if y_min is None else min(y_min, min(ys))
    y1 = max(ys)
    if x1 == x0:
        x1 = x0 + 1
    if y1 == y0:
        y1 = y0 + 1
    pad_l, pad_r, pad_t, pad_b = 46, 12, 14, 30

    def sx(x):
        return pad_l + (x - x0) / (x1 - x0) * (width - pad_l - pad_r)

    def sy(y):
        return height - pad_b - (y - y0) / (y1 - y0) * (height - pad_t - pad_b)

    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">']
    for i in range(5):
        y = y0 + (y1 - y0) * i / 4
        parts.append(
            f'<line class="grid" x1="{pad_l}" x2="{width - pad_r}" y1="{sy(y):.1f}" y2="{sy(y):.1f}"/>'
        )
        parts.append(
            f'<text class="tick" x="{pad_l - 6}" y="{sy(y) + 4:.1f}" text-anchor="end">{y:.3g}</text>'
        )
    for x in sorted(set(xs))[:: max(1, len(set(xs)) // 10)]:
        parts.append(
            f'<text class="tick" x="{sx(x):.1f}" y="{height - 10}" text-anchor="middle">{x:g}</text>'
        )
    for k, (name, data) in enumerate(series):
        color = PALETTE[k % len(PALETTE)]
        segment = []
        for x, y in data + [(None, None)]:
            if y is None:
                if len(segment) > 1:
                    pts = " ".join(f"{sx(a):.1f},{sy(b):.1f}" for a, b in segment)
                    parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{pts}"/>')
                segment = []
            else:
                segment.append((x, y))
        if len(data) <= 60:
            for x, y in data:
                if y is not None:
                    parts.append(
                        f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="3" fill="{color}"><title>{html.escape(name)}: {y:.3g} (x={x:g})</title></circle>'
                    )
    parts.append("</svg>")
    legend = "".join(
        f'<span><i style="background:{PALETTE[k % len(PALETTE)]}"></i>{html.escape(name)}</span>'
        for k, (name, _) in enumerate(series)
    )
    return (
        f'<figure class="chart"><figcaption>{html.escape(title)}'
        f'<span class="muted"> {html.escape(y_label)}</span></figcaption>{"".join(parts)}'
        f'<div class="legend">{legend}</div></figure>'
    )


# --------------------------------------------------------------------------------------------- html
def fmt(v, digits=2, unit=""):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{digits}f}{unit}"
    return f"{v}{unit}"


def render_html(d, refresh):
    s, rows, cur = d["state"], per_round(d["rounds"]), d["current"] or {}
    running = d["pid"] is not None
    best_i, best_c, best_m = (
        s.get("best_isaac") or {},
        s.get("best_capability") or {},
        s.get("best_mujoco") or {},
    )
    validations = s.get("validations", [])
    val_best_m = best_m.get("validation_10seeds_60s_max_passed_mps")
    gpu = d["gpu"]
    status = (
        f'<span class="pill ok">ĐANG CHẠY · PID {d["pid"]}</span>'
        if running
        else '<span class="pill bad">KHÔNG CHẠY</span>'
    )
    if s.get("finished"):
        status = f'<span class="pill">ĐÃ KẾT THÚC {html.escape(s["finished"])}</span>'
    progress = ""
    if cur.get("iterations"):
        pct = 100 * cur["iteration"] / cur["iterations"]
        progress = (
            f'<div class="bar"><div style="width:{pct:.1f}%"></div></div>'
            f'<div class="muted">Vòng {cur["round"]}: {cur["phase"]}, iteration {cur["iteration"]}/{cur["iterations"]}</div>'
        )
    elif cur:
        progress = f'<div class="muted">Vòng {cur["round"]}: {cur["phase"]}</div>'

    cards = [
        (
            "MuJoCo, kiểm chứng 10 seed × 60 s",
            fmt(val_best_m, 1, " m/s"),
            f"bản vòng {best_m.get('round', '—')} · chấm nhanh 2 seed: {fmt(best_m.get('speed'), 1, ' m/s')}",
        ),
        (
            "Isaac, đạt đủ tiêu chí",
            fmt(best_i.get("speed"), 1, " m/s"),
            f"vòng {best_i.get('round', '—')} · thực {fmt(best_i.get('vx'))} m/s · bay {fmt((best_i.get('flight_fraction') or 0) * 100, 0, '%')}",
        ),
        (
            "Isaac, chỉ xét tốc độ",
            fmt(best_c.get("speed"), 1, " m/s"),
            f"vòng {best_c.get('round', '—')} · RMS quay {fmt(best_c.get('yaw_rate_rms'))} rad/s",
        ),
        (
            "Curriculum / LR",
            fmt(s.get("level"), 1, " m/s"),
            f"LR {s.get('lr', 0):.1e} · nút chỉnh: {len(s.get('knobs_applied', []))}",
        ),
    ]
    cards_html = "".join(
        f'<div class="card"><div class="label">{html.escape(a)}</div><div class="value">{b}</div><div class="muted">{html.escape(c)}</div></div>'
        for a, b, c in cards
    )
    x = [r["round"] for r in rows]
    charts = [
        svg_chart(
            "Tốc độ giữ được theo vòng",
            [
                ("Isaac đủ tiêu chí", list(zip(x, [r["strict"] for r in rows]))),
                ("Isaac chỉ xét tốc độ", list(zip(x, [r["capability"] for r in rows]))),
                ("MuJoCo (2 seed)", list(zip(x, [r["mujoco"] for r in rows]))),
                ("Mức curriculum", list(zip(x, [r["level_out"] for r in rows]))),
            ],
            "m/s",
            y_min=0,
        ),
        svg_chart(
            "Dáng chạy ở mức tốc độ cao nhất (Isaac)",
            [
                ("Tỷ lệ pha bay", list(zip(x, [r["flight"] for r in rows]))),
                ("RMS quay (ngưỡng 0,25)", list(zip(x, [r["yaw"] for r in rows]))),
            ],
            "",
            y_min=0,
        ),
    ]
    if cur.get("length"):
        n = len(cur["length"])
        charts.append(
            svg_chart(
                f"Vòng {cur['round']} đang train: độ dài tập",
                [("bước (tối đa 1000)", list(enumerate(cur["length"])))],
                "theo iteration",
            )
        )
        charts.append(
            svg_chart(
                f"Vòng {cur['round']} đang train: reward / tỷ lệ ngã",
                [
                    ("reward trung bình", list(enumerate(cur.get("reward", [])))),
                    ("tỷ lệ ngã × 100", [(i, v * 100) for i, v in enumerate(cur.get("falls", []))]),
                ],
                f"{n} iteration",
            )
        )
    table_rows = "".join(
        "<tr>"
        + "".join(
            f"<td>{c}</td>"
            for c in [
                r["round"],
                fmt(r["minutes"], 0),
                f"{fmt(r['level_in'], 1)} → {fmt(r['level_out'], 1)}",
                f"{r['lr']:.1e}" if r["lr"] else "—",
                fmt(r["strict"], 1),
                fmt(r["capability"], 1),
                fmt(r["vx"]),
                fmt(r["yaw"]),
                fmt(r["flight"]),
                fmt(r["mujoco"], 1),
                "" if r["ok"] else "train lỗi",
            ]
        )
        + "</tr>"
        for r in reversed(rows)
    )
    val_rows = "".join(
        f"<tr><td>{v['round']}</td><td>{fmt(v['max_speed_all_10_seeds_passed_mps'], 1)}</td><td>"
        + ", ".join(f"{p['speed_cmd_mps']:g}: {p['passed']}/{p['count']}" for p in v["per_speed"])
        + "</td></tr>"
        for v in reversed(validations)
    )
    events = "".join(
        f"<li><code>{html.escape(e['time'])}</code> <b>{html.escape(e['kind'])}</b> "
        f"{html.escape(json.dumps({k: v for k, v in e.items() if k not in ('time', 'kind', 'to', 'cmd')}, ensure_ascii=False))}</li>"
        for e in reversed(s.get("events", [])[-40:])
    )
    gpu_html = (
        f"GPU1: {gpu['util']}% · {gpu['mem_used_gb']:.1f}/{gpu['mem_total_gb']:.0f} GB · {gpu['temp']}°C"
        if gpu
        else "GPU1: không đọc được"
    )
    run_name = html.escape(Path(d["run"]).name)
    return f"""<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="{refresh}">
<title>G1 Sprint Autopilot</title>
<style>
:root {{ --bg:#f6f7f9; --panel:#ffffff; --text:#1b1f24; --muted:#606a78; --line:#dde2e8; --grid:#e8ecf1; --ok:#1f8a4c; --bad:#c23b3b; --accent:#2f6fdf; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#111418; --panel:#1a1f25; --text:#e6e9ee; --muted:#9aa4b2; --line:#2b323b; --grid:#252b33; --ok:#3fbf73; --bad:#e06464; --accent:#6c9cff; }} }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--text); font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif; }}
main {{ max-width:1200px; margin:0 auto; padding:16px; }}
header {{ display:flex; flex-wrap:wrap; gap:12px; align-items:center; justify-content:space-between; }}
h1 {{ font-size:20px; margin:0; }} h2 {{ font-size:16px; margin:24px 0 8px; }}
.muted {{ color:var(--muted); font-size:12.5px; }}
.pill {{ padding:3px 10px; border-radius:999px; border:1px solid var(--line); font-weight:600; font-size:12px; }}
.pill.ok {{ color:var(--ok); border-color:var(--ok); }} .pill.bad {{ color:var(--bad); border-color:var(--bad); }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:12px; margin-top:12px; }}
.card, .chart, .tablewrap, .events {{ background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:12px; }}
.label {{ color:var(--muted); font-size:12.5px; }} .value {{ font-size:26px; font-weight:700; margin:2px 0; font-variant-numeric:tabular-nums; }}
.bar {{ height:8px; background:var(--grid); border-radius:6px; overflow:hidden; margin-top:12px; }} .bar div {{ height:100%; background:var(--accent); }}
.charts {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(340px,1fr)); gap:12px; }}
.chart {{ margin:0; }} figcaption {{ font-weight:600; margin-bottom:6px; }} svg {{ width:100%; height:auto; }}
.grid {{ stroke:var(--grid); }} .tick {{ fill:var(--muted); font-size:11px; }}
.legend {{ display:flex; flex-wrap:wrap; gap:12px; font-size:12px; color:var(--muted); }} .legend i {{ display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:5px; }}
.tablewrap {{ overflow-x:auto; }} table {{ border-collapse:collapse; width:100%; font-variant-numeric:tabular-nums; }}
th, td {{ text-align:right; padding:5px 8px; border-bottom:1px solid var(--line); white-space:nowrap; }} th:first-child, td:first-child {{ text-align:left; }}
th {{ color:var(--muted); font-weight:600; font-size:12px; }}
.events ul {{ margin:0; padding-left:18px; max-height:320px; overflow:auto; }} code {{ color:var(--muted); }}
</style></head>
<body><main>
<header><div><h1>G1 Sprint Autopilot · {run_name}</h1>
<div class="muted">Cập nhật {html.escape(d["generated"])} · tự tải lại mỗi {refresh} s · bắt đầu {html.escape(s.get("started", "—"))} · {html.escape(gpu_html)}</div></div>
{status}</header>
{progress}
<div class="cards">{cards_html}</div>
<h2>Biểu đồ</h2><div class="charts">{"".join(charts)}</div>
<h2>Kiểm chứng MuJoCo (10 seed mới, giữ 60 s)</h2>
<div class="tablewrap"><table><thead><tr><th>Vòng</th><th>Tốc độ cao nhất đạt mọi seed</th><th>Lệnh: đạt/tổng</th></tr></thead><tbody>{val_rows or '<tr><td colspan="3">Chưa có</td></tr>'}</tbody></table></div>
<h2>Từng vòng</h2>
<div class="tablewrap"><table><thead><tr><th>Vòng</th><th>Phút</th><th>Curriculum</th><th>LR</th><th>Isaac đủ tiêu chí</th><th>Isaac chỉ tốc độ</th><th>vx thực</th><th>RMS quay</th><th>Pha bay</th><th>MuJoCo 2 seed</th><th></th></tr></thead><tbody>{table_rows}</tbody></table></div>
<h2>Sự kiện gần nhất</h2><div class="events"><ul>{events}</ul></div>
<p class="muted">Isaac: sống ≥ 90%, vx ≥ 85% lệnh, RMS quay &lt; 0,25 rad/s, lệch hướng &lt; 30° (2 seed × 32 robot, lấy kém hơn). MuJoCo: không ngã, vx ≥ 85%, RMSE, quay, hướng, dừng, khớp không vượt tốc độ định mức. Dừng êm: <code>echo '{{"stop": true}}' &gt; {html.escape(d["run"])}/control.json</code></p>
</main></body></html>
"""


# ----------------------------------------------------------------------------------------- terminal
def render_text(d):
    s, rows, cur = d["state"], per_round(d["rounds"]), d["current"] or {}
    best_i, best_c, best_m = (
        s.get("best_isaac") or {},
        s.get("best_capability") or {},
        s.get("best_mujoco") or {},
    )
    lines = [
        f"G1 Sprint Autopilot · {Path(d['run']).name} · {d['generated']}",
        f"Trạng thái: {'ĐANG CHẠY (PID %s)' % d['pid'] if d['pid'] else 'KHÔNG CHẠY'}"
        + (f" · GPU1 {d['gpu']['util']}% {d['gpu']['mem_used_gb']:.1f} GB" if d["gpu"] else ""),
    ]
    if cur.get("iterations"):
        lines.append(f"Vòng {cur['round']}: {cur['phase']} {cur['iteration']}/{cur['iterations']}")
    elif cur:
        lines.append(f"Vòng {cur['round']}: {cur['phase']}")
    lines += [
        f"MuJoCo kiểm chứng 10 seed × 60 s: {fmt(best_m.get('validation_10seeds_60s_max_passed_mps'), 1, ' m/s')} "
        f"(bản vòng {best_m.get('round', '—')}, chấm nhanh {fmt(best_m.get('speed'), 1, ' m/s')})",
        f"Isaac đủ tiêu chí: {fmt(best_i.get('speed'), 1, ' m/s')} (vòng {best_i.get('round', '—')})"
        f" · chỉ tốc độ: {fmt(best_c.get('speed'), 1, ' m/s')} (vòng {best_c.get('round', '—')}, quay {fmt(best_c.get('yaw_rate_rms'))})",
        f"Curriculum {fmt(s.get('level'), 1)} m/s · LR {s.get('lr', 0):.1e} · nút chỉnh {s.get('knobs_applied', [])}",
        "",
        f"{'vòng':>4} {'phút':>5} {'Isaac':>6} {'tốc độ':>7} {'vx':>6} {'quay':>6} {'bay':>6} {'MuJoCo':>7}",
    ]
    for r in rows[-12:]:
        lines.append(
            f"{r['round']:>4} {fmt(r['minutes'], 0):>5} {fmt(r['strict'], 1):>6} {fmt(r['capability'], 1):>7} "
            f"{fmt(r['vx']):>6} {fmt(r['yaw']):>6} {fmt(r['flight']):>6} {fmt(r['mujoco'], 1):>7}"
        )
    lines.append("")
    for e in s.get("events", [])[-6:]:
        lines.append(f"{e['time']} {e['kind']}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run", required=True, help="autopilot output directory (contains state.json)")
    parser.add_argument("--output", help="HTML path (default <run>/dashboard.html)")
    parser.add_argument(
        "--terminal", action="store_true", help="print a text summary instead of writing HTML"
    )
    parser.add_argument("--watch", type=float, default=0, help="repeat every N seconds")
    parser.add_argument("--refresh", type=int, default=60, help="browser auto-reload period (s)")
    args = parser.parse_args()
    output = Path(args.output or Path(args.run) / "dashboard.html")
    while True:
        try:
            data = load(args.run)
            if args.terminal:
                if args.watch:
                    os.system("clear")
                print(render_text(data), flush=True)
            else:
                tmp = output.with_suffix(".tmp")
                tmp.write_text(render_html(data, args.refresh))
                tmp.replace(output)
                if not args.watch:
                    print(f"wrote {output.resolve()}")
        except Exception as error:  # keep watching through a half-written state.json
            print(f"dashboard: {error}", flush=True)
        if not args.watch:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
