#!/usr/bin/env python3
"""
Parameter Golf Experiment Dashboard
Run on the GCP instance, access via SSH tunnel:
  gcloud compute ssh parameter-golf-mig-kpx5 --zone=us-central1-a -- -L 8080:localhost:8080
Then open http://localhost:8080
"""

import http.server
import json
import os
import re
import subprocess
import time
from pathlib import Path

PORT = 8080
REPO_DIR = os.environ.get("REPO_DIR", os.path.expanduser("~/parameter-golf"))
RESULTS_TSV = os.path.join(REPO_DIR, "results.tsv")
RUN_LOG = os.path.join(REPO_DIR, "run.log")
RUN_LOG_RECENT_SECONDS = int(os.environ.get("RUN_LOG_RECENT_SECONDS", "300"))
FINAL_METRIC_PREFIXES = (
    "final_int6_roundtrip_exact",
    "final_int6_sliding_window_exact",
    "final_int8_zlib_roundtrip_exact",
    "legal_ttt_exact",
    "legal_ttt ",
)


def parse_results_tsv():
    """Parse results.tsv into list of experiment dicts."""
    experiments = []
    if not os.path.exists(RESULTS_TSV):
        return experiments
    with open(RESULTS_TSV) as f:
        lines = f.readlines()
    if len(lines) < 2:
        return experiments
    for i, line in enumerate(lines[1:], 1):
        parts = line.strip().split("\t")
        if len(parts) >= 5:
            experiments.append({
                "idx": i,
                "commit": parts[0],
                "val_bpb": float(parts[1]) if parts[1] != "0.000000" else None,
                "artifact_mb": float(parts[2]) if parts[2] != "0.00" else None,
                "status": parts[3],
                "description": parts[4] if len(parts) > 4 else "",
            })
    return experiments


def empty_run_data():
    return {
        "train_steps": [],
        "val_checks": [],
        "ttt_chunks": [],
        "sliding_eval": [],
        "diagnostics": [],
        "final_metrics": {},
        "wallclock_stopped": None,
        "eval_only": False,
        "ttt_started": None,
        "completed": False,
        "crashed": False,
        "log_mtime": None,
    }


def parse_run_log():
    """Parse current run.log for training, eval progress, and final metrics."""
    run_data = empty_run_data()
    ttt_finished = False

    if not os.path.exists(RUN_LOG):
        return run_data

    run_data["log_mtime"] = os.path.getmtime(RUN_LOG)

    with open(RUN_LOG) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith("EVAL_ONLY=1:"):
                run_data["eval_only"] = True

            # Training steps
            m = re.match(r"step:(\d+)/(\d+)\s+train_loss:([\d.]+)\s+train_time:(\d+)ms", line)
            if m:
                run_data["train_steps"].append({
                    "step": int(m.group(1)),
                    "total": int(m.group(2)),
                    "loss": float(m.group(3)),
                    "time_ms": int(m.group(4)),
                })

            # Val checks during training
            m = re.match(r"step:(\d+)/(\d+)\s+val_loss:([\d.]+)\s+val_bpb:([\d.]+)", line)
            if m:
                run_data["val_checks"].append({
                    "step": int(m.group(1)),
                    "val_loss": float(m.group(3)),
                    "val_bpb": float(m.group(4)),
                })

            # TTT chunks
            m = re.match(r"\s*ttt_chunk\s*\[(\d+)/(\d+)\]\s*bpb=([\d.]+)\s*time=([\d.]+)s", line)
            if m:
                run_data["ttt_chunks"].append({
                    "chunk": int(m.group(1)),
                    "total": int(m.group(2)),
                    "bpb": float(m.group(3)),
                    "time_s": float(m.group(4)),
                })

            # Sliding eval progress (prepare.py / eval-only progress)
            m = re.match(r"\s*sliding_eval\s*\[\s*([\d.]+)%\]\s*(\d+)/(\d+)\s+windows\s+running_bpb=([\d.]+)", line)
            if m:
                run_data["sliding_eval"].append({
                    "pct": float(m.group(1)),
                    "done": int(m.group(2)),
                    "total": int(m.group(3)),
                    "bpb": float(m.group(4)),
                })

            # TTT config/start
            m = re.match(r"ttt_sliding:start chunks=(\d+)\s+chunk_tokens=(\d+)\s+total_windows=(\d+)\s+stride=(\d+)", line)
            if m:
                run_data["ttt_started"] = {
                    "chunks": int(m.group(1)),
                    "chunk_tokens": int(m.group(2)),
                    "total_windows": int(m.group(3)),
                    "stride": int(m.group(4)),
                }

            # Wallclock stop
            if "stopping_early" in line or "wallclock_cap" in line:
                m2 = re.search(r"step:(\d+)", line)
                if m2:
                    run_data["wallclock_stopped"] = int(m2.group(1))

            # Diagnostic lines
            if "DIAGNOSTIC" in line:
                m = re.search(r"val_bpb:([\d.]+)", line)
                if m:
                    run_data["diagnostics"].append({"label": "post_ema", "val_bpb": float(m.group(1))})

            # Final metrics
            for prefix in FINAL_METRIC_PREFIXES:
                if line.startswith(prefix.strip()):
                    m = re.search(r"val_bpb:([\d.]+)", line)
                    if m:
                        key = prefix.strip().replace(" ", "_")
                        run_data["final_metrics"][key] = float(m.group(1))

            # Submission size
            m = re.search(r"Total submission size.*?:\s*(\d+)\s*bytes", line)
            if m:
                run_data["final_metrics"]["artifact_bytes"] = int(m.group(1))

            # SWA/QAT events
            if line.startswith("swa:start"):
                m2 = re.search(r"step:(\d+)", line)
                if m2:
                    run_data["diagnostics"].append({"label": "swa_start", "step": int(m2.group(1))})
            if line.startswith("late_qat:enabled"):
                m2 = re.search(r"step:(\d+)", line)
                if m2:
                    run_data["diagnostics"].append({"label": "qat_start", "step": int(m2.group(1))})

            if line.startswith("ttt_sliding:done"):
                ttt_finished = True

            if line.startswith("Traceback (most recent call last):") or "ChildFailedError" in line:
                run_data["crashed"] = True

    if run_data["ttt_started"]:
        run_data["completed"] = (
            ttt_finished
            or "legal_ttt_exact" in run_data["final_metrics"]
            or "legal_ttt" in run_data["final_metrics"]
        )
    elif run_data["eval_only"]:
        run_data["completed"] = (
            "final_int8_zlib_roundtrip_exact" in run_data["final_metrics"]
            or "legal_ttt_exact" in run_data["final_metrics"]
            or "legal_ttt" in run_data["final_metrics"]
        )
    else:
        run_data["completed"] = any(
            key in run_data["final_metrics"]
            for key in (
                "final_int6_roundtrip_exact",
                "final_int6_sliding_window_exact",
                "final_int8_zlib_roundtrip_exact",
                "legal_ttt_exact",
                "legal_ttt",
            )
        )

    return run_data


def get_git_log():
    """Get recent experiment commits."""
    try:
        result = subprocess.run(
            ["git", "log", "--oneline", "-30", "--format=%h\t%s\t%ci"],
            capture_output=True, text=True, cwd=REPO_DIR, timeout=5
        )
        commits = []
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            parts = line.split("\t", 2)
            if len(parts) >= 2:
                commits.append({
                    "hash": parts[0],
                    "message": parts[1],
                    "date": parts[2] if len(parts) > 2 else "",
                })
        return commits
    except Exception:
        return []


def get_disk_usage():
    """Get artifact size info."""
    try:
        result = subprocess.run(["du", "-sh", REPO_DIR], capture_output=True, text=True, timeout=5)
        return result.stdout.strip().split("\t")[0] if result.stdout else "?"
    except Exception:
        return "?"


def get_active_run_process():
    """Return the active training/eval command if one is running."""
    try:
        result = subprocess.run(["pgrep", "-af", "train_gpt.py|launch_train_gcp_h100.sh"], capture_output=True, text=True, timeout=5)
        for line in result.stdout.strip().splitlines():
            parts = line.strip().split(None, 1)
            if len(parts) != 2:
                continue
            cmd = parts[1]
            if "dashboard.py" in cmd:
                continue
            if "train_gpt.py" not in cmd and "launch_train_gcp_h100.sh" not in cmd:
                continue
            if not any(token in cmd for token in ("torchrun", "python", "timeout", "launch_train_gcp_h100.sh")):
                continue
            return {"pid": int(parts[0]), "command": cmd}
        return None
    except Exception:
        return None


def is_running(run_data=None):
    """Check if a training/eval run is currently active."""
    if get_active_run_process():
        return True

    if run_data is None:
        run_data = parse_run_log()

    if not run_data["log_mtime"]:
        return False

    recently_updated = (time.time() - run_data["log_mtime"]) <= RUN_LOG_RECENT_SECONDS
    has_progress_signal = bool(
        run_data["train_steps"]
        or run_data["val_checks"]
        or run_data["ttt_chunks"]
        or run_data["sliding_eval"]
        or run_data["eval_only"]
        or run_data["ttt_started"]
    )
    return recently_updated and has_progress_signal and not run_data["completed"] and not run_data["crashed"]


def build_html():
    experiments = parse_results_tsv()
    run_data = parse_run_log()
    commits = get_git_log()
    running = is_running(run_data)
    now = time.strftime("%Y-%m-%d %H:%M:%S")

    # Compute waterfall from final metrics
    fm = run_data["final_metrics"]
    waterfall = []
    if run_data["val_checks"]:
        last_val = run_data["val_checks"][-1]
        waterfall.append({"stage": "End of Training", "bpb": last_val["val_bpb"]})
    for d in run_data["diagnostics"]:
        if d["label"] == "post_ema":
            waterfall.append({"stage": "Post-EMA", "bpb": d["val_bpb"]})
    if "final_int6_roundtrip_exact" in fm:
        waterfall.append({"stage": "Post-Quantization", "bpb": fm["final_int6_roundtrip_exact"]})
    if "final_int6_sliding_window_exact" in fm:
        waterfall.append({"stage": "Sliding Window", "bpb": fm["final_int6_sliding_window_exact"]})
    if "legal_ttt_exact" in fm:
        waterfall.append({"stage": "Post-TTT (Final)", "bpb": fm["legal_ttt_exact"]})

    # Best result
    valid = [e for e in experiments if e["val_bpb"] and e["status"] in ("keep",)]
    best = min(valid, key=lambda x: x["val_bpb"]) if valid else None
    best_bpb = f"{best['val_bpb']:.4f}" if best else "N/A"
    best_desc = f"{best['description'][:40]}..." if best else "No valid runs yet"

    # Status badge
    if running:
        if run_data["ttt_chunks"]:
            last_chunk = run_data["ttt_chunks"][-1]
            status_text = f"EVAL/TTT in progress - chunk {last_chunk['chunk']}/{last_chunk['total']} - bpb {last_chunk['bpb']:.4f}"
            status_color = "#f59e0b"
        elif run_data["sliding_eval"]:
            last_eval = run_data["sliding_eval"][-1]
            status_text = f"EVAL in progress - windows {last_eval['done']}/{last_eval['total']} - bpb {last_eval['bpb']:.4f}"
            status_color = "#f59e0b"
        elif run_data["ttt_started"]:
            status_text = f"EVAL/TTT starting - 0/{run_data['ttt_started']['chunks']} chunks"
            status_color = "#8b5cf6"
        elif run_data["train_steps"]:
            last_step = run_data["train_steps"][-1]
            status_text = f"TRAINING in progress - step {last_step['step']}/{last_step['total']} - loss {last_step['loss']:.4f}"
            status_color = "#3b82f6"
        elif run_data["eval_only"]:
            status_text = "EVAL_ONLY loading artifact"
            status_color = "#8b5cf6"
        else:
            status_text = "RUN STARTING..."
            status_color = "#8b5cf6"
    else:
        if run_data["crashed"]:
            status_text = "ERROR - last run crashed"
            status_color = "#ef4444"
        elif fm and run_data["completed"]:
            status_text = "IDLE - last run complete"
            status_color = "#10b981"
        elif run_data["ttt_chunks"]:
            last_chunk = run_data["ttt_chunks"][-1]
            status_text = f"IDLE - last eval stopped at chunk {last_chunk['chunk']}/{last_chunk['total']}"
            status_color = "#f97316"
        elif run_data["sliding_eval"]:
            last_eval = run_data["sliding_eval"][-1]
            status_text = f"IDLE - last eval stopped at windows {last_eval['done']}/{last_eval['total']}"
            status_color = "#f97316"
        elif run_data["eval_only"] or run_data["ttt_started"]:
            status_text = "IDLE - last eval incomplete"
            status_color = "#f97316"
        else:
            status_text = "IDLE - no run data"
            status_color = "#6b7280"

    progress_points = []
    progress_name = "BPB"
    progress_x_title = "Eval Chunk"
    progress_hover = "Chunk %{x}<br>BPB: %{y:.4f}<br>Time: %{customdata:.0f}s"
    progress_x = []
    progress_y = []
    progress_customdata = []
    if run_data["ttt_chunks"]:
        progress_points = run_data["ttt_chunks"]
        progress_name = "TTT BPB"
        progress_x = [t["chunk"] for t in progress_points]
        progress_y = [t["bpb"] for t in progress_points]
        progress_customdata = [t["time_s"] for t in progress_points]
    elif run_data["sliding_eval"]:
        progress_points = run_data["sliding_eval"]
        progress_name = "Eval BPB"
        progress_x_title = "Windows Scored"
        progress_x = [p["done"] for p in progress_points]
        progress_y = [p["bpb"] for p in progress_points]
        progress_customdata = [p["pct"] for p in progress_points]
        progress_hover = (
            f"Windows %{{x}}/{progress_points[-1]['total']}<br>"
            "BPB: %{y:.4f}<br>Done: %{customdata:.1f}%"
        )

    waterfall_bpb = [w["bpb"] for w in waterfall]
    waterfall_range = [min(waterfall_bpb) - 0.02, max(waterfall_bpb) + 0.02] if waterfall_bpb else [0, 1]
    waterfall_annotations = (
        []
        if waterfall_bpb
        else [{"text": "No completed eval stage yet", "xref": "paper", "yref": "paper", "x": 0.5, "y": 0.5, "showarrow": False, "font": {"size": 12, "color": "#64748b"}}]
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta http-equiv="refresh" content="30">
<title>Parameter Golf Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, monospace; background: #0f172a; color: #e2e8f0; padding: 20px; }}
  .header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; padding: 16px 20px; background: #1e293b; border-radius: 12px; border: 1px solid #334155; }}
  .header h1 {{ font-size: 22px; font-weight: 700; }}
  .header .meta {{ font-size: 13px; color: #94a3b8; }}
  .status-badge {{ display: inline-block; padding: 6px 14px; border-radius: 20px; font-size: 13px; font-weight: 600; color: #fff; }}
  .stats {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 20px; }}
  .stat-card {{ background: #1e293b; border: 1px solid #334155; border-radius: 10px; padding: 16px; }}
  .stat-card .label {{ font-size: 11px; text-transform: uppercase; letter-spacing: 1px; color: #64748b; margin-bottom: 4px; }}
  .stat-card .value {{ font-size: 28px; font-weight: 700; }}
  .stat-card .sub {{ font-size: 12px; color: #94a3b8; margin-top: 2px; }}
  .charts {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 20px; }}
  .chart-card {{ background: #1e293b; border: 1px solid #334155; border-radius: 10px; padding: 16px; }}
  .chart-card.full {{ grid-column: 1 / -1; }}
  .chart-card h3 {{ font-size: 14px; margin-bottom: 10px; color: #94a3b8; }}
  .table-wrap {{ background: #1e293b; border: 1px solid #334155; border-radius: 10px; padding: 16px; margin-bottom: 20px; overflow-x: auto; }}
  .table-wrap h3 {{ font-size: 14px; margin-bottom: 10px; color: #94a3b8; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th {{ text-align: left; padding: 8px 12px; border-bottom: 2px solid #334155; color: #64748b; text-transform: uppercase; font-size: 11px; letter-spacing: 1px; }}
  td {{ padding: 8px 12px; border-bottom: 1px solid #1e293b; }}
  tr:hover td {{ background: #1e293b; }}
  .status-keep {{ color: #10b981; font-weight: 600; }}
  .status-discard {{ color: #ef4444; }}
  .status-crash {{ color: #f59e0b; }}
  .status-invalid {{ color: #8b5cf6; }}
  .status-early-stop {{ color: #f97316; }}
  .commit-hash {{ font-family: monospace; color: #60a5fa; }}
  .reference-line {{ font-size: 11px; color: #475569; margin-top: 4px; }}
</style>
</head>
<body>

<div class="header">
  <div>
    <h1>Parameter Golf Dashboard</h1>
    <div class="meta">Last updated: {now} | Auto-refresh: 30s</div>
  </div>
  <div>
    <span class="status-badge" style="background:{status_color}">{status_text}</span>
  </div>
</div>

<div class="stats">
  <div class="stat-card">
    <div class="label">Best BPB</div>
    <div class="value" style="color:#10b981">{best_bpb}</div>
    <div class="sub">{best_desc}</div>
  </div>
  <div class="stat-card">
    <div class="label">Total Experiments</div>
    <div class="value">{len(experiments)}</div>
    <div class="sub">{sum(1 for e in experiments if e['status']=='keep')} kept, {sum(1 for e in experiments if e['status']=='discard')} discarded, {sum(1 for e in experiments if e['status']=='crash')} crashed</div>
  </div>
  <div class="stat-card">
    <div class="label">Artifact Size</div>
    <div class="value">{fm.get('artifact_bytes', 0)/1_000_000:.2f} MB</div>
    <div class="sub">Limit: 16.00 MB ({fm.get('artifact_bytes', 0)/16_000_000*100:.1f}% used)</div>
  </div>
  <div class="stat-card">
    <div class="label">Competition Target</div>
    <div class="value" style="color:#f59e0b">0.9850</div>
    <div class="sub">PR #741 (Cosine TTT + N-gram Cache)</div>
  </div>
</div>

<div class="charts">

  <!-- Experiment BPB Timeline -->
  <div class="chart-card full">
    <h3>Experiment Results Over Time</h3>
    <div id="exp-timeline" style="height:300px"></div>
  </div>

  <!-- Training Loss Curve (current run) -->
  <div class="chart-card">
    <h3>Current Run: Training Loss</h3>
    <div id="train-loss" style="height:280px"></div>
  </div>

  <!-- TTT Progression (current run) -->
  <div class="chart-card">
    <h3>Current Run: TTT/Eval Progression</h3>
    <div id="ttt-progress" style="height:280px"></div>
  </div>

  <!-- Loss Waterfall -->
  <div class="chart-card">
    <h3>Loss Waterfall (Current/Last Run)</h3>
    <div id="waterfall" style="height:280px"></div>
  </div>

  <!-- Artifact Size vs BPB -->
  <div class="chart-card">
    <h3>Artifact Size vs BPB</h3>
    <div id="size-bpb" style="height:280px"></div>
  </div>

</div>

<!-- Experiments Table -->
<div class="table-wrap">
  <h3>Experiment Log</h3>
  <table>
    <thead>
      <tr><th>#</th><th>Commit</th><th>BPB</th><th>Size (MB)</th><th>Status</th><th>Description</th></tr>
    </thead>
    <tbody>
"""

    for e in reversed(experiments):
        status_class = f"status-{e['status']}"
        bpb_str = f"{e['val_bpb']:.4f}" if e['val_bpb'] else "---"
        size_str = f"{e['artifact_mb']:.2f}" if e['artifact_mb'] else "---"
        html += f"""      <tr>
        <td>{e['idx']}</td>
        <td class="commit-hash">{e['commit']}</td>
        <td><b>{bpb_str}</b></td>
        <td>{size_str}</td>
        <td class="{status_class}">{e['status']}</td>
        <td>{e['description']}</td>
      </tr>\n"""

    html += """    </tbody>
  </table>
</div>

<!-- Recent Commits -->
<div class="table-wrap">
  <h3>Recent Git Commits</h3>
  <table>
    <thead>
      <tr><th>Hash</th><th>Message</th><th>Date</th></tr>
    </thead>
    <tbody>
"""

    for c in commits[:15]:
        html += f"""      <tr>
        <td class="commit-hash">{c['hash']}</td>
        <td>{c['message']}</td>
        <td style="color:#64748b;white-space:nowrap">{c['date'][:19]}</td>
      </tr>\n"""

    html += """    </tbody>
  </table>
</div>

<script>
const plotBg = '#1e293b';
const plotGrid = '#334155';
const plotText = '#94a3b8';
const layout_base = {
  paper_bgcolor: plotBg, plot_bgcolor: plotBg, font: {color: plotText, size: 11},
  margin: {l: 50, r: 20, t: 10, b: 40},
  xaxis: {gridcolor: plotGrid, zerolinecolor: plotGrid},
  yaxis: {gridcolor: plotGrid, zerolinecolor: plotGrid},
};

// -- Experiment Timeline --
"""

    # Experiment timeline data
    keep_exps = [e for e in experiments if e["val_bpb"] and e["status"] == "keep"]
    discard_exps = [e for e in experiments if e["val_bpb"] and e["status"] == "discard"]
    crash_exps = [e for e in experiments if e["status"] == "crash"]
    invalid_exps = [e for e in experiments if e["val_bpb"] and e["status"] == "invalid"]

    html += f"""
var keep_x = {json.dumps([e['idx'] for e in keep_exps])};
var keep_y = {json.dumps([e['val_bpb'] for e in keep_exps])};
var keep_text = {json.dumps([e['description'][:50] for e in keep_exps])};
var discard_x = {json.dumps([e['idx'] for e in discard_exps])};
var discard_y = {json.dumps([e['val_bpb'] for e in discard_exps])};
var discard_text = {json.dumps([e['description'][:50] for e in discard_exps])};
var crash_x = {json.dumps([e['idx'] for e in crash_exps])};
var invalid_x = {json.dumps([e['idx'] for e in invalid_exps])};
var invalid_y = {json.dumps([e['val_bpb'] for e in invalid_exps])};

Plotly.newPlot('exp-timeline', [
  {{x: keep_x, y: keep_y, text: keep_text, mode: 'lines+markers', name: 'Kept', marker: {{color: '#10b981', size: 10}}, line: {{color: '#10b981', width: 2}}, hovertemplate: '%{{text}}<br>BPB: %{{y:.4f}}'}},
  {{x: discard_x, y: discard_y, text: discard_text, mode: 'markers', name: 'Discarded', marker: {{color: '#ef4444', size: 8, symbol: 'x'}}, hovertemplate: '%{{text}}<br>BPB: %{{y:.4f}}'}},
  {{x: invalid_x, y: invalid_y, mode: 'markers', name: 'Invalid (>16MB)', marker: {{color: '#8b5cf6', size: 8, symbol: 'diamond'}}}},
], {{
  ...layout_base,
  xaxis: {{...layout_base.xaxis, title: 'Experiment #'}},
  yaxis: {{...layout_base.yaxis, title: 'val_bpb'}},
  shapes: [
    {{type:'line', x0:0, x1:{len(experiments)+1}, y0:1.1194, y1:1.1194, line:{{color:'#475569', dash:'dot', width:1}}}},
    {{type:'line', x0:0, x1:{len(experiments)+1}, y0:0.9850, y1:0.9850, line:{{color:'#f59e0b', dash:'dot', width:1}}}},
  ],
  annotations: [
    {{x:{len(experiments)}, y:1.1194, text:'Merged SOTA (1.1194)', showarrow:false, font:{{size:10, color:'#475569'}}, yshift:12}},
    {{x:{len(experiments)}, y:0.9850, text:'PR #741 (0.9850)', showarrow:false, font:{{size:10, color:'#f59e0b'}}, yshift:12}},
  ],
  showlegend: true, legend: {{x:1, y:1, xanchor:'right', bgcolor:'rgba(0,0,0,0)'}},
}});
"""

    # Training loss curve
    train_steps = run_data["train_steps"]
    html += f"""
var train_x = {json.dumps([s['step'] for s in train_steps])};
var train_y = {json.dumps([s['loss'] for s in train_steps])};
var val_x = {json.dumps([v['step'] for v in run_data['val_checks']])};
var val_y = {json.dumps([v['val_bpb'] for v in run_data['val_checks']])};

Plotly.newPlot('train-loss', [
  {{x: train_x, y: train_y, mode: 'lines', name: 'Train Loss', line: {{color: '#3b82f6', width: 1.5}}}},
  {{x: val_x, y: val_y, mode: 'markers+lines', name: 'Val BPB', line: {{color: '#f59e0b', width: 2}}, marker: {{size: 8}}, yaxis: 'y2'}},
], {{
  ...layout_base,
  xaxis: {{...layout_base.xaxis, title: 'Step'}},
  yaxis: {{...layout_base.yaxis, title: 'Train Loss', side: 'left'}},
  yaxis2: {{title: 'Val BPB', overlaying: 'y', side: 'right', gridcolor: plotGrid, titlefont: {{color: '#f59e0b'}}, tickfont: {{color: '#f59e0b'}}}},
  showlegend: true, legend: {{x:1, y:1, xanchor:'right', bgcolor:'rgba(0,0,0,0)'}},
}});
"""

    # TTT progression
    html += f"""
var progress_x = {json.dumps(progress_x)};
var progress_y = {json.dumps(progress_y)};
var progress_customdata = {json.dumps(progress_customdata)};
var progress_hover = {json.dumps(progress_hover)};

Plotly.newPlot('ttt-progress', [
  {{x: progress_x, y: progress_y, mode: 'lines', name: {json.dumps(progress_name)}, line: {{color: '#10b981', width: 2}},
    hovertemplate: progress_hover, customdata: progress_customdata}},
], {{
  ...layout_base,
  xaxis: {{...layout_base.xaxis, title: {json.dumps(progress_x_title)}}},
  yaxis: {{...layout_base.yaxis, title: 'Running BPB'}},
}});
"""

    # Waterfall
    html += f"""
var wf_stages = {json.dumps([w['stage'] for w in waterfall])};
var wf_bpb = {json.dumps(waterfall_bpb)};
var wf_colors = wf_bpb.map((v, i) => i === wf_bpb.length - 1 ? '#10b981' : (i > 0 && v < wf_bpb[i-1] ? '#3b82f6' : '#ef4444'));

Plotly.newPlot('waterfall', [
  {{x: wf_stages, y: wf_bpb, type: 'bar', marker: {{color: wf_colors}},
    text: wf_bpb.map(v => v.toFixed(4)), textposition: 'outside', textfont: {{color: plotText, size: 11}}}},
], {{
  ...layout_base,
  xaxis: {{...layout_base.xaxis}},
  yaxis: {{...layout_base.yaxis, title: 'BPB', range: {json.dumps(waterfall_range)}}},
  annotations: {json.dumps(waterfall_annotations)},
}});
"""

    # Size vs BPB scatter
    sized_exps = [e for e in experiments if e["val_bpb"] and e["artifact_mb"]]
    size_limit_y = max((e["val_bpb"] for e in sized_exps if e["val_bpb"] > 0), default=1.2)
    html += f"""
var sb_x = {json.dumps([e['artifact_mb'] for e in sized_exps])};
var sb_y = {json.dumps([e['val_bpb'] for e in sized_exps])};
var sb_text = {json.dumps([e['description'][:40] for e in sized_exps])};
var sb_colors = {json.dumps(['#10b981' if e['status']=='keep' else '#ef4444' if e['status']=='discard' else '#8b5cf6' for e in sized_exps])};

Plotly.newPlot('size-bpb', [
  {{x: sb_x, y: sb_y, text: sb_text, mode: 'markers', marker: {{color: sb_colors, size: 10}},
    hovertemplate: '%{{text}}<br>BPB: %{{y:.4f}}<br>Size: %{{x:.2f}} MB'}},
], {{
  ...layout_base,
  xaxis: {{...layout_base.xaxis, title: 'Artifact Size (MB)', range: [0, 17]}},
  yaxis: {{...layout_base.yaxis, title: 'val_bpb'}},
  shapes: [{{type:'line', x0:16, x1:16, y0:0, y1:2, line:{{color:'#ef4444', dash:'dash', width:2}}}}],
  annotations: [{{x:16, y: {size_limit_y}, text:'16MB limit', showarrow:false, font:{{size:10, color:'#ef4444'}}, xshift:-30}}],
}});
</script>

<div class="reference-line" style="text-align:center; margin-top:20px; padding:10px;">
  Reference: Merged SOTA 1.1194 | PR #741 0.9850 | PR #745 1.0222 | PR #740 1.0909
</div>

</body>
</html>"""
    return html


class DashboardHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            html = build_html()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(html.encode())
        elif self.path == "/api/data":
            data = {
                "experiments": parse_results_tsv(),
                "run": parse_run_log(),
                "commits": get_git_log(),
                "running": is_running(),
                "timestamp": time.time(),
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(data).encode())
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        pass  # suppress access logs


if __name__ == "__main__":
    print(f"Dashboard starting on http://localhost:{PORT}")
    print(f"Repo dir: {REPO_DIR}")
    print(f"Results: {RESULTS_TSV}")
    print(f"Run log: {RUN_LOG}")
    print(f"Auto-refreshes every 30s")
    server = http.server.HTTPServer(("0.0.0.0", PORT), DashboardHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()
