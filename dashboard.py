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

PORT = 8080
REPO_DIR = os.environ.get("REPO_DIR", os.path.expanduser("~/parameter-golf"))
RESULTS_TSV = os.path.join(REPO_DIR, "results.tsv")
RUN_LOG = os.path.join(REPO_DIR, "run.log")


def parse_results_tsv():
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


def parse_run_log():
    train_steps = []
    val_checks = []
    ttt_chunks = []
    diagnostics = []
    final_metrics = {}
    wallclock_stopped = None

    if not os.path.exists(RUN_LOG):
        return {"train_steps": [], "val_checks": [], "ttt_chunks": [],
                "diagnostics": [], "final_metrics": {}, "wallclock_stopped": None}

    with open(RUN_LOG) as f:
        for line in f:
            line = line.strip()

            m = re.match(r"step:(\d+)/(\d+)\s+train_loss:([\d.]+)\s+train_time:(\d+)ms", line)
            if m:
                train_steps.append({
                    "step": int(m.group(1)), "total": int(m.group(2)),
                    "loss": float(m.group(3)), "time_ms": int(m.group(4)),
                })

            m = re.match(r"step:(\d+)/(\d+)\s+val_loss:([\d.]+)\s+val_bpb:([\d.]+)", line)
            if m:
                val_checks.append({
                    "step": int(m.group(1)),
                    "val_loss": float(m.group(3)), "val_bpb": float(m.group(4)),
                })

            m = re.match(r"\s*ttt_chunk\s*\[(\d+)/(\d+)\]\s*bpb=([\d.]+)\s*time=([\d.]+)s", line)
            if m:
                ttt_chunks.append({
                    "chunk": int(m.group(1)), "total": int(m.group(2)),
                    "bpb": float(m.group(3)), "time_s": float(m.group(4)),
                })

            if "stopping_early" in line or "wallclock_cap" in line:
                m2 = re.search(r"step:(\d+)", line)
                if m2:
                    wallclock_stopped = int(m2.group(1))

            if "DIAGNOSTIC" in line:
                m = re.search(r"val_bpb:([\d.]+)", line)
                if m:
                    diagnostics.append({"label": "post_ema", "val_bpb": float(m.group(1))})

            for prefix in ["final_int6_roundtrip_exact", "final_int6_sliding_window_exact",
                           "final_int8_zlib_roundtrip_exact", "legal_ttt_exact", "legal_ttt "]:
                if line.startswith(prefix.strip()):
                    m = re.search(r"val_bpb:([\d.]+)", line)
                    if m:
                        key = prefix.strip().replace(" ", "_")
                        final_metrics[key] = float(m.group(1))

            m = re.search(r"Total submission size.*?:\s*(\d+)\s*bytes", line)
            if m:
                final_metrics["artifact_bytes"] = int(m.group(1))

            if line.startswith("swa:start"):
                m2 = re.search(r"step:(\d+)", line)
                if m2:
                    diagnostics.append({"label": "swa_start", "step": int(m2.group(1))})
            if line.startswith("late_qat:enabled"):
                m2 = re.search(r"step:(\d+)", line)
                if m2:
                    diagnostics.append({"label": "qat_start", "step": int(m2.group(1))})

    return {
        "train_steps": train_steps, "val_checks": val_checks,
        "ttt_chunks": ttt_chunks, "diagnostics": diagnostics,
        "final_metrics": final_metrics, "wallclock_stopped": wallclock_stopped,
    }


def get_git_log():
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


def is_running():
    try:
        result = subprocess.run(
            ["pgrep", "-f", "torchrun.*train_gpt"],
            capture_output=True, text=True, timeout=5
        )
        return bool(result.stdout.strip())
    except Exception:
        return False


def build_html():
    experiments = parse_results_tsv()
    run_data = parse_run_log()
    commits = get_git_log()
    running = is_running()
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    fm = run_data["final_metrics"]

    # Waterfall
    waterfall = []
    if run_data["val_checks"]:
        waterfall.append({"stage": "End of Training", "bpb": run_data["val_checks"][-1]["val_bpb"]})
    for d in run_data["diagnostics"]:
        if d["label"] == "post_ema":
            waterfall.append({"stage": "Post-EMA", "bpb": d["val_bpb"]})
    if "final_int6_roundtrip_exact" in fm:
        waterfall.append({"stage": "Post-Quantization", "bpb": fm["final_int6_roundtrip_exact"]})
    if "final_int6_sliding_window_exact" in fm:
        waterfall.append({"stage": "Sliding Window", "bpb": fm["final_int6_sliding_window_exact"]})
    if "legal_ttt_exact" in fm:
        waterfall.append({"stage": "Post-TTT (Final)", "bpb": fm["legal_ttt_exact"]})

    # Best
    valid = [e for e in experiments if e["val_bpb"] and e["status"] == "keep"]
    best = min(valid, key=lambda x: x["val_bpb"]) if valid else None

    # Status
    if running:
        if run_data["ttt_chunks"]:
            lc = run_data["ttt_chunks"][-1]
            status_text = "EVAL/TTT - chunk %d/%d - bpb %.4f" % (lc["chunk"], lc["total"], lc["bpb"])
            status_color = "#f59e0b"
        elif run_data["train_steps"]:
            ls = run_data["train_steps"][-1]
            status_text = "TRAINING - step %d/%d - loss %.4f" % (ls["step"], ls["total"], ls["loss"])
            status_color = "#3b82f6"
        else:
            status_text = "RUN STARTING..."
            status_color = "#8b5cf6"
    else:
        if fm:
            status_text = "IDLE - last run complete"
            status_color = "#10b981"
        else:
            status_text = "IDLE - no run data"
            status_color = "#6b7280"

    best_bpb_str = "%.4f" % best["val_bpb"] if best else "N/A"
    best_desc_str = best["description"][:40] if best else "No valid runs yet"
    n_total = len(experiments)
    n_keep = sum(1 for e in experiments if e["status"] == "keep")
    n_discard = sum(1 for e in experiments if e["status"] == "discard")
    n_crash = sum(1 for e in experiments if e["status"] == "crash")
    artifact_mb = fm.get("artifact_bytes", 0) / 1_000_000
    artifact_pct = fm.get("artifact_bytes", 0) / 16_000_000 * 100

    # Experiment categories
    keep_exps = [e for e in experiments if e["val_bpb"] and e["status"] == "keep"]
    discard_exps = [e for e in experiments if e["val_bpb"] and e["status"] == "discard"]
    invalid_exps = [e for e in experiments if e["val_bpb"] and e["status"] == "invalid"]
    sized_exps = [e for e in experiments if e["val_bpb"] and e["artifact_mb"]]

    # Build experiment table rows
    table_rows = ""
    for e in reversed(experiments):
        bpb_str = "%.4f" % e["val_bpb"] if e["val_bpb"] else "---"
        size_str = "%.2f" % e["artifact_mb"] if e["artifact_mb"] else "---"
        table_rows += '<tr><td>%d</td><td class="commit-hash">%s</td><td><b>%s</b></td><td>%s</td><td class="status-%s">%s</td><td>%s</td></tr>\n' % (
            e["idx"], e["commit"], bpb_str, size_str, e["status"], e["status"], e["description"]
        )

    # Build commit table rows
    commit_rows = ""
    for c in commits[:15]:
        commit_rows += '<tr><td class="commit-hash">%s</td><td>%s</td><td style="color:#64748b;white-space:nowrap">%s</td></tr>\n' % (
            c["hash"], c["message"], c["date"][:19]
        )

    # All chart data as JSON to inject safely
    chart_data = json.dumps({
        "keep": {"x": [e["idx"] for e in keep_exps], "y": [e["val_bpb"] for e in keep_exps],
                 "text": [e["description"][:50] for e in keep_exps]},
        "discard": {"x": [e["idx"] for e in discard_exps], "y": [e["val_bpb"] for e in discard_exps],
                    "text": [e["description"][:50] for e in discard_exps]},
        "invalid": {"x": [e["idx"] for e in invalid_exps], "y": [e["val_bpb"] for e in invalid_exps]},
        "train": {"x": [s["step"] for s in run_data["train_steps"]],
                  "y": [s["loss"] for s in run_data["train_steps"]]},
        "val": {"x": [v["step"] for v in run_data["val_checks"]],
                "y": [v["val_bpb"] for v in run_data["val_checks"]]},
        "ttt": {"x": [t["chunk"] for t in run_data["ttt_chunks"]],
                "y": [t["bpb"] for t in run_data["ttt_chunks"]],
                "time": [t["time_s"] for t in run_data["ttt_chunks"]]},
        "waterfall": {"stages": [w["stage"] for w in waterfall],
                      "bpb": [w["bpb"] for w in waterfall]},
        "sizebpb": {"x": [e["artifact_mb"] for e in sized_exps],
                    "y": [e["val_bpb"] for e in sized_exps],
                    "text": [e["description"][:40] for e in sized_exps],
                    "colors": ["#10b981" if e["status"] == "keep" else "#ef4444" if e["status"] == "discard" else "#8b5cf6" for e in sized_exps]},
        "n_exp": n_total,
    })

    # Use %s substitution for the HTML template — no f-strings touching JS
    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta http-equiv="refresh" content="30">
<title>Parameter Golf Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, monospace; background: #0f172a; color: #e2e8f0; padding: 20px; }
  .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; padding: 16px 20px; background: #1e293b; border-radius: 12px; border: 1px solid #334155; }
  .header h1 { font-size: 22px; font-weight: 700; }
  .header .meta { font-size: 13px; color: #94a3b8; }
  .status-badge { display: inline-block; padding: 6px 14px; border-radius: 20px; font-size: 13px; font-weight: 600; color: #fff; }
  .stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 20px; }
  .stat-card { background: #1e293b; border: 1px solid #334155; border-radius: 10px; padding: 16px; }
  .stat-card .label { font-size: 11px; text-transform: uppercase; letter-spacing: 1px; color: #64748b; margin-bottom: 4px; }
  .stat-card .value { font-size: 28px; font-weight: 700; }
  .stat-card .sub { font-size: 12px; color: #94a3b8; margin-top: 2px; }
  .charts { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 20px; }
  .chart-card { background: #1e293b; border: 1px solid #334155; border-radius: 10px; padding: 16px; }
  .chart-card.full { grid-column: 1 / -1; }
  .chart-card h3 { font-size: 14px; margin-bottom: 10px; color: #94a3b8; }
  .table-wrap { background: #1e293b; border: 1px solid #334155; border-radius: 10px; padding: 16px; margin-bottom: 20px; overflow-x: auto; }
  .table-wrap h3 { font-size: 14px; margin-bottom: 10px; color: #94a3b8; }
  table { width: 100%%; border-collapse: collapse; font-size: 13px; }
  th { text-align: left; padding: 8px 12px; border-bottom: 2px solid #334155; color: #64748b; text-transform: uppercase; font-size: 11px; letter-spacing: 1px; }
  td { padding: 8px 12px; border-bottom: 1px solid #1e293b; }
  tr:hover td { background: #1e293b; }
  .status-keep { color: #10b981; font-weight: 600; }
  .status-discard { color: #ef4444; }
  .status-crash { color: #f59e0b; }
  .status-invalid { color: #8b5cf6; }
  .status-early-stop { color: #f97316; }
  .commit-hash { font-family: monospace; color: #60a5fa; }
</style>
</head>
<body>

<div class="header">
  <div>
    <h1>Parameter Golf Dashboard</h1>
    <div class="meta">Last updated: %s | Auto-refresh: 30s</div>
  </div>
  <div>
    <span class="status-badge" style="background:%s">%s</span>
  </div>
</div>

<div class="stats">
  <div class="stat-card">
    <div class="label">Best BPB</div>
    <div class="value" style="color:#10b981">%s</div>
    <div class="sub">%s</div>
  </div>
  <div class="stat-card">
    <div class="label">Total Experiments</div>
    <div class="value">%d</div>
    <div class="sub">%d kept, %d discarded, %d crashed</div>
  </div>
  <div class="stat-card">
    <div class="label">Artifact Size</div>
    <div class="value">%.2f MB</div>
    <div class="sub">Limit: 16.00 MB (%.1f%% used)</div>
  </div>
  <div class="stat-card">
    <div class="label">Competition Target</div>
    <div class="value" style="color:#f59e0b">0.9850</div>
    <div class="sub">PR #741 (Cosine TTT + N-gram Cache)</div>
  </div>
</div>

<div class="charts">
  <div class="chart-card full">
    <h3>Experiment Results Over Time</h3>
    <div id="exp-timeline" style="height:300px"></div>
  </div>
  <div class="chart-card">
    <h3>Current Run: Training Loss</h3>
    <div id="train-loss" style="height:280px"></div>
  </div>
  <div class="chart-card">
    <h3>Current Run: TTT/Eval Progression</h3>
    <div id="ttt-progress" style="height:280px"></div>
  </div>
  <div class="chart-card">
    <h3>Loss Waterfall (Current/Last Run)</h3>
    <div id="waterfall" style="height:280px"></div>
  </div>
  <div class="chart-card">
    <h3>Artifact Size vs BPB</h3>
    <div id="size-bpb" style="height:280px"></div>
  </div>
</div>

<div class="table-wrap">
  <h3>Experiment Log</h3>
  <table>
    <thead><tr><th>#</th><th>Commit</th><th>BPB</th><th>Size (MB)</th><th>Status</th><th>Description</th></tr></thead>
    <tbody>%s</tbody>
  </table>
</div>

<div class="table-wrap">
  <h3>Recent Git Commits</h3>
  <table>
    <thead><tr><th>Hash</th><th>Message</th><th>Date</th></tr></thead>
    <tbody>%s</tbody>
  </table>
</div>

<script>
var D = %s;

var plotBg = '#1e293b', plotGrid = '#334155', plotText = '#94a3b8';
var lb = {paper_bgcolor: plotBg, plot_bgcolor: plotBg, font: {color: plotText, size: 11},
  margin: {l: 50, r: 20, t: 10, b: 40},
  xaxis: {gridcolor: plotGrid, zerolinecolor: plotGrid},
  yaxis: {gridcolor: plotGrid, zerolinecolor: plotGrid}};

// Experiment timeline
Plotly.newPlot('exp-timeline', [
  {x: D.keep.x, y: D.keep.y, text: D.keep.text, mode: 'lines+markers', name: 'Kept',
   marker: {color: '#10b981', size: 10}, line: {color: '#10b981', width: 2},
   hovertemplate: '%%{text}<br>BPB: %%{y:.4f}'},
  {x: D.discard.x, y: D.discard.y, text: D.discard.text, mode: 'markers', name: 'Discarded',
   marker: {color: '#ef4444', size: 8, symbol: 'x'},
   hovertemplate: '%%{text}<br>BPB: %%{y:.4f}'},
  {x: D.invalid.x, y: D.invalid.y, mode: 'markers', name: 'Invalid (>16MB)',
   marker: {color: '#8b5cf6', size: 8, symbol: 'diamond'}},
], Object.assign({}, lb, {
  xaxis: Object.assign({}, lb.xaxis, {title: 'Experiment #'}),
  yaxis: Object.assign({}, lb.yaxis, {title: 'val_bpb'}),
  shapes: [
    {type:'line', x0:0, x1:D.n_exp+1, y0:1.1194, y1:1.1194, line:{color:'#475569', dash:'dot', width:1}},
    {type:'line', x0:0, x1:D.n_exp+1, y0:0.9850, y1:0.9850, line:{color:'#f59e0b', dash:'dot', width:1}},
  ],
  annotations: [
    {x:D.n_exp, y:1.1194, text:'Merged SOTA (1.1194)', showarrow:false, font:{size:10, color:'#475569'}, yshift:12},
    {x:D.n_exp, y:0.9850, text:'PR #741 (0.9850)', showarrow:false, font:{size:10, color:'#f59e0b'}, yshift:12},
  ],
  showlegend: true, legend: {x:1, y:1, xanchor:'right', bgcolor:'rgba(0,0,0,0)'},
}));

// Training loss
Plotly.newPlot('train-loss', [
  {x: D.train.x, y: D.train.y, mode: 'lines', name: 'Train Loss', line: {color: '#3b82f6', width: 1.5}},
  {x: D.val.x, y: D.val.y, mode: 'markers+lines', name: 'Val BPB', line: {color: '#f59e0b', width: 2}, marker: {size: 8}, yaxis: 'y2'},
], Object.assign({}, lb, {
  xaxis: Object.assign({}, lb.xaxis, {title: 'Step'}),
  yaxis: Object.assign({}, lb.yaxis, {title: 'Train Loss', side: 'left'}),
  yaxis2: {title: 'Val BPB', overlaying: 'y', side: 'right', gridcolor: plotGrid, titlefont: {color: '#f59e0b'}, tickfont: {color: '#f59e0b'}},
  showlegend: true, legend: {x:1, y:1, xanchor:'right', bgcolor:'rgba(0,0,0,0)'},
}));

// TTT progress
Plotly.newPlot('ttt-progress', [
  {x: D.ttt.x, y: D.ttt.y, mode: 'lines', name: 'BPB', line: {color: '#10b981', width: 2},
   customdata: D.ttt.time, hovertemplate: 'Chunk %%{x}<br>BPB: %%{y:.4f}<br>Time: %%{customdata:.0f}s'},
], Object.assign({}, lb, {
  xaxis: Object.assign({}, lb.xaxis, {title: 'Eval Chunk'}),
  yaxis: Object.assign({}, lb.yaxis, {title: 'Running BPB'}),
}));

// Waterfall
var wfColors = D.waterfall.bpb.map(function(v, i) {
  return i === D.waterfall.bpb.length - 1 ? '#10b981' : (i > 0 && v < D.waterfall.bpb[i-1] ? '#3b82f6' : '#ef4444');
});
var wfMin = D.waterfall.bpb.length > 0 ? Math.min.apply(null, D.waterfall.bpb) - 0.02 : 0;
var wfMax = D.waterfall.bpb.length > 0 ? Math.max.apply(null, D.waterfall.bpb) + 0.02 : 2;
Plotly.newPlot('waterfall', [
  {x: D.waterfall.stages, y: D.waterfall.bpb, type: 'bar', marker: {color: wfColors},
   text: D.waterfall.bpb.map(function(v){return v.toFixed(4)}), textposition: 'outside', textfont: {color: plotText, size: 11}},
], Object.assign({}, lb, {
  xaxis: Object.assign({}, lb.xaxis),
  yaxis: Object.assign({}, lb.yaxis, {title: 'BPB', range: [wfMin, wfMax]}),
}));

// Size vs BPB
var sbMaxY = D.sizebpb.y.length > 0 ? Math.max.apply(null, D.sizebpb.y) : 1.3;
Plotly.newPlot('size-bpb', [
  {x: D.sizebpb.x, y: D.sizebpb.y, text: D.sizebpb.text, mode: 'markers',
   marker: {color: D.sizebpb.colors, size: 10},
   hovertemplate: '%%{text}<br>BPB: %%{y:.4f}<br>Size: %%{x:.2f} MB'},
], Object.assign({}, lb, {
  xaxis: Object.assign({}, lb.xaxis, {title: 'Artifact Size (MB)', range: [0, 17]}),
  yaxis: Object.assign({}, lb.yaxis, {title: 'val_bpb'}),
  shapes: [{type:'line', x0:16, x1:16, y0:0, y1:2, line:{color:'#ef4444', dash:'dash', width:2}}],
  annotations: [{x:16, y:sbMaxY, text:'16MB limit', showarrow:false, font:{size:10, color:'#ef4444'}, xshift:-30}],
}));
</script>

<div style="text-align:center; margin-top:20px; padding:10px; font-size:11px; color:#475569;">
  Reference: Merged SOTA 1.1194 | PR #741 0.9850 | PR #745 1.0222 | PR #740 1.0909
</div>

</body>
</html>""" % (
        now, status_color, status_text,
        best_bpb_str, best_desc_str,
        n_total, n_keep, n_discard, n_crash,
        artifact_mb, artifact_pct,
        table_rows, commit_rows, chart_data
    )

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
        pass


if __name__ == "__main__":
    print("Dashboard starting on http://localhost:%d" % PORT)
    print("Repo dir: %s" % REPO_DIR)
    print("Results: %s" % RESULTS_TSV)
    print("Run log: %s" % RUN_LOG)
    print("Auto-refreshes every 30s")
    server = http.server.HTTPServer(("0.0.0.0", PORT), DashboardHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()
