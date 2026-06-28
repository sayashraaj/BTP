from __future__ import annotations

"""
Visualization module: renders solver results as a self-contained HTML page.

Produces three panels:
1. Tour summary card — key metrics at a glance
2. Journey timeline — horizontal bar chart showing train legs, transfers, waits
3. Route map — node-link diagram of the selected tour

All rendering is done with inline SVG and CSS — no external dependencies
beyond the standard library and the project's own modules.
"""

import html
import json
import math
from pathlib import Path

from .formulation_3d import ThreeDInput, ThreeDResult
from .formulation_tetn import TETNResult
from .network import TETN
from .solver import SolverResult

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def _seconds_to_hhmm(s: int | float) -> str:
    s = int(s)
    if s < 0:
        return f"-{_seconds_to_hhmm(-s)}"
    days = s // 86400
    remainder = s % 86400
    h = remainder // 3600
    m = (remainder % 3600) // 60
    if days > 0:
        return f"Day {days + 1}, {h:02d}:{m:02d}"
    return f"{h:02d}:{m:02d}"


def _duration_str(s: int | float) -> str:
    s = int(s)
    h = s // 3600
    m = (s % 3600) // 60
    if h > 0:
        return f"{h}h {m}m"
    return f"{m}m"


def _esc(text: str) -> str:
    return html.escape(str(text))


# ---------------------------------------------------------------------------
# 3D Matrix visualization
# ---------------------------------------------------------------------------


def _render_3d_html(inp: ThreeDInput, res: ThreeDResult) -> str:
    n = inp.n
    tour = res.tour
    obj = res.objective_value

    node_names = inp.node_names or {i: f"Node {i}" for i in range(1, n + 1)}

    # Build leg data
    legs = []
    for i, j, k in res.tour_links:
        start_time = inp.Start.get((i, j, k), 0)
        cost = inp.C.get((i, j, k), 0)
        legs.append({
            "from": node_names.get(i, f"Node {i}"),
            "to": node_names.get(j, f"Node {j}"),
            "from_id": i, "to_id": j, "link_k": k,
            "depart": start_time,
            "arrive": start_time + cost,
            "travel_time": cost,
            "t_from": res.t_vals.get(i, 0),
            "t_to": res.t_vals.get(j, res.t_vals.get(n + 1, 0)),
        })

    # Summary card
    summary = {
        "Method": "3D Matrix (Chapter 2)",
        "Status": res.solver_status,
        "Nodes": n,
        "Return time": f"{obj:.0f}",
        "Tour": " → ".join(node_names.get(t, str(t)) for t in tour),
        "Solve time": f"{res.solve_time_seconds:.3f}s",
    }

    # Route diagram — circular layout
    cx, cy, r = 250, 180, 120
    positions = {}
    unique_nodes = [t for i, t in enumerate(tour) if t not in tour[:i]]
    for idx, node in enumerate(unique_nodes):
        angle = -math.pi / 2 + 2 * math.pi * idx / len(unique_nodes)
        positions[node] = (cx + r * math.cos(angle), cy + r * math.sin(angle))

    route_svg_parts = []
    # Draw links
    for leg in legs:
        fi, ti = leg["from_id"], leg["to_id"]
        if fi == ti:
            continue
        p_to = positions.get(ti, positions.get(1))
        x1, y1 = positions[fi]
        x2, y2 = p_to
        # Shorten arrow so it doesn't overlap the node circle
        dx, dy = x2 - x1, y2 - y1
        dist = math.sqrt(dx * dx + dy * dy) or 1
        ux, uy = dx / dist, dy / dist
        x1s, y1s = x1 + ux * 24, y1 + uy * 24
        x2s, y2s = x2 - ux * 24, y2 - uy * 24
        route_svg_parts.append(
            f'<line x1="{x1s:.1f}" y1="{y1s:.1f}" x2="{x2s:.1f}" y2="{y2s:.1f}" '
            f'stroke="#6366f1" stroke-width="2.5" marker-end="url(#arrowhead)"/>'
        )
    # Draw nodes
    for node, (x, y) in positions.items():
        is_origin = node == 1
        fill = "#4f46e5" if is_origin else "#ffffff"
        text_fill = "#ffffff" if is_origin else "#1e293b"
        stroke = "#4f46e5"
        route_svg_parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="22" fill="{fill}" stroke="{stroke}" stroke-width="2.5"/>'
        )
        name = _esc(node_names.get(node, str(node)))
        route_svg_parts.append(
            f'<text x="{x:.1f}" y="{y + 5:.1f}" text-anchor="middle" '
            f'fill="{text_fill}" font-size="11" font-weight="600">{name}</text>'
        )

    route_svg = (
        '<svg viewBox="0 0 500 370" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:500px">'
        '<defs><marker id="arrowhead" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">'
        '<polygon points="0 0, 8 3, 0 6" fill="#6366f1"/></marker></defs>'
        + "\n".join(route_svg_parts)
        + "</svg>"
    )

    # Legs table
    legs_rows = ""
    for leg in legs:
        legs_rows += (
            f'<tr><td>{_esc(leg["from"])}</td><td>{_esc(leg["to"])}</td>'
            f'<td class="mono">k={leg["link_k"]}</td>'
            f'<td class="mono">{leg["depart"]:.0f}</td>'
            f'<td class="mono">{leg["arrive"]:.0f}</td>'
            f'<td class="mono">{leg["travel_time"]:.0f}</td></tr>'
        )

    # u/t value table
    ut_rows = ""
    for node in unique_nodes:
        name = _esc(node_names.get(node, str(node)))
        u_val = res.u_vals.get(node, "-")
        t_val = res.t_vals.get(node, "-")
        u_str = f"{u_val:.0f}" if isinstance(u_val, float) else str(u_val)
        t_str = f"{t_val:.1f}" if isinstance(t_val, float) else str(t_val)
        ut_rows += f'<tr><td>{name}</td><td class="mono">{u_str}</td><td class="mono">{t_str}</td></tr>'
    # Add return
    ret_t = res.t_vals.get(n + 1, "-")
    ret_str = f"{ret_t:.1f}" if isinstance(ret_t, float) else str(ret_t)
    ut_rows += f'<tr><td>Return to Origin</td><td class="mono">-</td><td class="mono">{ret_str}</td></tr>'

    return summary, route_svg, legs_rows, ut_rows


# ---------------------------------------------------------------------------
# TETN visualization
# ---------------------------------------------------------------------------


def _render_tetn_html(
    net: TETN, res: TETNResult, solver_result: SolverResult | None = None
) -> str:
    details = res.tour_details
    if not details:
        return {}, "", "", ""

    # Group consecutive nodes into legs (same train = one leg)
    legs = []
    current_leg = None
    for d in details:
        if current_leg is None or d["train_id"] != current_leg["train_id"]:
            if current_leg is not None:
                legs.append(current_leg)
            current_leg = {
                "train_id": d["train_id"],
                "train_name": d["train_name"],
                "stations": [d],
            }
        else:
            current_leg["stations"].append(d)
    if current_leg:
        legs.append(current_leg)

    # Build timeline segments
    segments = []
    unique_stations = []
    seen_stations = set()
    for leg in legs:
        first = leg["stations"][0]
        last = leg["stations"][-1]
        for s in leg["stations"]:
            sname = s["station_name"]
            if sname not in seen_stations:
                unique_stations.append(sname)
                seen_stations.add(sname)
        segments.append({
            "train_name": leg["train_name"],
            "train_id": leg["train_id"],
            "from_station": first["station_name"],
            "to_station": last["station_name"],
            "depart_time": first["time_seconds"],
            "arrive_time": last["time_seconds"],
            "n_stops": len(leg["stations"]),
            "type": "travel",
        })

    # Insert wait segments between legs
    full_segments = []
    for i, seg in enumerate(segments):
        if i > 0:
            prev = segments[i - 1]
            wait = seg["depart_time"] - prev["arrive_time"]
            if wait > 0:
                full_segments.append({
                    "train_name": "Transfer / Wait",
                    "train_id": "",
                    "from_station": prev["to_station"],
                    "to_station": seg["from_station"],
                    "depart_time": prev["arrive_time"],
                    "arrive_time": seg["depart_time"],
                    "n_stops": 0,
                    "type": "wait",
                })
        full_segments.append(seg)

    # Timeline SVG
    if full_segments:
        t_min = full_segments[0]["depart_time"]
        t_max = full_segments[-1]["arrive_time"]
        t_range = t_max - t_min or 1
    else:
        t_min = t_max = t_range = 1

    bar_height = 36
    bar_gap = 6
    label_width = 200
    chart_width = 520
    total_width = label_width + chart_width + 40
    svg_height = 60 + (bar_height + bar_gap) * len(full_segments) + 50

    timeline_parts = []

    # Time axis
    n_ticks = min(6, max(2, t_range // 3600))
    for i in range(n_ticks + 1):
        t = t_min + (t_range * i / n_ticks)
        x = label_width + 20 + (chart_width * i / n_ticks)
        timeline_parts.append(
            f'<line x1="{x:.1f}" y1="40" x2="{x:.1f}" y2="{svg_height - 20}" '
            f'stroke="#cbd5e1" stroke-width="1" stroke-dasharray="4,4"/>'
        )
        timeline_parts.append(
            f'<text x="{x:.1f}" y="32" text-anchor="middle" fill="#64748b" font-size="10">'
            f'{_esc(_seconds_to_hhmm(t))}</text>'
        )

    # Bars
    colors_travel = ["#6366f1", "#8b5cf6", "#a78bfa", "#7c3aed", "#4f46e5"]
    travel_idx = 0
    for idx, seg in enumerate(full_segments):
        y = 50 + idx * (bar_height + bar_gap)
        x_start = label_width + 20 + chart_width * (seg["depart_time"] - t_min) / t_range
        x_end = label_width + 20 + chart_width * (seg["arrive_time"] - t_min) / t_range
        bar_w = max(x_end - x_start, 6)

        if seg["type"] == "wait":
            fill = "#fbbf24"
            text_color = "#92400e"
            label = f"Wait at {seg['from_station'].split()[0]}"
        else:
            fill = colors_travel[travel_idx % len(colors_travel)]
            text_color = "#ffffff"
            label = seg["train_name"][:28]
            travel_idx += 1

        # Left label
        timeline_parts.append(
            f'<text x="{label_width + 10}" y="{y + bar_height / 2 + 4}" '
            f'text-anchor="end" fill="{"#92400e" if seg["type"] == "wait" else "#475569"}" '
            f'font-size="10" font-style="{"italic" if seg["type"] == "wait" else "normal"}">'
            f'{_esc(label)}</text>'
        )

        # Bar
        timeline_parts.append(
            f'<rect x="{x_start:.1f}" y="{y}" width="{bar_w:.1f}" height="{bar_height}" '
            f'rx="4" fill="{fill}" opacity="0.85"/>'
        )

        # Duration label — inside if bar is wide enough, to the right otherwise
        dur = seg["arrive_time"] - seg["depart_time"]
        dur_text = _esc(_duration_str(dur))
        if bar_w > 60:
            timeline_parts.append(
                f'<text x="{x_start + bar_w / 2:.1f}" y="{y + bar_height / 2 + 4}" '
                f'text-anchor="middle" fill="{text_color}" font-size="10" font-weight="500">{dur_text}</text>'
            )
        else:
            timeline_parts.append(
                f'<text x="{x_start + bar_w + 6:.1f}" y="{y + bar_height / 2 + 4}" '
                f'text-anchor="start" fill="{text_color}" font-size="10" font-weight="500">{dur_text}</text>'
            )

    timeline_svg = (
        f'<svg viewBox="0 0 {total_width} {svg_height}" xmlns="http://www.w3.org/2000/svg" '
        f'style="width:100%;max-width:{total_width}px">'
        + "\n".join(timeline_parts)
        + "</svg>"
    )

    # Summary
    total_time = (details[-1]["time_seconds"] - details[0]["time_seconds"]) if len(details) > 1 else 0
    n_transfers = sum(1 for s in full_segments if s["type"] == "wait")
    total_wait = sum(s["arrive_time"] - s["depart_time"] for s in full_segments if s["type"] == "wait")
    travel_legs = [s for s in full_segments if s["type"] == "travel"]

    summary = {
        "Method": "TETN (Chapter 3)",
        "Status": res.solver_status,
        "Total duration": _duration_str(total_time),
        "Train legs": str(len(travel_legs)),
        "Transfers": str(n_transfers),
        "Wait time": _duration_str(total_wait) if total_wait > 0 else "None",
        "Objective": f"{res.objective_value:.0f}s",
        "Solve time": f"{res.solve_time_seconds:.1f}s",
    }

    if solver_result and solver_result.network_stats:
        ns = solver_result.network_stats
        summary["Pruning"] = f"{ns.get('original_trains', '?')} → {ns.get('pruned_trains', '?')} trains ({ns.get('reduction_pct', '?')}%)"
        summary["Network"] = f"{ns.get('final_nodes', '?')} nodes, {ns.get('final_links', '?')} links"

    # Itinerary table
    itin_rows = ""
    for seg in full_segments:
        if seg["type"] == "wait":
            dur = seg["arrive_time"] - seg["depart_time"]
            itin_rows += (
                f'<tr class="wait-row"><td colspan="2">⏳ Wait at {_esc(seg["from_station"])}</td>'
                f'<td class="mono">{_esc(_seconds_to_hhmm(seg["depart_time"]))}</td>'
                f'<td class="mono">{_esc(_seconds_to_hhmm(seg["arrive_time"]))}</td>'
                f'<td class="mono">{_esc(_duration_str(dur))}</td></tr>'
            )
        else:
            dur = seg["arrive_time"] - seg["depart_time"]
            itin_rows += (
                f'<tr><td>{_esc(seg["train_name"])}<br><span class="train-id">#{_esc(seg["train_id"])}</span></td>'
                f'<td>{_esc(seg["from_station"])} → {_esc(seg["to_station"])}</td>'
                f'<td class="mono">{_esc(_seconds_to_hhmm(seg["depart_time"]))}</td>'
                f'<td class="mono">{_esc(_seconds_to_hhmm(seg["arrive_time"]))}</td>'
                f'<td class="mono">{_esc(_duration_str(dur))}</td></tr>'
            )

    # Station stops table
    stops_rows = ""
    for d in details:
        stops_rows += (
            f'<tr><td class="mono">{d["node_id"]}</td>'
            f'<td>{_esc(d["station_name"])}</td>'
            f'<td>{_esc(d["train_name"])}</td>'
            f'<td class="mono">{_esc(_seconds_to_hhmm(d["time_seconds"]))}</td>'
            f'<td><span class="badge badge-{d["node_type"].lower()}">{d["node_type"]}</span></td></tr>'
        )

    return summary, timeline_svg, itin_rows, stops_rows


# ---------------------------------------------------------------------------
# HTML assembly
# ---------------------------------------------------------------------------

_CSS = """
:root {
  --bg: #f1f5f9; --surface: #ffffff; --surface2: #d1d9e6;
  --text: #0f172a; --text-muted: #475569; --accent: #4338ca;
  --accent2: #6d28d9; --warn: #b45309; --success: #047857;
  --border: #94a3b8; --font: 'Inter', -apple-system, system-ui, sans-serif;
  --mono: 'JetBrains Mono', 'Fira Code', monospace;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body { background: var(--bg); color: var(--text); font-family: var(--font); padding: 32px; max-width: 960px; margin: 0 auto; }
h1 { font-size: 22px; font-weight: 700; margin-bottom: 4px; }
.subtitle { color: var(--text-muted); font-size: 13px; margin-bottom: 28px; }
h2 { font-size: 15px; font-weight: 600; color: var(--accent2); text-transform: uppercase; letter-spacing: 1.2px; margin: 32px 0 14px; }
.card { background: var(--surface); border-radius: 10px; padding: 20px 24px; margin-bottom: 20px; border: 1px solid var(--surface2); box-shadow: 0 1px 3px rgba(0,0,0,0.06); }
.summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 14px; }
.summary-item label { display: block; font-size: 10px; text-transform: uppercase; letter-spacing: 1px; color: var(--text-muted); margin-bottom: 3px; }
.summary-item span { font-size: 15px; font-weight: 600; }
.summary-item span.tour-path { font-size: 13px; font-weight: 500; word-break: break-word; }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th { text-align: left; padding: 8px 10px; color: var(--text-muted); font-size: 10px; text-transform: uppercase; letter-spacing: 1px; border-bottom: 1px solid var(--surface2); }
td { padding: 8px 10px; border-bottom: 1px solid #e2e8f0; vertical-align: middle; }
tr:last-child td { border-bottom: none; }
.mono { font-family: var(--mono); font-size: 12px; }
.wait-row { background: #fffbeb; }
.wait-row td { color: #92400e; font-style: italic; }
.train-id { font-size: 11px; color: var(--text-muted); font-family: var(--mono); }
.badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 10px; font-weight: 600; letter-spacing: 0.5px; }
.badge-arr { background: #d1fae5; color: #065f46; }
.badge-dep { background: #e0e7ff; color: #3730a3; }
.chart-container { display: flex; justify-content: center; padding: 12px 0; overflow-x: auto; }
.two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
@media (max-width: 700px) { .two-col { grid-template-columns: 1fr; } body { padding: 16px; } }
.footer { text-align: center; color: var(--text-muted); font-size: 11px; margin-top: 40px; padding-top: 20px; border-top: 1px solid var(--surface2); }
.route-banner { display: flex; align-items: center; gap: 0; flex-wrap: wrap; padding: 16px 20px; margin-bottom: 20px; }
.route-node { display: flex; flex-direction: column; align-items: center; gap: 2px; }
.route-node .role { font-size: 9px; text-transform: uppercase; letter-spacing: 1.2px; font-weight: 700; }
.route-node .role-origin { color: var(--accent); }
.route-node .role-via { color: var(--warn); }
.route-node .role-dest { color: var(--success); }
.route-node .station-name { font-size: 14px; font-weight: 700; color: var(--text); }
.route-arrow { font-size: 18px; color: var(--border); margin: 0 10px; padding-top: 10px; }
"""


def render_html(solver_result: SolverResult, inp: ThreeDInput | None = None, net: TETN | None = None) -> str:
    """Render a solver result as a self-contained HTML page."""
    r = solver_result.result
    is_3d = isinstance(r, ThreeDResult)

    if is_3d and inp is not None:
        summary, route_svg, legs_rows, ut_rows = _render_3d_html(inp, r)
    elif isinstance(r, TETNResult) and net is not None:
        summary, timeline_svg, itin_rows, stops_rows = _render_tetn_html(net, r, solver_result)
    elif isinstance(r, TETNResult):
        summary, timeline_svg, itin_rows, stops_rows = _render_tetn_html(TETN(), r, solver_result)
    else:
        summary = {"Error": "Cannot render — missing input data"}
        is_3d = True
        route_svg = legs_rows = ut_rows = ""

    # Build route banner
    query = solver_result.query
    route_banner = ""
    if query:
        origin = query.get("origin", "")
        intermediates = query.get("intermediates", [])
        destination = query.get("destination", origin)

        parts = []
        parts.append(
            f'<div class="route-node"><span class="role role-origin">Origin</span>'
            f'<span class="station-name">{_esc(origin)}</span></div>'
        )
        for station in intermediates:
            parts.append('<span class="route-arrow">→</span>')
            parts.append(
                f'<div class="route-node"><span class="role role-via">Via</span>'
                f'<span class="station-name">{_esc(station)}</span></div>'
            )
        parts.append('<span class="route-arrow">→</span>')
        if destination == origin:
            parts.append(
                f'<div class="route-node"><span class="role role-dest">Return</span>'
                f'<span class="station-name">{_esc(destination)}</span></div>'
            )
        else:
            parts.append(
                f'<div class="route-node"><span class="role role-dest">Destination</span>'
                f'<span class="station-name">{_esc(destination)}</span></div>'
            )
        route_banner = f'<div class="card route-banner">{"".join(parts)}</div>'

    # Build summary card
    summary_html = ""
    for label, value in summary.items():
        css_class = ' class="tour-path"' if "tour" in label.lower() else ""
        summary_html += f'<div class="summary-item"><label>{_esc(label)}</label><span{css_class}>{_esc(value)}</span></div>'

    # Assemble body
    if is_3d:
        body = f"""
        <h1>Journey Planner — Solution</h1>
        <p class="subtitle">3D Matrix IP Formulation (Chapter 2)</p>

        {route_banner}
        <div class="card"><div class="summary-grid">{summary_html}</div></div>

        <div class="two-col">
            <div>
                <h2>Route Diagram</h2>
                <div class="card"><div class="chart-container">{route_svg}</div></div>
            </div>
            <div>
                <h2>Visit Order &amp; Timing</h2>
                <div class="card">
                    <table>
                        <thead><tr><th>Node</th><th>u (rank)</th><th>t (exit time)</th></tr></thead>
                        <tbody>{ut_rows}</tbody>
                    </table>
                </div>
            </div>
        </div>

        <h2>Selected Links</h2>
        <div class="card">
            <table>
                <thead><tr><th>From</th><th>To</th><th>Link</th><th>Depart</th><th>Arrive</th><th>Cost</th></tr></thead>
                <tbody>{legs_rows}</tbody>
            </table>
        </div>
        """
    else:
        body = f"""
        <h1>Journey Planner — Solution</h1>
        <p class="subtitle">Time-Expanded Transit Network (Chapter 3)</p>

        {route_banner}
        <div class="card"><div class="summary-grid">{summary_html}</div></div>

        <h2>Journey Timeline</h2>
        <div class="card"><div class="chart-container">{timeline_svg}</div></div>

        <h2>Itinerary</h2>
        <div class="card">
            <table>
                <thead><tr><th>Train</th><th>Route</th><th>Depart</th><th>Arrive</th><th>Duration</th></tr></thead>
                <tbody>{itin_rows}</tbody>
            </table>
        </div>

        <h2>All Stops</h2>
        <div class="card">
            <table>
                <thead><tr><th>Node</th><th>Station</th><th>Train</th><th>Time</th><th>Type</th></tr></thead>
                <tbody>{stops_rows}</tbody>
            </table>
        </div>
        """

    timing_parts = ""
    if solver_result.timing:
        for stage, secs in solver_result.timing.items():
            timing_parts += f"{stage}: {secs:.2f}s &nbsp;|&nbsp; "
        timing_parts = timing_parts.rstrip(" &nbsp;|&nbsp; ")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Journey Planner — Solution</title>
<style>{_CSS}</style>
</head>
<body>
{body}
<div class="footer">Journey Planner IP &nbsp;·&nbsp; {timing_parts}</div>
</body>
</html>"""


def save_html(
    solver_result: SolverResult,
    output_path: Path | str | None = None,
    inp: ThreeDInput | None = None,
    net: TETN | None = None,
) -> Path:
    """Render and save the visualization to an HTML file."""
    if output_path is None:
        output_path = RESULTS_DIR / "solution.html"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    html_content = render_html(solver_result, inp=inp, net=net)
    output_path.write_text(html_content, encoding="utf-8")
    return output_path
