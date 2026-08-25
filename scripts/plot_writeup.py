"""Render the writeup figures from the collected CSV files with Plotly Express.

Run ``collect_writeup_data.py`` first when raw artifacts have changed. Plotting
depends only on the committed CSVs, so it also works on a checkout without the
gitignored multi-gigabyte experiment artifacts.

Each figure is written as an interactive HTML file and as static SVG and PNG
assets for embedding in Markdown.

Usage:

    uv run --extra plots python scripts/plot_writeup.py
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import plotly.express as px
from plotly.graph_objects import Figure
from plotly.subplots import make_subplots

DDP_COLOR = "#146B8C"
DILOCO_COLOR = "#D46A1F"
GRAY = "#777777"
TCP_COLOR = "#9B4D96"
COLORS = {
    "DDP (8xA100, NVLink)": DDP_COLOR,
    "DiLoCo (8xA100, NVLink)": DILOCO_COLOR,
}
WIDTH = 864
HEIGHT = 516

# The transport rows all describe the default 162,220,800-parameter model at
# seq-1024 and global batch 480 on eight A100-SXM4-40GB GPUs.  Keep the MFU
# calculation identical to src/distrain/mfu.py and docs/decisions.md section 3:
# PaLM-style model FLOPs (including attention) divided by the measured dense
# bf16 roofline.  Reconstructed upper-bound step times consequently produce
# lower-bound MFU values.
#
# N is the 123,587,328 matmul parameters (GPT.flops_params()), not all 162M: the
# untied wte is a gather.  The pre-2026-08-22 value here was 1,086,571,008, which
# charged 6N for it and inflated every MFU below by 27% (decisions.md section 23).
MODEL_FLOPS_PER_TOKEN = 854_770_176
TOKENS_PER_STEP = 480 * 1024
A100_SXM4_40GB_TFLOPS = 270.1
TRANSPORT_GPUS = 8

import plotly.io as pio

pio.templates.default = "simple_white"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def finish(fig: Figure, path: Path, *, height: int = HEIGHT) -> None:
    """Apply shared styling and write interactive and static versions."""
    fig.update_layout(
        template="plotly_white",
        width=WIDTH,
        height=height,
        margin={"l": 75, "r": 35, "t": 75, "b": 65},
        font={"family": "Arial, sans-serif", "size": 13, "color": "#222222"},
        title={"x": 0.5, "xanchor": "center"},
        legend={"title_text": "", "orientation": "h", "yanchor": "bottom", "y": 1.01, "x": 0},
        hoverlabel={"font_size": 13},
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(0, 0, 0, 0.10)", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0, 0, 0, 0.10)", zeroline=False)
    # fig.write_html(path.with_suffix(".html"), include_plotlyjs="cdn", full_html=True)
    fig.write_image(path.with_suffix(".svg"))
    # fig.write_image(path.with_suffix(".png"), scale=2)


def plot_validation(data_dir: Path, plots_dir: Path) -> None:
    rows = read_csv(data_dir / "validation_curves.csv")
    data: dict[str, list[Any]] = {
        "Series": [row["series"] for row in rows],
        "Training tokens (billions)": [float(row["tokens_billions"]) for row in rows],
        "Validation loss": [float(row["val_loss"]) for row in rows],
        "Training step": [int(row["step"]) for row in rows],
    }
    fig = px.line(
        data,
        x="Training tokens (billions)",
        y="Validation loss",
        color="Series",
        color_discrete_map=COLORS,
        markers=True,
        custom_data=["Training step"],
        title="Equal-token training quality on 8×A100",
    )
    for trace in fig.data:
        trace.name = trace.name.split(" (")[0]
        trace.update(
            line_width=2.5,
            marker_size=5,
            hovertemplate=(
                "%{fullData.name}<br>Tokens: %{x:.3f}B<br>"
                "Validation loss: %{y:.4f}<br>Step: %{customdata[0]}<extra></extra>"
            ),
        )
    fig.add_hline(
        y=3.28,
        line_dash="dash",
        line_color="#333333",
        line_width=1.2,
        annotation_text="3.28 target",
        annotation_position="top left",
    )
    fig.update_yaxes(range=[3.2, 7.0])
    finish(fig, plots_dir / "validation_loss_vs_tokens")


def plot_scaling(data_dir: Path, plots_dir: Path) -> None:
    rows = [row for row in read_csv(data_dir / "scaling.csv") if row["step_ms"]]
    labels = [row["config"] for row in rows]
    times = [float(row["step_ms"]) for row in rows]
    data: dict[str, list[Any]] = {
        "Configuration": labels,
        "Optimizer step time (ms)": times,
        "Standard deviation (ms)": [float(row["step_std_ms"]) for row in rows],
    }
    color_map = {labels[0]: GRAY, labels[1]: DDP_COLOR, labels[2]: TCP_COLOR}
    fig = px.bar(
        data,
        x="Configuration",
        y="Optimizer step time (ms)",
        color="Configuration",
        color_discrete_map=color_map,
        error_y="Standard deviation (ms)",
        text="Optimizer step time (ms)",
        title="Matched-micro-batch scaling at global batch 480",
    )
    fig.update_traces(
        texttemplate="%{y:.0f} ms",
        textposition="outside",
        cliponaxis=False,
        hovertemplate="%{x}<br>Step time: %{y:.1f} ms<extra></extra>",
    )
    fig.update_layout(showlegend=False)
    fig.update_yaxes(range=[0, max(times) * 1.18])
    fig.add_annotation(
        x=0.98,
        y=0.96,
        xref="paper",
        yref="paper",
        text="NVLink: 7.66× speedup, 95.8% efficiency",
        showarrow=False,
        xanchor="right",
        yanchor="top",
        font={"color": DDP_COLOR},
    )
    finish(fig, plots_dir / "matched_batch_scaling")


def plot_time_to_target(data_dir: Path, plots_dir: Path) -> None:
    """Compare the defensible 1- and 8-GPU time-to-target estimates.

    Both bars use measured steady-state step times at the same global batch and
    per-device micro-batch, multiplied by the independently observed step-9999
    crossing.  The directly converged 8-GPU run used different chunking and is
    therefore deliberately excluded from this scaling comparison.
    """
    wanted = {"1xA100", "8xA100 NVLink"}
    rows = [
        row
        for row in read_csv(data_dir / "scaling.csv")
        if row["config"] in wanted and row["time_to_3_28_hours"]
    ]
    rows.sort(key=lambda row: int(row["gpus"]))
    labels = [row["config"] for row in rows]
    hours = [float(row["time_to_3_28_hours"]) for row in rows]
    data: dict[str, list[Any]] = {
        "Configuration": labels,
        "Time to validation loss 3.28 (hours)": hours,
    }
    fig = px.bar(
        data,
        x="Configuration",
        y="Time to validation loss 3.28 (hours)",
        color="Configuration",
        color_discrete_map={labels[0]: GRAY, labels[1]: DDP_COLOR},
        text="Time to validation loss 3.28 (hours)",
        title="Time to target at global batch 480",
    )
    fig.update_traces(
        texttemplate="%{y:.2f} h",
        textposition="outside",
        cliponaxis=False,
        hovertemplate="%{x}<br>Time to 3.28: %{y:.3f} h<extra></extra>",
    )
    fig.update_layout(showlegend=False)
    fig.update_yaxes(range=[0, max(hours) * 1.18])
    fig.add_annotation(
        x=0.98,
        y=0.96,
        xref="paper",
        yref="paper",
        text="7.66× speedup; both bars extrapolate measured step times",
        showarrow=False,
        xanchor="right",
        yanchor="top",
        font={"color": DDP_COLOR},
    )
    finish(fig, plots_dir / "time_to_target_scaling")


def plot_transport(data_dir: Path, plots_dir: Path) -> None:
    rows = read_csv(data_dir / "transport.csv")
    ordered = sorted(rows, key=lambda row: float(row["effective_bus_gbps"]))
    status_labels = {
        "measured_step_time": "Measured step time",
        "reconstructed_upper_bound": "Reconstructed upper bound",
    }
    data: dict[str, list[Any]] = {
        "Transport": [row["transport"] for row in rows],
        "Effective all-reduce bus bandwidth (GB/s)": [
            float(row["effective_bus_gbps"]) for row in rows
        ],
        "Time to 3.28 (hours)": [float(row["time_to_3_28_hours"]) for row in rows],
        "Evidence": [status_labels[row["step_status"]] for row in rows],
        "Step time (ms)": [float(row["step_ms_batch480"]) for row in rows],
    }
    fig = px.scatter(
        data,
        x="Effective all-reduce bus bandwidth (GB/s)",
        y="Time to 3.28 (hours)",
        color="Evidence",
        symbol="Evidence",
        text="Transport",
        color_discrete_map={
            "Measured step time": DDP_COLOR,
            "Reconstructed upper bound": DILOCO_COLOR,
        },
        symbol_map={
            "Measured step time": "circle",
            "Reconstructed upper bound": "triangle-up-open",
        },
        custom_data=["Transport", "Step time (ms)"],
        log_x=True,
        log_y=True,
        title="Transport sensitivity of 8-GPU DDP",
    )
    fig.add_scatter(
        x=[float(row["effective_bus_gbps"]) for row in ordered],
        y=[float(row["time_to_3_28_hours"]) for row in ordered],
        mode="lines",
        line={"color": GRAY, "width": 1},
        opacity=0.6,
        showlegend=False,
        hoverinfo="skip",
    )
    fig.update_traces(
        selector={"mode": "markers+text"},
        marker_size=11,
        marker_line_width=2,
        textposition="top right",
        overwrite=False,
        textfont_size=11,
        hovertemplate=(
            "%{customdata[0]}<br>Effective bus bandwidth: %{x:.3f} GB/s<br>"
            "Time to 3.28: %{y:.3f} h<br>Step time: %{customdata[1]:.1f} ms<extra></extra>"
        ),
    )
    # Where eight GPUs stop being worth renting at all: the single-GPU time, and
    # the bandwidth at which an 8-GPU DDP step costs the same as a 1-GPU step.
    single = next(
        row for row in read_csv(data_dir / "scaling.csv") if row["config"] == "1xA100"
    )
    single_hours = float(single["time_to_3_28_hours"])
    breakeven = next(
        row
        for row in read_csv(data_dir / "transport_crossovers.csv")
        if row["crossover"].startswith("8-GPU DDP")
    )
    fig.add_hline(
        y=single_hours,
        line_dash="dash",
        line_color="#333333",
        line_width=1.2,
    )
    # add_hline places its own annotation in data units without log-transforming
    # them, which puts it at 10^7.19 on this axis. Position it explicitly.
    fig.add_annotation(
        x=0.01,
        y=math.log10(single_hours),
        xref="paper",
        yref="y",
        text=f"1×A100 alone: {single_hours:.2f} h",
        showarrow=False,
        xanchor="left",
        yanchor="bottom",
        font={"size": 12, "color": "#333333"},
    )
    fig.add_vline(
        x=float(breakeven["effective_bus_gbps"]),
        line_dash="dot",
        line_color="#333333",
        line_width=1.2,
    )
    fig.add_annotation(
        x=0.01,
        y=0.01,
        xref="paper",
        yref="paper",
        text=(
            "Left of the dotted line, eight A100s on DDP are slower than one."
            "<br>All target times use the measured step-9999 crossing."
        ),
        showarrow=False,
        xanchor="left",
        yanchor="bottom",
        font={"size": 11, "color": "#555555"},
    )
    fig.update_xaxes(
        tickmode="array",
        tickvals=[0.1, 1, 10, 100],
        ticktext=["0.1", "1", "10", "100"],
    )
    fig.update_traces(selector={"name": "Measured step time"}, textposition="bottom right")
    fig.update_traces(
        selector={"name": "Reconstructed upper bound"},
        textposition=["bottom right", "top right"],
    )
    # An annotation outside the data would otherwise autorange the log axis wide.
    fig.update_yaxes(
        range=[math.log10(0.7), math.log10(30)],
        tickmode="array",
        tickvals=[1, 2, 5, 10, 20],
        ticktext=["1", "2", "5", "10", "20"],
    )
    finish(fig, plots_dir / "transport_sensitivity")


def plot_transport_mfu(data_dir: Path, plots_dir: Path) -> None:
    rows = read_csv(data_dir / "transport.csv")
    status_labels = {
        "measured_step_time": "From measured step time",
        "reconstructed_upper_bound": "Lower bound from reconstructed time",
    }
    mfu = [
        100
        * MODEL_FLOPS_PER_TOKEN
        * TOKENS_PER_STEP
        / (float(row["step_ms_batch480"]) / 1000)
        / (TRANSPORT_GPUS * A100_SXM4_40GB_TFLOPS * 1e12)
        for row in rows
    ]
    data: dict[str, list[Any]] = {
        "Transport": [row["transport"] for row in rows],
        "MFU (%)": mfu,
        "Evidence": [status_labels[row["step_status"]] for row in rows],
        "Step time (ms)": [float(row["step_ms_batch480"]) for row in rows],
    }
    fig = px.bar(
        data,
        x="Transport",
        y="MFU (%)",
        color="Evidence",
        color_discrete_map={
            "From measured step time": DDP_COLOR,
            "Lower bound from reconstructed time": DILOCO_COLOR,
        },
        pattern_shape="Evidence",
        pattern_shape_map={
            "From measured step time": "",
            "Lower bound from reconstructed time": "/",
        },
        text="MFU (%)",
        custom_data=["Step time (ms)", "Evidence"],
        title="Effective model FLOPs utilization across transports",
    )
    fig.update_traces(
        texttemplate="%{y:.1f}%",
        textposition="outside",
        cliponaxis=False,
        hovertemplate=(
            "%{x}<br>MFU: %{y:.2f}%<br>Step time: %{customdata[0]:.1f} ms"
            "<br>%{customdata[1]}<extra></extra>"
        ),
    )
    # One A100 on its own, same batch and same roofline: the bar every 8-GPU
    # configuration here is really competing with.
    single = next(
        row for row in read_csv(data_dir / "scaling.csv") if row["config"] == "1xA100"
    )
    single_mfu = (
        100
        * MODEL_FLOPS_PER_TOKEN
        * TOKENS_PER_STEP
        / (float(single["step_ms"]) / 1000)
        / A100_SXM4_40GB_TFLOPS
        / 1e12
    )
    fig.add_hline(
        y=single_mfu,
        line_dash="dash",
        line_color="#333333",
        line_width=1.2,
        annotation_text=f"1×A100 alone: {single_mfu:.1f}%",
        annotation_position="top right",
    )
    fig.update_yaxes(range=[0, max(mfu + [single_mfu]) * 1.2])
    finish(fig, plots_dir / "transport_mfu")


def plot_equal_token_runtime(data_dir: Path, plots_dir: Path) -> None:
    rows = read_csv(data_dir / "validation_curves.csv")
    final_rows: dict[str, dict[str, str]] = {}
    for row in rows:
        previous = final_rows.get(row["series"])
        if previous is None or int(row["step"]) > int(previous["step"]):
            final_rows[row["series"]] = row
    selected = [final_rows[series] for series in COLORS if series in final_rows]
    minutes = [float(row["train_time_s"]) / 60 for row in selected]
    data: dict[str, list[Any]] = {
        "Method": [row["series"].split(" (")[0] for row in selected],
        "Training time (minutes)": minutes,
        "Validation loss": [float(row["val_loss"]) for row in selected],
        "Training step": [int(row["step"]) for row in selected],
    }
    method_colors = {row["series"].split(" (")[0]: COLORS[row["series"]] for row in selected}
    fig = px.bar(
        data,
        x="Method",
        y="Training time (minutes)",
        color="Method",
        color_discrete_map=method_colors,
        text="Training time (minutes)",
        custom_data=["Validation loss", "Training step"],
        title="Wall clock at an equal 4.915B-token budget",
    )
    fig.update_traces(
        texttemplate="%{y:.1f} min",
        textposition="outside",
        cliponaxis=False,
        hovertemplate=(
            "%{x}<br>Training time: %{y:.2f} min<br>Validation loss: "
            "%{customdata[0]:.4f}<br>Step: %{customdata[1]}<extra></extra>"
        ),
    )
    fig.update_layout(showlegend=False)
    fig.update_yaxes(range=[0, max(minutes) * 1.2])
    fig.add_annotation(
        x=0.98,
        y=0.96,
        xref="paper",
        yref="paper",
        text="DiLoCo ends at 3.5183; this is not time to convergence",
        showarrow=False,
        xanchor="right",
        yanchor="top",
        font={"color": DILOCO_COLOR},
    )
    finish(fig, plots_dir / "equal_token_runtime")


def plot_ddp_modes(data_dir: Path, plots_dir: Path) -> None:
    """The DDP implementations, with the spread that decides what is real.

    The top panel draws every timed step, not a mean with an error bar: at 15
    steps over a jittery socket transport the distributions have heavy right
    tails, and the *mean* ranking of the three bucketing modes is an artifact of
    those tails -- their minima agree to about 2%.  The bottom panel puts the
    anchor batch on all three measured fabrics, because that is where the only
    reproducible finding lives: the sign of the interleaved-vs-PyTorch gap
    flips with bandwidth (decisions.md section 26).

    PyTorch DDP is drawn in brick rather than the transport purple used
    elsewhere: purple means "forced TCP" in this figure's own x axis, and
    teal/purple is the one pair here that a protanope cannot separate
    (ΔE 3.6).  Teal/brick is ΔE 13.4.  Naive keeps the neutral gray -- it is a
    strawman, and every series in both panels is direct-labelled, so no
    identity in this figure rests on colour alone.
    """
    steps = read_csv(data_dir / "ddp_mode_steps.csv")
    summary = read_csv(data_dir / "ddp_modes.csv")
    order = ["Naive", "Bucketed", "Interleaved", "PyTorch DDP"]
    colors = {"Naive": GRAY, "Bucketed": "#6A8E3A", "Interleaved": DDP_COLOR,
              "PyTorch DDP": "#A63A3A"}

    fig = make_subplots(
        rows=2,
        cols=1,
        vertical_spacing=0.14,
        row_heights=[0.58, 0.42],
        subplot_titles=(
            "Every timed step, global batch 64 over forced TCP/loopback",
            "Anchor batch 480, by measured all-reduce bandwidth — % is PyTorch DDP against interleaved",
        ),
    )
    for label in order:
        # ddp_mode_steps.csv carries every measured (mode, transport, batch);
        # this panel is the batch-64 matrix only.
        values = [
            float(row["step_ms"])
            for row in steps
            if row["label"] == label and row["global_batch_seqs"] == "64"
        ]
        fig.add_box(
            y=values,
            name=label,
            marker_color=colors[label],
            boxpoints="all",
            jitter=0.55,
            pointpos=0,
            marker_size=6,
            line_width=1.5,
            showlegend=False,
            hovertemplate="%{fullData.name}<br>%{y:.0f} ms<extra></extra>",
            row=1,
            col=1,
        )

    # Ordered by the measured effective all-reduce bandwidth in transport.csv,
    # which is the axis the finding runs along.  Rank counts differ (the PCIe
    # box was rentable at 2 GPUs only), so the tick labels carry them: compare
    # the two bars inside a fabric, never bar heights across fabrics.
    anchor = [row for row in summary if row["global_batch_seqs"] == "480"]
    transports = [
        ("NVLink NV12", "NVLink NV12<br>8 ranks · 151 GB/s"),
        ("A100 PCIe (P2P off, SHM)", "A100 PCIe, host-staged<br>2 ranks · 2.29 GB/s"),
        ("Forced TCP/loopback", "Forced TCP/loopback<br>8 ranks · 0.92 GB/s"),
    ]
    ticks = [tick for _, tick in transports]
    means = {
        label: [
            float(next(row["mean_ms"] for row in anchor
                       if row["label"] == label and row["transport"] == transport))
            for transport, _ in transports
        ]
        for label in ("Interleaved", "PyTorch DDP")
    }
    gaps = [
        (torch - interleaved) / interleaved * 100
        for interleaved, torch in zip(means["Interleaved"], means["PyTorch DDP"])
    ]
    for label in ("Interleaved", "PyTorch DDP"):
        picked = [
            next(row for row in anchor
                 if row["label"] == label and row["transport"] == transport)
            for transport, _ in transports
        ]
        text = [f"{label}<br>{value:.0f} ms" for value in means[label]]
        if label == "PyTorch DDP":
            text = [f"{cell} ({gap:+.1f}%)" for cell, gap in zip(text, gaps)]
        fig.add_bar(
            x=ticks,
            y=means[label],
            name=label,
            marker_color=colors[label],
            showlegend=False,
            error_y={"type": "data", "array": [float(row["std_ms"]) for row in picked]},
            text=text,
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{x}<br>%{fullData.name}: %{y:.1f} ms<extra></extra>",
            row=2,
            col=1,
        )

    fig.update_yaxes(title_text="Step time (ms)", row=1, col=1)
    fig.update_yaxes(
        title_text="Step time (ms)",
        type="log",
        range=[2.4, 3.45],
        tickmode="array",
        tickvals=[300, 500, 1000, 2000],
        ticktext=["300", "500", "1000", "2000"],
        row=2,
        col=1,
    )
    for annotation in fig.layout.annotations:
        annotation.update(font={"size": 13, "color": "#222222"})
    fig.update_layout(
        title_text="The gap between DDP implementations changes sign with the fabric",
        barmode="group",
        showlegend=False,
    )
    # The truncated y axis is what makes this panel misreadable on its own: at 8
    # sequences per rank the collective is 96% of every step, so mark the floor
    # all four modes stand on and say what is left for them to differ by.
    floor_ms = min(
        float(row["step_ms"]) for row in steps if row["global_batch_seqs"] == "64"
    )
    fig.add_hline(
        y=floor_ms,
        line_dash="dot",
        line_color="#333333",
        line_width=1.2,
        row=1,
        col=1,
    )
    fig.add_annotation(
        xref="x domain",
        yref="y",
        x=0.01,
        y=floor_ms,
        text=f"{floor_ms:.0f} ms — fastest step in any mode",
        showarrow=False,
        xanchor="left",
        yanchor="bottom",
        font={"size": 11, "color": "#333333"},
    )
    # The empty band above the three fast modes, inside the top panel.
    fig.add_annotation(
        xref="x domain",
        yref="y",
        x=0.60,
        y=1620,
        text=(
            "Only naive separates cleanly, and for a communication reason: it"
            "<br>sends 75 small collectives where the others send 14 fused ones."
            "<br>The other three share one 649 MB all-reduce that is 96% of the"
            "<br>step at this batch, leaving them at most the 49.5 ms of compute"
            "<br>to differ by. Their minima agree to 0.2%; the rest is tails."
        ),
        showarrow=False,
        xanchor="center",
        yanchor="middle",
        align="center",
        font={"size": 11, "color": "#555555"},
    )
    # The only free space in the lower panel is above the NVLink pair, which is
    # 5x shorter than the other two columns. Keep this to what the figure must
    # say on its own; the mechanism is the appendix's job.
    fig.add_annotation(
        xref="x2 domain",
        yref="y2",
        x=0.02,
        y=math.log10(2000),
        text=(
            "Both modes compiled, same"
            "<br>25 MB buckets. Under compile"
            "<br>only PyTorch DDP still"
            "<br>overlaps — appendix A1."
        ),
        showarrow=False,
        xanchor="left",
        yanchor="top",
        align="left",
        font={"size": 11, "color": "#555555"},
    )
    finish(fig, plots_dir / "ddp_mode_comparison", height=780)


def plot_ratio_sweep(data_dir: Path, plots_dir: Path) -> None:
    """Which DDP implementation wins, as a function of compute per communication.

    The four-way comparison used to exist at a single operating point, and that
    point was the worst possible one: at 8 sequences per rank the collective is
    96% of the step, so no implementation could differ from another by more than
    the compute (decisions.md section 26).  This sweeps the micro-batch, which is
    the knob that actually moves compute against a fixed 649 MB all-reduce -- the
    model size cancels out of that ratio, and gradient accumulation does not
    widen the overlap window.

    Every arm runs a different global batch by construction, so absolute step
    times are not comparable along x.  What is comparable, and what the question
    asks for, is each mode's step time *relative to the fastest mode at that same
    micro-batch* -- so the winner sits on 0% and the plot reads as "how much does
    picking the wrong implementation cost me here".

    Medians, not means: three of four modes showed late-run tails at 15 steps
    (section 26).  Each point carries a downward whisker to that arm's fastest
    step, so a position that rests on a tail is visible rather than implied, and
    ratio_sweep.csv carries mean, median, std and min for every arm.
    """
    path = data_dir / "ratio_sweep.csv"
    if not path.exists():
        return
    rows = read_csv(path)
    order = ["Naive", "Bucketed", "Interleaved", "PyTorch DDP"]
    colors = {"Naive": GRAY, "Bucketed": "#6A8E3A", "Interleaved": DDP_COLOR,
              "PyTorch DDP": "#A63A3A"}
    fabrics = [f for f in ("Native fabric", "Forced TCP/loopback")
               if any(row["fabric"] == f for row in rows)]

    # No subplot_titles: finish() pins the legend to the row they would occupy,
    # so the facets name themselves inside their own plotting area instead.
    fig = make_subplots(rows=1, cols=len(fabrics), shared_yaxes=True,
                        horizontal_spacing=0.06)
    for column, fabric in enumerate(fabrics, start=1):
        here = [row for row in rows if row["fabric"] == fabric]
        micros = sorted({int(row["micro_batch"]) for row in here})
        best = {
            micro: min(float(row["median_ms"]) for row in here
                       if int(row["micro_batch"]) == micro)
            for micro in micros
        }
        for label in order:
            points = {int(row["micro_batch"]): float(row["median_ms"])
                      for row in here if row["label"] == label}
            if not points:
                continue
            fastest = {int(row["micro_batch"]): float(row["min_ms"])
                       for row in here if row["label"] == label}
            xs = [micro for micro in micros if micro in points]
            fig.add_scatter(
                x=xs,
                y=[(points[micro] / best[micro] - 1) * 100 for micro in xs],
                name=label,
                mode="lines+markers",
                line={"color": colors[label], "width": 2.5},
                marker={"size": 8, "color": colors[label]},
                # Downward only, to each arm's fastest step: these are 10-step
                # arms with right tails (section 26), so the whisker shows how
                # much of a mode's position is its tail rather than its speed.
                error_y={
                    "type": "data",
                    "symmetric": False,
                    "array": [0] * len(xs),
                    "arrayminus": [(points[micro] - fastest[micro]) / best[micro] * 100
                                   for micro in xs],
                    "color": colors[label],
                    "thickness": 1.2,
                    "width": 3,
                },
                showlegend=(column == 1),
                hovertemplate=(f"{label}<br>micro-batch %{{x}}<br>"
                               "%{y:.1f}% slower than the best mode here<extra></extra>"),
                row=1,
                col=column,
            )
        fig.update_xaxes(title_text="Sequences per rank per backward", type="log",
                         tickmode="array", tickvals=micros,
                         ticktext=[str(micro) for micro in micros], row=1, col=column)
        fig.add_annotation(
            xref=f"x{'' if column == 1 else column} domain", yref="y domain",
            x=0.02, y=0.99, text=f"<b>{fabric}</b>", showarrow=False,
            xanchor="left", yanchor="top", font={"size": 13, "color": "#222222"},
        )
    fig.update_yaxes(title_text="Step time above the best mode (%)", row=1, col=1)
    fig.add_hline(y=0, line_color="#333333", line_width=1)
    fig.update_layout(
        title_text="What the implementation is worth, against compute per collective",
        showlegend=True,
    )
    # Two ratios on the throttled fabric: the segment carries a direction, not a
    # shape.  TCP_MICROS was not forwarded to the pod on the 2026-08-24 run.
    tcp_micros = sorted({int(row["micro_batch"]) for row in rows
                         if row["fabric"] == "Forced TCP/loopback"})
    if len(fabrics) > 1 and len(tcp_micros) < 3:
        fig.add_annotation(
            xref="x2 domain", yref="y domain",
            x=0.45, y=0.20,
            text=(f"Measured at {len(tcp_micros)} ratios; the segment"
                  "<br>shows the direction, not the shape."),
            showarrow=False, xanchor="center", align="center",
            font={"size": 11, "color": "#555555"},
        )
    finish(fig, plots_dir / "ratio_sweep", height=560)


def plot_pcie_modes(data_dir: Path, plots_dir: Path) -> None:
    """Compilation versus overlap on the one real PCIe box that was rentable.

    Reconstructed after an accidental deletion; the rendered SVG and pcie.csv
    fixed every element (categories, colours, labels, error bars, caption).
    """
    rows = [row for row in read_csv(data_dir / "pcie.csv") if row["gpus"] == "2"]
    labels = [row["arm"].removeprefix("2 GPU ") for row in rows]
    data: dict[str, list[Any]] = {
        "Arm": labels,
        "Optimizer step time (ms)": [float(row["step_ms"]) for row in rows],
        "Standard deviation (ms)": [float(row["step_std_ms"]) for row in rows],
        "Build": ["compiled" if row["compiled"] == "true" else "uncompiled" for row in rows],
    }
    fig = px.bar(
        data,
        x="Arm",
        y="Optimizer step time (ms)",
        color="Build",
        color_discrete_map={"compiled": DDP_COLOR, "uncompiled": GRAY},
        error_y="Standard deviation (ms)",
        text="Optimizer step time (ms)",
        category_orders={"Arm": labels},
        title="Compilation versus overlap on A100 PCIe",
    )
    fig.update_traces(
        texttemplate="%{y:.0f} ms",
        textposition="outside",
        cliponaxis=False,
        hovertemplate="%{x}<br>Step time: %{y:.1f} ms<extra></extra>",
    )
    headroom = max(
        float(row["step_ms"]) + float(row["step_std_ms"]) for row in rows
    )
    fig.update_yaxes(range=[0, headroom * 1.18])
    fig.add_annotation(
        x=0.01,
        y=0.98,
        xref="paper",
        yref="paper",
        text="2×A100 80GB PCIe, global batch 480, P2P unavailable (NCCL via SHM)",
        showarrow=False,
        xanchor="left",
        yanchor="top",
        font={"size": 11, "color": "#555555"},
    )
    finish(fig, plots_dir / "pcie_modes")


def plot_diloco_sync(data_dir: Path, plots_dir: Path) -> None:
    rows = read_csv(data_dir / "diloco_sync.csv")
    metric_order = ["Replica loss spread", "Synced − replica mean"]
    long_rows = [
        {
            "Training step": int(row["step"]),
            "Metric": metric,
            "Value": float(row[column]),
        }
        for row in rows
        for metric, column in (
            ("Replica loss spread", "replica_spread"),
            ("Synced − replica mean", "merge_delta"),
        )
    ]
    data: dict[str, list[Any]] = {
        key: [row[key] for row in long_rows] for key in ("Training step", "Metric", "Value")
    }
    fig = px.line(
        data,
        x="Training step",
        y="Value",
        color="Metric",
        facet_row="Metric",
        category_orders={"Metric": metric_order},
        color_discrete_map={
            "Replica loss spread": DDP_COLOR,
            "Synced − replica mean": DILOCO_COLOR,
        },
        markers=True,
        title="What each DiLoCo outer synchronization changes",
    )
    fig.update_traces(
        line_width=2.5,
        marker_size=6,
        hovertemplate="Step: %{x}<br>Value: %{y:.4f}<extra>%{fullData.name}</extra>",
    )
    fig.update_yaxes(matches=None)
    fig.update_layout(showlegend=False)
    fig.for_each_annotation(lambda annotation: annotation.update(visible=False))
    fig.layout.yaxis.title.text = "Synced − replica mean<br>(validation loss)"
    fig.layout.yaxis2.title.text = "Replica loss spread"
    fig.add_hline(y=0, row=1, col=1, line_color="#333333", line_width=1)
    finish(fig, plots_dir / "diloco_outer_sync", height=672)


def plot_loss_curves(data_dir: Path, plots_dir: Path) -> None:
    """The writeup's opening figure: both halves of the converged run's loss curve.

    Training loss is noisy per-step and validation is a 250-step grid, so they go
    on one axis with the training series drawn thin and semi-transparent behind
    the validation markers.
    """
    training = [
        row
        for row in read_csv(data_dir / "training_curves.csv")
        if row["series"].startswith("DDP") and int(row["step"]) > 0
    ]
    validation = [
        row
        for row in read_csv(data_dir / "validation_curves.csv")
        if row["series"].startswith("DDP")
    ]
    fig = Figure()
    fig.add_scatter(
        x=[float(row["tokens_billions"]) for row in training],
        y=[float(row["train_loss"]) for row in training],
        mode="lines",
        name="Training loss",
        line={"color": GRAY, "width": 1},
        opacity=0.55,
        hovertemplate="Tokens: %{x:.3f}B<br>Training loss: %{y:.4f}<extra></extra>",
    )
    fig.add_scatter(
        x=[float(row["tokens_billions"]) for row in validation],
        y=[float(row["val_loss"]) for row in validation],
        mode="lines+markers",
        name="Validation loss",
        line={"color": DDP_COLOR, "width": 2.5},
        marker={"size": 6},
        hovertemplate="Tokens: %{x:.3f}B<br>Validation loss: %{y:.4f}<extra></extra>",
    )
    fig.add_hline(
        y=3.28,
        line_dash="dash",
        line_color="#333333",
        line_width=1.2,
        annotation_text="3.28 target",
        annotation_position="bottom right",
    )
    fig.update_layout(
        title="Converged 8×A100 run: training and validation loss",
        xaxis_title="Training tokens (billions)",
        yaxis_title="Loss",
    )
    fig.update_yaxes(range=[3.1, 7.0])
    fig.add_annotation(
        x=0.98,
        y=0.96,
        xref="paper",
        yref="paper",
        text="Crossed 3.28 at step 9999 / 4.915B tokens, 3147.1 s of training",
        showarrow=False,
        xanchor="right",
        yanchor="top",
        font={"size": 11, "color": "#555555"},
    )
    finish(fig, plots_dir / "loss_curves")


def plot_capacity(data_dir: Path, plots_dir: Path) -> None:
    """Measured rentability of the SKUs this study wanted."""
    rows = read_csv(data_dir / "capacity_availability.csv")
    labels = [f"{row['venue']}<br>{row['sku']}" for row in rows]
    percent = [100 * float(row["available_fraction"]) for row in rows]
    data: dict[str, list[Any]] = {
        "Offer": labels,
        "Polls with stock (%)": percent,
        "Polls": [int(row["polls"]) for row in rows],
        "Hits": [int(row["polls_available"]) for row in rows],
    }
    fig = px.bar(
        data,
        x="Offer",
        y="Polls with stock (%)",
        color="Offer",
        color_discrete_sequence=[TCP_COLOR, GRAY, DDP_COLOR],
        text="Polls with stock (%)",
        custom_data=["Hits", "Polls"],
        title="How often an 8-GPU node was actually rentable",
    )
    fig.update_traces(
        texttemplate="%{y:.1f}%",
        textposition="outside",
        cliponaxis=False,
        hovertemplate="%{x}<br>%{customdata[0]} of %{customdata[1]} polls<extra></extra>",
    )
    fig.update_layout(showlegend=False)
    fig.update_yaxes(range=[0, 118])
    fig.add_annotation(
        x=0.5,
        y=0.98,
        xref="paper",
        yref="paper",
        text=(
            "2026-08-19 to 2026-08-21, polled about every 5 minutes. The A100 that was "
            "in stock<br>billed at &#36;22.32/h; the one that was not billed at &#36;12.72/h."
        ),
        showarrow=False,
        xanchor="center",
        yanchor="top",
        font={"size": 11, "color": "#555555"},
    )
    finish(fig, plots_dir / "capacity_availability")


def plot_diloco_k_penalty(data_dir: Path, plots_dir: Path) -> None:
    """What the DiLoCo merge penalty costs, measured two ways.

    Two panels, because the loss gap on its own is misleading. The gap is
    contaminated by the reference's own slope: the reference falls about 10x
    faster in the warmdown than on the plateau, so a *shrinking* token lag still
    reads as a *growing* loss gap. The bottom panel inverts the reference curve
    instead and reports the ratio the project committed to -- how many more
    tokens DiLoCo needs to reach the same loss.

    Both arms run the same 10000-step trapezoid, the same 500-step validation
    grid, and the same 491,520 tokens/step, so equal token counts sit at the same
    point in the LR schedule.  Each is measured against a no-DiLoCo run of that
    config on its own box; those two references are the same experiment and agree
    to within 0.016 everywhere.
    """
    rows = [
        row
        for row in read_csv(data_dir / "diloco_k_penalty.csv")
        if int(row["step"]) > 0  # step 0 is the shared init, before any merge
    ]
    # Replica count is the only variable left: the schedule, the token budget and
    # the reference config now match, and the hardware does not enter a loss plot.
    series = {}
    for row in rows:
        series.setdefault(f"K={int(row['replicas_k'])} replicas", []).append(row)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.13,
        subplot_titles=(
            "Validation loss above the reference",
            "Tokens needed to reach the same loss",
        ),
    )
    colors = {2: TCP_COLOR, 8: DILOCO_COLOR}
    for label, points in series.items():
        replicas = int(points[0]["replicas_k"])
        color = colors[replicas]
        fig.add_scatter(
            x=[float(row["tokens_billions"]) for row in points],
            y=[float(row["penalty"]) for row in points],
            mode="lines+markers",
            name=label,
            legendgroup=label,
            line={"color": color, "width": 2.5},
            marker={"size": 7},
            customdata=[
                [int(row["step"]), float(row["reference_val_loss"]), float(row["diloco_val_loss"])]
                for row in points
            ],
            hovertemplate=(
                "%{fullData.name}<br>Step: %{customdata[0]}<br>"
                "Reference: %{customdata[1]:.4f}<br>DiLoCo: %{customdata[2]:.4f}<br>"
                "Penalty: +%{y:.4f}<extra></extra>"
            ),
            row=1,
            col=1,
        )
        final = points[-1]
        fig.add_annotation(
            x=float(final["tokens_billions"]),
            y=math.log10(float(final["penalty"])),
            text=f"+{float(final['penalty']):.3f}",
            showarrow=False,
            xanchor="left",
            xshift=8,
            font={"color": color, "size": 13},
            row=1,
            col=1,
        )

        # Early rounds are worse than any loss the reference logged, so they carry
        # no ratio -- the curve starts where the comparison first becomes defined.
        ratio_points = [row for row in points if row["token_ratio"]]
        clean = [row for row in ratio_points if row["ratio_comparison"] == "clean"]
        # Mismatched points keep the last clean one so the dashed line joins up.
        mismatched = ratio_points[len(clean) - 1 :] if len(clean) < len(ratio_points) else []
        hover = (
            "%{fullData.name}<br>Step: %{customdata[0]}<br>"
            "Reference reached this loss at %{customdata[1]:.2f}B tokens<br>"
            "DiLoCo needs %{y:.2f}x the tokens<extra></extra>"
        )
        for segment, dashed in ((clean, False), (mismatched, True)):
            if not segment:
                continue
            fig.add_scatter(
                x=[float(row["tokens_billions"]) for row in segment],
                y=[float(row["token_ratio"]) for row in segment],
                mode="lines+markers",
                name=label,
                legendgroup=label,
                showlegend=False,
                line={"color": color, "width": 2.5, "dash": "dot" if dashed else "solid"},
                marker=(
                    {"size": 8, "symbol": "circle-open", "line": {"color": color, "width": 2}}
                    if dashed
                    else {"size": 7}
                ),
                customdata=[
                    [int(row["step"]), float(row["reference_tokens_at_equal_loss"])]
                    for row in segment
                ],
                hovertemplate=hover,
                row=2,
                col=1,
            )
        # Label the last *clean* point: past it the ratio flatters DiLoCo.
        final_ratio = clean[-1]
        fig.add_annotation(
            x=float(final_ratio["tokens_billions"]),
            y=float(final_ratio["token_ratio"]),
            text=f"{float(final_ratio['token_ratio']):.2f}×",
            showarrow=False,
            xanchor="left",
            xshift=8,
            # Up and right: clear of the parity line at K=2, and of the dashed
            # continuation, which dives away downward at K=8.
            yshift=12,
            font={"color": color, "size": 13},
            row=2,
            col=1,
        )

    # Both arms share a warmdown start now, so this is one line, not one per arm.
    # It is the point the two panels disagree about: the gap widens there while
    # the token ratio falls.
    for warmdown_step in sorted({int(row["warmdown_start_step"]) for row in rows}):
        warmdown_start = warmdown_step * TOKENS_PER_STEP / 1e9
        fig.add_vline(
            x=warmdown_start,
            line_dash="dot",
            line_color="#555555",
            line_width=1.2,
            opacity=0.7,
        )
        fig.add_annotation(
            x=warmdown_start,
            y=math.log10(0.45),
            text="warmdown",
            showarrow=False,
            xanchor="right",
            xshift=-4,
            textangle=-90,
            font={"size": 10, "color": "#555555"},
            row=1,
            col=1,
        )
    # Parity: the reference needed exactly as many tokens for the same loss.
    fig.add_hline(
        y=1.0,
        line_dash="dash",
        line_color=GRAY,
        line_width=1.2,
        row=2,
        col=1,
    )
    fig.add_annotation(
        x=0.08,
        y=1.0,
        text="no penalty",
        showarrow=False,
        xanchor="left",
        yshift=9,
        font={"size": 10, "color": GRAY},
        row=2,
        col=1,
    )

    fig.update_yaxes(
        title_text="Loss above reference",
        type="log",
        range=[math.log10(0.018), math.log10(3.0)],
        tickmode="array",
        tickvals=[0.03, 0.1, 0.3, 1, 3],
        ticktext=["+0.03", "+0.1", "+0.3", "+1.0", "+3.0"],
        row=1,
        col=1,
    )
    fig.update_yaxes(
        title_text="× the reference's tokens",
        range=[0.85, 3.7],
        tickmode="array",
        tickvals=[1, 2, 3],
        ticktext=["1×", "2×", "3×"],
        row=2,
        col=1,
    )
    fig.update_xaxes(range=[0, 5.6])
    fig.update_xaxes(title_text="Training tokens (billions)", row=2, col=1)
    for annotation in fig.layout.annotations:
        if annotation.text in (
            "Validation loss above the reference",
            "Tokens needed to reach the same loss",
        ):
            annotation.update(font={"size": 13, "color": "#222222"})
    fig.update_layout(title_text="What the DiLoCo merge penalty costs")
    fig.add_annotation(
        x=4.3,
        y=math.log10(2.6),
        text=(
            "Both arms: same 10,000-step trapezoid, same 500-step validation"
            "<br>grid, 491,520 tokens/step. Each against a no-DiLoCo run of that"
            "<br>config on its own box; the two references agree to within 0.016."
        ),
        showarrow=False,
        xanchor="right",
        yanchor="top",
        align="right",
        font={"size": 11, "color": "#555555"},
        row=1,
        col=1,
    )
    fig.add_annotation(
        x=0.12,
        y=2.42,
        text=(
            "The reference curve inverted. Rounds before ~1B tokens are worse"
            "<br>than anything the reference logged, so no ratio is defined."
            "<br>Dotted: both arms run the same 10,000-step schedule, but the"
            "<br>reference first reached these losses around step 3,400-4,200,"
            "<br>well before its own warmdown. A warmed-down DiLoCo point is"
            "<br>matched to a pre-warmdown reference point, which flatters"
            "<br>DiLoCo — 3.41× is the last clean K=8 value."
        ),
        showarrow=False,
        xanchor="left",
        yanchor="top",
        align="left",
        font={"size": 11, "color": "#555555"},
        row=2,
        col=1,
    )
    finish(fig, plots_dir / "diloco_k_penalty", height=760)


def plot_diloco_transport(data_dir: Path, plots_dir: Path) -> None:
    """Where DiLoCo's low communication actually buys something.

    Two panels, because the per-step advantage is not the answer on its own:
    DiLoCo needs about 2.5x the steps to reach the same loss, so the top panel
    flatters it and the bottom panel charges for that.
    """
    rows = read_csv(data_dir / "diloco_transport.csv")
    order = list(reversed([row["transport"] for row in rows]))
    index = {transport: position for position, transport in enumerate(order)}
    rows = sorted(rows, key=lambda row: index[row["transport"]])
    transports = [row["transport"] for row in rows]

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=(
            "Per optimizer step",
            "End to end, charging DiLoCo 2.49× the steps to reach 3.28",
        ),
    )
    panels = [
        (1, "ddp_step_ms_batch480", "diloco_step_ms_batch480", "%{y:.0f}", " ms"),
        (2, "ddp_time_to_3_28_hours", "diloco_time_to_3_28_hours", "%{y:.2f}", " h"),
    ]
    for row_index, ddp_column, diloco_column, template, unit in panels:
        for name, column, color in (
            ("DDP", ddp_column, DDP_COLOR),
            ("DiLoCo (H=500)", diloco_column, DILOCO_COLOR),
        ):
            fig.add_bar(
                x=transports,
                y=[float(row[column]) for row in rows],
                name=name,
                marker_color=color,
                legendgroup=name,
                showlegend=row_index == 1,
                text=[float(row[column]) for row in rows],
                texttemplate=template,
                textposition="outside",
                cliponaxis=False,
                hovertemplate="%{x}<br>%{fullData.name}: %{y:.2f}" + unit + "<extra></extra>",
                row=row_index,
                col=1,
            )
    fig.update_yaxes(title_text="Step time (ms)", type="log", range=[2, 4.3], row=1, col=1)
    fig.update_yaxes(title_text="Time to 3.28 (hours)", range=[0, 26], row=2, col=1)
    fig.update_xaxes(title_text="Transport", row=2, col=1)
    for annotation in fig.layout.annotations:
        annotation.update(font={"size": 13, "color": "#222222"})
    fig.update_layout(title_text="Where DiLoCo's low communication pays", barmode="group")
    fig.add_annotation(
        x=0.99,
        y=0.40,
        xref="paper",
        yref="paper",
        text=(
            "DiLoCo bars are reconstructed: measured DiLoCo compute plus this transport's"
            "<br>measured all-reduce divided by H=500. The 2.49× step ratio is §18's"
            "<br>estimate — DiLoCo never actually reached 3.28 inside this corpus."
        ),
        showarrow=False,
        xanchor="right",
        yanchor="top",
        align="right",
        font={"size": 11, "color": "#555555"},
    )
    finish(fig, plots_dir / "diloco_transport", height=760)


# Owned hardware against rented hardware.  Not the neutral gray used elsewhere
# for a baseline: gray against DDP_COLOR is ΔE 11.8 to normal vision, which is
# below the readable floor, while this green is 19.6.  It must never share a
# figure with DILOCO_COLOR, which it is ΔE 4.2 from under protanopia -- and it
# does not: the cost-basis figures carry no DiLoCo series.
OWNED_COLOR = "#6A8E3A"


def cost_rows(data_dir: Path) -> dict[str, dict[str, str]]:
    return {row["config_id"]: row for row in read_csv(data_dir / "costs.csv")}


def _bar_panel(
    fig, *, row, labels, values, colors, template, hover,
    error=None, text=None, textposition="outside",
):
    """One direct-labelled bar panel.

    Every value is printed on its bar, so identity never rests on colour alone
    and the panels stay readable in grayscale and under any CVD.  `textposition`
    takes a per-bar list too, for the case where a reference line would otherwise
    strike through one label.
    """
    fig.add_bar(
        x=labels,
        y=values,
        marker_color=colors,
        text=text if text is not None else values,
        texttemplate=template,
        textposition=textposition,
        cliponaxis=False,
        showlegend=False,
        error_y=error,
        hovertemplate=hover,
        row=row,
        col=1,
    )


def plot_time_and_cost(
    data_dir: Path,
    plots_dir: Path,
    *,
    configs: list[str],
    name: str,
    title: str,
    caption: str,
) -> None:
    """Time and price for one converged run, on the machines available to it.

    Two panels rather than two y axes on one plot: hours and dollars are
    different scales and a twin axis makes their crossing point an artifact of
    where the axes are pinned.  Stacked panels share the categorical x axis and
    each keeps a single scale, so the comparison is the reader's to make.

    Colour carries the one thing that must not be silently averaged: the desktop
    is *owned*, priced from wall power and electricity, while every other bar is
    a rental rate.  Those are not the same kind of dollar and the legend says so.
    """
    priced = cost_rows(data_dir)
    rows = [priced[config] for config in configs if config in priced]
    labels = [row["display_name"] for row in rows]
    hours = [float(row["runtime_hours"]) for row in rows]
    costs = [float(row["total_cost"]) for row in rows]
    bases = [row["cost_basis"] for row in rows]
    colors = [OWNED_COLOR if basis.startswith("wall power") else DDP_COLOR for basis in bases]

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=("", "Price of that one run"),
    )
    _bar_panel(
        fig,
        row=1,
        labels=labels,
        values=hours,
        colors=colors,
        template="%{y:.2f} h",
        hover="%{x}<br>%{y:.3f} h<extra></extra>",
    )
    _bar_panel(
        fig,
        row=2,
        labels=labels,
        values=costs,
        colors=colors,
        template="$%{y:.2f}",
        hover="%{x}<br>$%{y:.2f}<extra></extra>",
    )
    # A legend for the cost basis, drawn as two empty traces: the bars above are
    # a single trace each, so this is the only way to name the two colours.
    present = {
        "Rented, hourly rate": any(basis == "rental" for basis in bases),
        "Owned, wall power": any(basis.startswith("wall power") for basis in bases),
    }
    for basis, color in (("Rented, hourly rate", DDP_COLOR), ("Owned, wall power", OWNED_COLOR)):
        if present[basis]:
            fig.add_bar(x=[None], y=[None], name=basis, marker_color=color, showlegend=True)

    fig.update_yaxes(
        title_text="Hours to validation loss 3.28", range=[0, max(hours) * 1.22],
        row=1, col=1,
    )
    fig.update_yaxes(title_text="US dollars", range=[0, max(costs) * 1.22], row=2, col=1)
    for annotation in fig.layout.annotations:
        annotation.update(font={"size": 13, "color": "#222222"})
    fig.update_layout(title_text=title, barmode="group")
    fig.add_annotation(
        x=0.99,
        y=0.99,
        xref="paper",
        yref="paper",
        text=caption,
        showarrow=False,
        xanchor="right",
        yanchor="top",
        align="right",
        font={"size": 11, "color": "#555555"},
    )
    finish(fig, plots_dir / name, height=700)


def plot_transport_cost(data_dir: Path, plots_dir: Path) -> None:
    """What a converged run costs on each fabric, and why the ranking is not the price.

    Three panels because price is a product of two independent things: the hours
    the fabric makes you buy and the hourly rate of the SKU.  Putting hours and
    cost on twin y axes would invite reading their crossing as meaningful;
    stacked single-scale panels let the reader multiply the first two into the
    third.  The result is the writeup's point: the PCIe box is the cheapest
    machine per hour and one of the more expensive runs.

    The single A100 leads the row as the thing every fabric is being bought
    *instead of* -- it has no all-reduce at all, so it belongs before the
    bandwidth ordering rather than inside it, and it is the only bar whose rate
    is a one-GPU rate.

    The netem rows carry no rate: nothing was ever quoted for a throttled link,
    so their rate and cost bars are deliberately absent rather than zero.
    """
    rows = read_csv(data_dir / "transport_costs.csv")
    rows.sort(key=lambda row: -float(row["effective_bus_gbps"]))
    single = next(
        row for row in read_csv(data_dir / "costs.csv") if row["config_id"] == "a100x1"
    )
    single_hours = float(single["runtime_hours"])
    single_cost = float(single["total_cost"])

    # Six categories leave about 125 px per tick and plotly rotates anything
    # wider, so the long transport names are pre-wrapped instead of tilted.
    wrapped = {
        "A100 PCIe (P2P off, SHM)": "A100 PCIe<br>(P2P off, SHM)",
        "netem nominal 40 Gbit/s": "netem nominal<br>40 Gbit/s",
        "netem nominal 10 Gbit/s": "netem nominal<br>10 Gbit/s",
    }
    labels = ["1×A100 baseline<br>no all-reduce"] + [
        f"{wrapped.get(row['transport'], row['transport'])}"
        f"<br>{float(row['effective_bus_gbps']):.2f} GB/s"
        for row in rows
    ]
    evidence = [single["evidence"]] + [row["ddp_evidence"] for row in rows]
    colors = [
        DDP_COLOR if item == "Measured step time" else DILOCO_COLOR for item in evidence
    ]
    # A missing rate reads as NaN and plots as no bar; only the priced ones may
    # set an axis range.
    rates = [float(single["hourly_rental_rate"])] + [
        float(row["usd_per_hour"]) for row in rows
    ]
    hours = [single_hours] + [float(row["ddp_hours"]) for row in rows]
    costs = [single_cost] + [float(row["ddp_cost"]) for row in rows]

    def axis_top(values: list[float], headroom: float) -> float:
        return max(value for value in values if not math.isnan(value)) * headroom

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.09,
        subplot_titles=(
            "",
            "What the machine costs by the hour",
            "Price of one converged run — the product of the two above",
        ),
    )
    # A bar ending just under the one-GPU rule would have its outside label
    # struck through by that line, so those few label themselves inside.
    clearance = axis_top(hours, 1.25) * 0.09
    _bar_panel(
        fig,
        row=1,
        labels=labels,
        values=hours,
        colors=colors,
        template="%{y:.2f} h",
        hover="%{x}<br>%{y:.2f} h<extra></extra>",
        textposition=[
            "inside" if 0 < single_hours - value < clearance else "outside"
            for value in hours
        ],
    )
    _bar_panel(
        fig,
        row=2,
        labels=labels,
        values=rates,
        colors=colors,
        template="$%{y:.2f}",
        hover="%{x}<br>$%{y:.2f}/h<extra></extra>",
    )
    _bar_panel(
        fig,
        row=3,
        labels=labels,
        values=costs,
        colors=colors,
        template="$%{y:.0f}",
        hover="%{x}<br>$%{y:.2f}<extra></extra>",
    )
    for name, color in (
        ("Measured step time", DDP_COLOR),
        ("Reconstructed step time", DILOCO_COLOR),
    ):
        fig.add_bar(x=[None], y=[None], name=name, marker_color=color, showlegend=True)

    # finish() pins the legend to the strip the first row's subplot title would
    # occupy, so that panel names itself inside its own plotting area.
    fig.add_annotation(
        xref="x domain",
        yref="y domain",
        x=0.02,
        y=0.99,
        text="How many hours the fabric makes you buy",
        showarrow=False,
        xanchor="left",
        yanchor="top",
        row=1,
        col=1,
    )
    # The same ruler in both panels: what one A100 charges you in time, and in
    # money. Its label dodges left of the netem bars, which reach past the line.
    fig.add_hline(
        y=single_hours,
        line_dash="dash",
        line_color="#333333",
        line_width=1.2,
        row=1,
        col=1,
    )
    fig.add_annotation(
        xref="x domain",
        yref="y",
        x=0.30,
        y=single_hours,
        text=f"One A100 alone: {single_hours:.2f} h",
        showarrow=False,
        xanchor="left",
        yanchor="bottom",
        font={"size": 11, "color": "#333333"},
        row=1,
        col=1,
    )
    fig.add_hline(
        y=single_cost,
        line_dash="dash",
        line_color="#333333",
        line_width=1.2,
        row=3,
        col=1,
    )
    fig.add_annotation(
        xref="x3 domain",
        yref="y3",
        x=0.98,
        y=single_cost,
        text=f"One A100 alone: ${single_cost:.2f}",
        showarrow=False,
        xanchor="right",
        yanchor="bottom",
        font={"size": 11, "color": "#333333"},
        row=3,
        col=1,
    )
    fig.update_yaxes(title_text="Hours", range=[0, axis_top(hours, 1.25)], row=1, col=1)
    fig.update_yaxes(
        title_text="US dollars / hour", range=[0, axis_top(rates, 1.3)], row=2, col=1
    )
    fig.update_yaxes(
        title_text="US dollars", range=[0, axis_top(costs, 1.25)], row=3, col=1
    )
    for annotation in fig.layout.annotations:
        annotation.update(font={"size": 13, "color": "#222222"})
    fig.update_layout(title_text="What each fabric costs per converged run")
    fig.add_annotation(
        xref="x3 domain",
        yref="y3 domain",
        x=0.02,
        y=0.99,
        text=(
            "Rates priced 2026-08-24 and quoted per SKU, so venue and fabric"
            "<br>move together — see cost_inputs.csv. The PCIe bar is a 2-rank"
            "<br>measurement projected to 8, and its rate is a listed price"
            "<br>with no stock behind it. The netem links were never quoted,"
            "<br>so they are priced nowhere."
        ),
        showarrow=False,
        xanchor="left",
        yanchor="top",
        align="left",
        font={"size": 11, "color": "#555555"},
    )
    finish(fig, plots_dir / "transport_cost", height=900)


def plot_equal_token_cost(data_dir: Path, plots_dir: Path) -> None:
    """What the same token budget costs under DDP and under DiLoCo.

    The companion to `equal_token_runtime`: same 4.915B tokens, same box, same
    rate, so the bars differ only by method.  They are nearly equal, which is
    the finding -- on NVLink DiLoCo buys no time and therefore no money, while
    ending 0.245 worse.  Cost per run is not cost per result.
    """
    priced = cost_rows(data_dir)
    rows = [priced[config] for config in ("ddp_equal_tokens", "diloco_equal_tokens")]
    labels = ["DDP", "DiLoCo (H=500)"]
    costs = [float(row["total_cost"]) for row in rows]
    losses = [3.2730, 3.5183]
    fig = Figure()
    fig.add_bar(
        x=labels,
        y=costs,
        marker_color=[DDP_COLOR, DILOCO_COLOR],
        text=[f"${value:.2f}" for value in costs],
        textposition="outside",
        cliponaxis=False,
        showlegend=False,
        customdata=[[loss, float(row["runtime_hours"])] for loss, row in zip(losses, rows)],
        hovertemplate=(
            "%{x}<br>$%{y:.2f}<br>%{customdata[1]:.3f} h"
            "<br>Validation loss %{customdata[0]:.4f}<extra></extra>"
        ),
    )
    for label, cost, loss in zip(labels, costs, losses):
        fig.add_annotation(
            x=label,
            y=cost / 2,
            text=f"ends at<br>val {loss:.4f}",
            showarrow=False,
            font={"size": 12, "color": "#FFFFFF"},
        )
    fig.update_layout(
        title="Price of an equal 4.915B-token budget on 8×A100",
        xaxis_title="Method",
        yaxis_title="US dollars",
    )
    fig.update_yaxes(range=[0, max(costs) * 1.25])
    fig.add_annotation(
        x=0.5,
        y=0.99,
        xref="paper",
        yref="paper",
        text=(
            "Same box and the same $17.56/h, so only the method differs. Equal spend, "
            "worse result:<br>only the DDP bar is also a time to 3.28 — DiLoCo never "
            "reached the target inside this corpus."
        ),
        showarrow=False,
        xanchor="center",
        yanchor="top",
        align="center",
        font={"size": 11, "color": "#555555"},
    )
    finish(fig, plots_dir / "equal_token_cost")


def plot_diloco_transport_cost(data_dir: Path, plots_dir: Path) -> None:
    """Where DiLoCo's low communication is worth money rather than milliseconds.

    The runtime companion of this figure already exists; this one multiplies each
    bar by that SKU's hourly rate, which reorders nothing on its own -- DDP and
    DiLoCo share a machine on every fabric -- but puts the gap in the units the
    question was asked in.  DiLoCo's hours charge section 18's 2.49x token ratio,
    an estimate rather than a measured crossing, so every orange bar inherits it.
    """
    rows = read_csv(data_dir / "transport_costs.csv")
    rows.sort(key=lambda row: -float(row["effective_bus_gbps"]))
    labels = [
        f"{row['transport']}<br>{float(row['effective_bus_gbps']):.2f} GB/s" for row in rows
    ]
    fig = Figure()
    for name, column, color in (
        ("DDP", "ddp_cost", DDP_COLOR),
        ("DiLoCo (H=500)", "diloco_cost", DILOCO_COLOR),
    ):
        values = [float(row[column]) for row in rows]
        fig.add_bar(
            x=labels,
            y=values,
            name=name,
            marker_color=color,
            text=[f"${value:.0f}" for value in values],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{x}<br>%{fullData.name}: $%{y:.2f}<extra></extra>",
        )
    costs = [float(row[column]) for row in rows for column in ("ddp_cost", "diloco_cost")]
    fig.update_layout(
        title="Price of one converged run: DDP against DiLoCo, by fabric",
        xaxis_title="Transport",
        yaxis_title="US dollars",
        barmode="group",
    )
    fig.update_yaxes(range=[0, max(costs) * 1.15])
    fig.add_annotation(
        x=0.03,
        y=0.93,
        xref="paper",
        yref="paper",
        text=(
            "Both methods pay the same hourly rate on each fabric, so this is"
            "<br>the runtime figure in dollars. DiLoCo bars are reconstructed and"
            "<br>charge §18's 2.49× token ratio — an estimate; DiLoCo never"
            "<br>reached 3.28 inside this corpus."
        ),
        showarrow=False,
        xanchor="left",
        yanchor="top",
        align="left",
        font={"size": 11, "color": "#555555"},
    )
    finish(fig, plots_dir / "diloco_transport_cost")


def plot_diloco_transport_mfu(data_dir: Path, plots_dir: Path) -> None:
    """How much of the GPUs each method actually uses as the fabric slows.

    The DDP-only version of this figure falls off a cliff; the point of putting
    DiLoCo beside it is that its bars barely move, because an outer sync every
    500 steps is 1/500th of the traffic.  MFU here is the same PaLM-style
    numerator and the same measured bf16 roofline as everywhere else, so a
    reconstructed step time yields a lower-bound utilization.
    """
    rows = read_csv(data_dir / "diloco_transport.csv")
    rows.sort(key=lambda row: -float(row["effective_bus_gbps"]))
    labels = [
        f"{row['transport']}<br>{float(row['effective_bus_gbps']):.2f} GB/s" for row in rows
    ]

    def mfu(step_ms: float) -> float:
        return (
            100
            * MODEL_FLOPS_PER_TOKEN
            * TOKENS_PER_STEP
            / (step_ms / 1000)
            / (TRANSPORT_GPUS * A100_SXM4_40GB_TFLOPS * 1e12)
        )

    fig = Figure()
    for name, column, color in (
        ("DDP", "ddp_step_ms_batch480", DDP_COLOR),
        ("DiLoCo (H=500)", "diloco_step_ms_batch480", DILOCO_COLOR),
    ):
        values = [mfu(float(row[column])) for row in rows]
        fig.add_bar(
            x=labels,
            y=values,
            name=name,
            marker_color=color,
            text=[f"{value:.1f}%" for value in values],
            textposition="outside",
            cliponaxis=False,
            customdata=[[float(row[column])] for row in rows],
            hovertemplate=(
                "%{x}<br>%{fullData.name}: %{y:.1f}%"
                "<br>Step time: %{customdata[0]:.0f} ms<extra></extra>"
            ),
        )
    single = next(
        row for row in read_csv(data_dir / "scaling.csv") if row["config"] == "1xA100"
    )
    single_mfu = (
        100
        * MODEL_FLOPS_PER_TOKEN
        * TOKENS_PER_STEP
        / (float(single["step_ms"]) / 1000)
        / A100_SXM4_40GB_TFLOPS
        / 1e12
    )
    fig.add_hline(
        y=single_mfu,
        line_dash="dash",
        line_color="#333333",
        line_width=1.2,
    )
    fig.update_layout(
        title="Model FLOPs utilization: DDP against DiLoCo, by fabric",
        xaxis_title="Transport",
        yaxis_title="MFU (%)",
        barmode="group",
    )
    # Headroom for a caption band: every DiLoCo bar reaches about 60%, so there
    # is no free corner left inside the data.
    fig.update_yaxes(range=[0, max(single_mfu, 62) * 1.6])
    fig.add_annotation(
        x=0.99,
        y=0.97,
        xref="paper",
        yref="paper",
        text=(
            f"Dashed: one A100 on its own, {single_mfu:.1f}%. DiLoCo holds that level on every"
            "<br>fabric, because it all-reduces once per 500 steps. That is a claim about the"
            "<br>GPUs, not about the result: at equal tokens DiLoCo ends 0.245 worse, and it"
            "<br>needs about 2.5× the tokens to catch up."
        ),
        showarrow=False,
        xanchor="right",
        yanchor="top",
        align="right",
        font={"size": 11, "color": "#555555"},
    )
    finish(fig, plots_dir / "diloco_transport_mfu")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--data", type=Path, default=root / "docs/writeup_data")
    parser.add_argument("--output", type=Path, default=root / "docs/plots")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)

    plot_validation(args.data, args.output)
    plot_loss_curves(args.data, args.output)
    plot_capacity(args.data, args.output)
    plot_diloco_k_penalty(args.data, args.output)
    plot_diloco_transport(args.data, args.output)
    plot_scaling(args.data, args.output)
    plot_time_to_target(args.data, args.output)
    plot_transport(args.data, args.output)
    plot_transport_mfu(args.data, args.output)
    plot_equal_token_runtime(args.data, args.output)
    plot_ddp_modes(args.data, args.output)
    plot_ratio_sweep(args.data, args.output)
    plot_pcie_modes(args.data, args.output)
    plot_diloco_sync(args.data, args.output)
    plot_time_and_cost(
        args.data,
        args.output,
        configs=["rtx3090_desktop_estimate", "a100x1"],
        name="time_and_cost_baseline",
        title="One GPU: how long the baseline takes, and what it costs",
        caption=(
            "Both bars extrapolate a measured step time to the measured step-9999"
            "<br>crossing. The desktop is owned hardware: 800 W average at $0.23/kWh,"
            "<br>no capital allocation, so it is a floor rather than a like-for-like price."
        ),
    )
    plot_time_and_cost(
        args.data,
        args.output,
        configs=["rtx3090_desktop_estimate", "a100x1", "a100x8_nvlink"],
        name="time_and_cost_scaling",
        title="Adding the ideal 8-GPU box: 7.66× the speed, roughly the same price",
        caption=(
            "Eight A100s over NVLink cost about what one A100 costs for the same run:"
            "<br>the rate is 8× but the hours are 7.66× fewer. Scaling is nearly free"
            "<br>here — it is the slower fabrics, further on, that are not."
        ),
    )
    plot_transport_cost(args.data, args.output)
    plot_equal_token_cost(args.data, args.output)
    plot_diloco_transport_cost(args.data, args.output)
    plot_diloco_transport_mfu(args.data, args.output)
    print(f"wrote interactive HTML, SVG, and PNG plots to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
