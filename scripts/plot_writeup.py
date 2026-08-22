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
    those tails -- their minima agree to about 2%.  The bottom panel is the same
    question at the anchor's batch, where the two modes that were measured on
    NVLink land within 1%.
    """
    steps = read_csv(data_dir / "ddp_mode_steps.csv")
    summary = read_csv(data_dir / "ddp_modes.csv")
    order = ["Naive", "Bucketed", "Interleaved", "PyTorch DDP"]
    colors = {"Naive": GRAY, "Bucketed": "#6A8E3A", "Interleaved": DDP_COLOR,
              "PyTorch DDP": TCP_COLOR}

    fig = make_subplots(
        rows=2,
        cols=1,
        vertical_spacing=0.14,
        row_heights=[0.58, 0.42],
        subplot_titles=(
            "Every timed step, global batch 64 over forced TCP/loopback",
            "Anchor batch 480 — only these two modes were run on NVLink",
        ),
    )
    for label in order:
        values = [float(row["step_ms"]) for row in steps if row["label"] == label]
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

    anchor = [row for row in summary if row["global_batch_seqs"] == "480"]
    transports = ["NVLink NV12", "Forced TCP/loopback"]
    for label in ("Interleaved", "PyTorch DDP"):
        picked = [
            next(row for row in anchor if row["label"] == label and row["transport"] == transport)
            for transport in transports
        ]
        fig.add_bar(
            x=transports,
            y=[float(row["mean_ms"]) for row in picked],
            name=label,
            marker_color=colors[label],
            showlegend=False,
            error_y={"type": "data", "array": [float(row["std_ms"]) for row in picked]},
            text=[f"{label}<br>{float(row['mean_ms']):.0f} ms" for row in picked],
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
        title_text="Four DDP implementations, and how much of the gap is real",
        barmode="group",
        showlegend=False,
    )
    # The empty band above the three fast modes, inside the top panel.
    fig.add_annotation(
        xref="x domain",
        yref="y",
        x=0.62,
        y=1620,
        text=(
            "Only naive separates cleanly — its best step is slower than"
            "<br>every other mode's worst. The other three overlap: their"
            "<br>fastest steps agree to 2%, and the means differ only"
            "<br>through the late-run tails."
        ),
        showarrow=False,
        xanchor="center",
        yanchor="middle",
        align="center",
        font={"size": 11, "color": "#555555"},
    )
    finish(fig, plots_dir / "ddp_mode_comparison", height=780)


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
    """How long the merge penalty lasts, at two replica counts.

    Plots the gap only. The two arms ran different schedule lengths on different
    hardware, so their absolute validation losses are not comparable and are
    deliberately not drawn -- only each arm's distance from its own reference.
    """
    rows = [
        row
        for row in read_csv(data_dir / "diloco_k_penalty.csv")
        if int(row["step"]) > 0  # step 0 is the shared init, before any merge
    ]
    series = {}
    for row in rows:
        label = f"K={row['replicas_k']} ({row['schedule_steps']} steps)"
        series.setdefault(label, []).append(row)

    fig = Figure()
    colors = {2: TCP_COLOR, 8: DILOCO_COLOR}
    for label, points in series.items():
        replicas = int(points[0]["replicas_k"])
        fig.add_scatter(
            x=[float(row["tokens_billions"]) for row in points],
            y=[float(row["penalty"]) for row in points],
            mode="lines+markers",
            name=label,
            line={"color": colors[replicas], "width": 2.5},
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
        )
        # Mark where this arm's warmdown begins: the tails diverge there, and the
        # two arms reach it at different token counts.
        warmdown_start = int(points[0]["warmdown_start_step"]) * TOKENS_PER_STEP / 1e9
        fig.add_vline(
            x=warmdown_start,
            line_dash="dot",
            line_color=colors[replicas],
            line_width=1.2,
            opacity=0.7,
        )
        fig.add_annotation(
            x=warmdown_start,
            y=math.log10(2.4),
            text="warmdown",
            showarrow=False,
            xanchor="right",
            xshift=-4,
            textangle=-90,
            font={"size": 10, "color": colors[replicas]},
        )
        final = points[-1]
        fig.add_annotation(
            x=float(final["tokens_billions"]),
            y=math.log10(float(final["penalty"])),
            text=f"+{float(final['penalty']):.3f}",
            showarrow=False,
            xanchor="left",
            xshift=8,
            font={"color": colors[replicas], "size": 13},
        )
    fig.update_layout(
        title="How long the DiLoCo merge penalty lasts",
        xaxis_title="Training tokens (billions)",
        yaxis_title="Validation loss above each arm's own reference",
    )
    fig.update_xaxes(range=[0, 5.6])
    fig.update_yaxes(
        type="log",
        range=[math.log10(0.02), math.log10(3.0)],
        tickmode="array",
        tickvals=[0.03, 0.1, 0.3, 1, 3],
        ticktext=["+0.03", "+0.1", "+0.3", "+1.0", "+3.0"],
    )
    fig.add_annotation(
        x=0.985,
        y=0.34,
        xref="paper",
        yref="paper",
        text=(
            "Each arm against its own reference, on the same token axis."
            "<br>The two ran different schedule lengths, so they are not at"
            "<br>the same point in the LR trapezoid at equal tokens — the"
            "<br>dotted lines mark where each one warms down."
        ),
        showarrow=False,
        xanchor="right",
        yanchor="top",
        align="right",
        font={"size": 11, "color": "#555555"},
    )
    # The K=2 reference logged no 5000/5500 point, so its last segment spans a
    # quarter of the schedule with nothing measured inside it.
    fig.add_annotation(
        x=2.35,
        y=math.log10(0.048),
        text="no reference point logged between these two",
        showarrow=False,
        xanchor="center",
        yanchor="bottom",
        font={"size": 10, "color": TCP_COLOR},
    )
    finish(fig, plots_dir / "diloco_k_penalty")


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
    plot_pcie_modes(args.data, args.output)
    plot_diloco_sync(args.data, args.output)
    print(f"wrote interactive HTML, SVG, and PNG plots to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
