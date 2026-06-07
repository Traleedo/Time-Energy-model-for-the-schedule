"""Visualization: gantt chart + convergence for slot-based scheduling."""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime, timedelta


def plot_schedule(scheduler, slot_indices, history=None, save_path="schedule_result.png"):
    """Main visualization: gantt chart with energy curve + optional convergence.

    Args:
        scheduler: SlotScheduler instance (holds tasks, energy_curve, busy_periods, etc.)
        slot_indices: list[int] chromosome
        history: optional list[float] fitness history for convergence subplot
        save_path: output file path
    """
    schedule = scheduler.assign(slot_indices)
    ecurve = scheduler.energy_curve
    n_tasks = len(schedule)
    n_days = scheduler.planning_days

    has_history = history is not None and len(history) > 1
    fig = plt.figure(figsize=(16, 6 + int(has_history) * 3))

    if has_history:
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 2], hspace=0.3)
        ax_conv = fig.add_subplot(gs[0])
        ax_conv.plot(history, color="#2ca02c", linewidth=1.5, alpha=0.85)
        ax_conv.set_xlabel("Generation")
        ax_conv.set_ylabel("Fitness")
        ax_conv.grid(alpha=0.15)
        ax_gantt = fig.add_subplot(gs[1])
    else:
        ax_gantt = fig.add_subplot(1, 1, 1)

    # Energy curve as background (relative to day hour)
    hours = np.linspace(scheduler.work_start, scheduler.work_end, 200)
    energies = [ecurve.get_energy(h) for h in hours]
    ax_energy = ax_gantt.twinx()
    ax_energy.plot(hours, energies, color="#d62728", linewidth=2, alpha=0.5, label="Energy")
    ax_energy.fill_between(hours, energies, alpha=0.06, color="#d62728")
    ax_energy.set_ylim(0, 1.1)
    ax_energy.set_ylabel("Energy", color="#d62728", alpha=0.6)
    ax_energy.tick_params(axis="y", colors="#d62728")

    # Busy periods (shaded)
    for b in scheduler.busy_periods:
        bh = b["start"].hour + b["start"].minute / 60
        eh = b["end"].hour + b["end"].minute / 60
        day_off = (b["start"].date() - scheduler.start_date.date()).days
        ax_gantt.axvspan(bh, eh, color="gray", alpha=0.12, zorder=0)

    # Day separators
    for d in range(1, n_days):
        ax_gantt.axvline(x=scheduler.work_end, ymin=0, ymax=1, color="black", alpha=0.15, linewidth=1, linestyle="--")

    # Task bars — color by how well energy matches
    cmap = plt.cm.RdYlGn
    for i, item in enumerate(schedule):
        task = item["task"]
        sh = item["start"].hour + item["start"].minute / 60
        eh = item["end"].hour + item["end"].minute / 60
        day_off = (item["start"].date() - scheduler.start_date.date()).days

        slot_e = ecurve.get_energy_datetime(item["start"])
        match = 1.0 - abs(task.energy_req - slot_e)
        color = cmap(0.3 + 0.7 * match)  # greener = better match

        y_center = n_tasks * (n_days - 1 - day_off) + (n_tasks - 1 - i)
        ax_gantt.barh(y_center, eh - sh, left=sh, height=0.55,
                      color=color, edgecolor="black", linewidth=0.5, zorder=3)

        on_time = "✓" if item["end"] <= task.deadline else "✗"
        label = f"{task.name}  E={task.energy_req:.1f} match={match:.2f} {on_time}"
        ax_gantt.text(sh + 0.05, y_center, label, va="center", fontsize=7.5)

    # Day labels on the right
    for d in range(n_days):
        y_mid = n_tasks * (n_days - 1 - d) + (n_tasks - 1) / 2
        day_label = (scheduler.start_date + timedelta(days=d)).strftime("%a %m/%d")
        ax_gantt.text(scheduler.work_end + 0.3, y_mid, day_label, va="center", fontsize=10,
                      fontweight="bold", color="#555555")

    ax_gantt.set_xlim(scheduler.work_start - 0.5, scheduler.work_end + 4)
    ax_gantt.set_ylim(-0.5, n_tasks * n_days + 0.5)
    ax_gantt.set_xlabel("Hour of day")
    ax_gantt.set_yticks([])
    ax_gantt.grid(axis="x", alpha=0.15)

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig
