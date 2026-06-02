"""Visualization: clean course-schedule-style timetable."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from datetime import datetime, timedelta

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def plot_schedule(scheduler, slot_indices, history=None, save_path="schedule_result.png"):
    """Minimal course-schedule timetable.

    Y-axis = dates (top to bottom), X-axis = time of day.
    White = free, Red = busy (courses/lunch), Green = scheduled tasks.
    """
    schedule = scheduler.assign(slot_indices)
    start_date = scheduler.start_date
    n_days = scheduler.planning_days
    work_start = scheduler.work_start
    work_end = scheduler.work_end

    hour_span = work_end - work_start
    row_h = 0.85

    fig_w = 2.0 + hour_span * 1.15 + 0.5
    fig_h = 0.6 + n_days * row_h + 1.2
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")

    ax_left = 2.0 / fig_w
    ax_bot = 1.0 / fig_h
    ax_w = (hour_span * 1.15) / fig_w
    ax_h = (n_days * row_h) / fig_h

    ax = fig.add_axes([ax_left, ax_bot, ax_w, ax_h])
    ax.set_xlim(work_start, work_end)
    ax.set_ylim(-0.2, n_days + 0.2)
    ax.invert_yaxis()

    # ── axes styling ───────────────────────────────────────────────
    ax.tick_params(left=False, bottom=False)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#ddd")

    # X ticks: every 2 hours
    xs = list(range(int(work_start) + (1 if work_start % 1 else 0), int(work_end), 2))
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{h:02d}:00" for h in xs], fontsize=8)
    ax.grid(True, axis="x", color="#e8e8e8", linewidth=0.4)

    # Y ticks: day labels
    day_names = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    for d in range(n_days):
        day = start_date + timedelta(days=d)
        label = day_names[d] if d < 7 else day.strftime("%a")
        ax.text(work_start - 0.5, d + row_h / 2,
                f"{label}\n{day.strftime('%m/%d')}",
                ha="right", va="center", fontsize=8, fontweight="bold", color="#333")
    ax.set_yticks([])

    # ── red: busy periods (courses + lunch) ────────────────────────
    for b in scheduler.busy_periods:
        d = (b["start"].date() - start_date.date()).days
        if d < 0 or d >= n_days:
            continue
        bh = b["start"].hour + b["start"].minute / 60
        eh = b["end"].hour + b["end"].minute / 60
        dur = eh - bh
        if dur <= 0:
            continue

        label = b.get("name", "课程")
        is_lunch = label == "午休"
        face = "#fff3cd" if is_lunch else "#fadbd8"
        edge = "#e0c068" if is_lunch else "#e74c3c"
        text_color = "#b08800" if is_lunch else "#c0392b"

        rect = FancyBboxPatch((bh, d + 0.06), dur, row_h - 0.12,
                              boxstyle="round,pad=0.04", linewidth=1.0,
                              facecolor=face, edgecolor=edge, alpha=0.55, zorder=1)
        ax.add_patch(rect)
        ax.text(bh + dur / 2, d + row_h / 2, label, ha="center", va="center",
                fontsize=7, color=text_color, alpha=0.85, fontweight="bold")

    # ── green: scheduled tasks ─────────────────────────────────────
    for item in schedule:
        task = item["task"]
        s, e = item["start"], item["end"]
        d = (s.date() - start_date.date()).days
        if d < 0 or d >= n_days:
            continue
        sh = s.hour + s.minute / 60
        eh = e.hour + e.minute / 60
        dur = max(eh - sh, 0.4)

        rect = FancyBboxPatch((sh, d + 0.1), dur, row_h - 0.2,
                              boxstyle="round,pad=0.05", linewidth=1.2,
                              facecolor="#2ecc71", edgecolor="white", alpha=0.85, zorder=3)
        ax.add_patch(rect)

        # text
        mid_x = sh + dur / 2
        mid_y = d + row_h / 2
        if dur >= 1.5:
            text = f"{task.name}  [{task.task_type}]"
        elif dur >= 0.8:
            text = task.name if len(task.name) <= 8 else task.name[:7] + ".."
        else:
            text = task.name[:4]
        ax.text(mid_x, mid_y, text, ha="center", va="center",
                fontsize=7, color="white", fontweight="bold", zorder=4)

    # ── hour separator lines ───────────────────────────────────────
    for h in range(int(work_start), int(work_end) + 1):
        lw = 0.6 if h % 2 == 0 else 0.3
        ax.axvline(h, color="#ccc" if h % 2 == 0 else "#e8e8e8",
                   linewidth=lw, zorder=0)

    # ── title ──────────────────────────────────────────────────────
    ax.set_title("周日程表  (白色=空闲  红色=占用  绿色=已排任务)",
                 fontsize=11, fontweight="bold", color="#333", pad=8)

    fig.savefig(save_path, dpi=100, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    return fig
