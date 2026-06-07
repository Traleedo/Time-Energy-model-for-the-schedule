"""Slot-based GA scheduling demo."""

from datetime import datetime, timedelta

from core.config import CONFIG

from core.energy_curve import EnergyCurve
from core.task import Task
from core.slot_scheduler import SlotScheduler
from core.slot_ga_solver import SlotGA
from core.visualization import plot_schedule
from solvers.slot_sa import SlotSA


if __name__ == "__main__":
    base = datetime.strptime(CONFIG.START_DATE, "%Y-%m-%d")
    tasks = [
        Task("OS-lab",    2.0, "coding",   (base + timedelta(days=2)).replace(hour=23, minute=59), 1.0),
        Task("Multimedia", 1.0, "coding",   (base + timedelta(days=6)).replace(hour=23, minute=59), 0.3),
        Task("Writing",    0.7, "writing",  (base + timedelta(days=6)).replace(hour=23, minute=59), 0.7),
        Task("Math",       0.5, "learning", (base + timedelta(days=7)).replace(hour=23, minute=59), 0.3),
    ]

    cfg = {k: getattr(CONFIG, k) for k in dir(CONFIG) if k.isupper()}
    scheduler = SlotScheduler(tasks, EnergyCurve(), cfg)
    ga = SlotSA(scheduler, config=cfg)
    best, history = ga.run(verbose=True)

    metrics = scheduler.evaluate_slots(best)
    print(f"fitness={metrics['fitness']:.1f} energy={metrics['energy_match']:.3f} "
          f"on_time={metrics['on_time_rate']:.3f} composite={metrics['composite']:.3f}")    
    for item in sorted(scheduler.assign(best), key=lambda x: x["start"]):
        t = item["task"]
        s, e = item["start"], item["end"]
        match = 1 - abs(t.energy_req - scheduler.energy_curve.get_energy_datetime(s))
        status = "OK" if e <= t.deadline else "LATE"
        print(f"  {t.name:12s} {s.strftime('%m/%d %H:%M')}-{e.strftime('%H:%M')}  "
              f"E_need={t.energy_req:.1f} E_slot={scheduler.energy_curve.get_energy_datetime(s):.2f} "
              f"match={match:.2f} {status}")

    plot_schedule(scheduler, best, history)
    print("Saved: schedule_result.png")
