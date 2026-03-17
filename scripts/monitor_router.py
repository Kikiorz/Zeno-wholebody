#!/usr/bin/env python3
"""Router training monitor — pretty live dashboard for train.log."""

import re
import sys
import time
import shutil
from pathlib import Path

LOG_PATH = Path(__file__).resolve().parent.parent / "model" / "router" / "train.log"
CKPT_DIR = Path(__file__).resolve().parent.parent / "model" / "router" / "checkpoints"
TOTAL_STEPS = 100_000

# ── ANSI colors ──────────────────────────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
CYAN   = "\033[36m"
WHITE  = "\033[97m"
RED    = "\033[31m"
BLUE   = "\033[34m"
MAGENTA= "\033[35m"
BG_DARK= "\033[48;5;235m"

LINE_RE = re.compile(
    r"step\s+(\d+)/(\d+)\s+\(\s*([\d.]+)%\)\s+\|\s+"
    r"loss\s+([\d.]+)\s+\|\s+"
    r"mean P\(B\)\s+([\d.]+)\s+\|\s+"
    r"lr_bb\s+([\d.e+-]+)\s+lr\s+([\d.e+-]+)\s+\|\s+"
    r"([\d.]+)\s+steps/s\s+\|\s+"
    r"ETA\s+(\S+)"
)


def parse_log(path):
    """Parse all log lines, return list of dicts."""
    entries = []
    if not path.exists():
        return entries
    with open(path) as f:
        for line in f:
            m = LINE_RE.search(line)
            if m:
                entries.append({
                    "step": int(m.group(1)),
                    "total": int(m.group(2)),
                    "pct": float(m.group(3)),
                    "loss": float(m.group(4)),
                    "mean_pb": float(m.group(5)),
                    "lr_bb": m.group(6),
                    "lr": m.group(7),
                    "speed": float(m.group(8)),
                    "eta": m.group(9),
                })
    return entries


def bar(pct, width=40):
    """Render a colored progress bar."""
    filled = int(width * pct / 100)
    empty = width - filled
    if pct < 25:
        color = RED
    elif pct < 60:
        color = YELLOW
    else:
        color = GREEN
    return f"{color}{'█' * filled}{DIM}{'░' * empty}{RESET}"


def sparkline(values, width=50):
    """Mini sparkline chart from a list of floats."""
    if not values:
        return ""
    blocks = " ▁▂▃▄▅▆▇█"
    mn, mx = min(values), max(values)
    rng = mx - mn if mx != mn else 1.0
    # sample evenly if too many points
    if len(values) > width:
        step = len(values) / width
        sampled = [values[int(i * step)] for i in range(width)]
    else:
        sampled = values
    return "".join(blocks[min(8, int((v - mn) / rng * 8))] for v in sampled)


def list_checkpoints():
    """Return sorted list of saved checkpoint step numbers."""
    if not CKPT_DIR.exists():
        return []
    ckpts = []
    for f in CKPT_DIR.glob("step_*.pt"):
        m = re.search(r"step_(\d+)\.pt", f.name)
        if m:
            ckpts.append(int(m.group(1)))
    return sorted(ckpts)


def render(entries):
    """Render the full dashboard."""
    cols = shutil.get_terminal_size((80, 24)).columns
    sep = f"{DIM}{'─' * cols}{RESET}"

    print(f"\033[2J\033[H", end="")  # clear screen

    # ── Header ───────────────────────────────────────────────────────────
    print(f"{BOLD}{CYAN}  ╔{'═' * (cols - 4)}╗{RESET}")
    title = "ROUTER TRAINING MONITOR"
    pad = (cols - 4 - len(title)) // 2
    print(f"{BOLD}{CYAN}  ║{' ' * pad}{WHITE}{title}{' ' * (cols - 4 - pad - len(title))}{CYAN}║{RESET}")
    print(f"{BOLD}{CYAN}  ╚{'═' * (cols - 4)}╝{RESET}")
    print()

    if not entries:
        print(f"  {YELLOW}Waiting for training to start...{RESET}")
        print(f"  {DIM}Log: {LOG_PATH}{RESET}")
        return

    e = entries[-1]
    total = e["total"]

    # ── Progress ─────────────────────────────────────────────────────────
    print(f"  {BOLD}Progress{RESET}")
    print(f"  {bar(e['pct'])}  {BOLD}{e['pct']:5.1f}%{RESET}  "
          f"step {WHITE}{BOLD}{e['step']:,}{RESET}{DIM}/{total:,}{RESET}")
    print()

    # ── Stats table ──────────────────────────────────────────────────────
    print(f"  {BOLD}{'Metric':<20} {'Current':>12} {'Best':>12} {'Start':>12}{RESET}")
    print(f"  {sep}")

    losses = [x["loss"] for x in entries]
    pbs    = [x["mean_pb"] for x in entries]

    best_loss = min(losses)
    print(f"  {CYAN}{'Loss':<20}{RESET} {WHITE}{e['loss']:>12.4f}{RESET} "
          f"{GREEN}{best_loss:>12.4f}{RESET} {DIM}{entries[0]['loss']:>12.4f}{RESET}")
    print(f"  {CYAN}{'Mean P(B)':<20}{RESET} {WHITE}{e['mean_pb']:>12.3f}{RESET} "
          f"{DIM}{max(pbs):>12.3f}{RESET} {DIM}{entries[0]['mean_pb']:>12.3f}{RESET}")
    print(f"  {CYAN}{'Speed':<20}{RESET} {WHITE}{e['speed']:>11.1f}/s{RESET} "
          f"{DIM}{max(x['speed'] for x in entries):>11.1f}/s{RESET} {DIM}{entries[0]['speed']:>11.1f}/s{RESET}")
    print(f"  {CYAN}{'ETA':<20}{RESET} {YELLOW}{BOLD}{e['eta']:>12}{RESET}")
    print(f"  {CYAN}{'LR (backbone)':<20}{RESET} {DIM}{e['lr_bb']:>12}{RESET}")
    print(f"  {CYAN}{'LR (rest)':<20}{RESET} {DIM}{e['lr']:>12}{RESET}")
    print()

    # ── Loss sparkline ───────────────────────────────────────────────────
    spark_w = min(60, cols - 20)
    print(f"  {BOLD}Loss curve{RESET}  {DIM}(last {len(losses)} logs){RESET}")
    print(f"  {GREEN}{sparkline(losses, spark_w)}{RESET}")
    print(f"  {DIM}{max(losses):.4f} ▲  ▼ {min(losses):.4f}{RESET}")
    print()

    # ── P(B) sparkline ───────────────────────────────────────────────────
    print(f"  {BOLD}Mean P(B) curve{RESET}")
    print(f"  {MAGENTA}{sparkline(pbs, spark_w)}{RESET}")
    print(f"  {DIM}{max(pbs):.3f} ▲  ▼ {min(pbs):.3f}{RESET}")
    print()

    # ── Checkpoints ──────────────────────────────────────────────────────
    ckpts = list_checkpoints()
    if ckpts:
        print(f"  {BOLD}Checkpoints{RESET}  {DIM}({len(ckpts)} saved){RESET}")
        ckpt_str = "  "
        for c in ckpts:
            ckpt_str += f"{GREEN}■{RESET} {c:,}  "
        # Mark remaining
        expected = list(range(10000, total + 1, 10000))
        for ex in expected:
            if ex not in ckpts and ex > e["step"]:
                ckpt_str += f"{DIM}□ {ex:,}  {RESET}"
                if len(ckpt_str) > cols * 2:
                    ckpt_str += f"{DIM}...{RESET}"
                    break
        print(ckpt_str)
    print()
    print(f"  {DIM}Log: {LOG_PATH}{RESET}")
    print(f"  {DIM}Updated: {time.strftime('%H:%M:%S')}  |  Ctrl+C to exit{RESET}")


def main():
    interval = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    print(f"Monitoring {LOG_PATH} (refresh every {interval}s)...")
    try:
        while True:
            entries = parse_log(LOG_PATH)
            render(entries)
            # Check if training finished
            if entries and entries[-1]["step"] >= entries[-1]["total"]:
                print(f"\n  {GREEN}{BOLD}  Training complete!{RESET}\n")
                break
            time.sleep(interval)
    except KeyboardInterrupt:
        print(f"\n{DIM}Monitor stopped.{RESET}")


if __name__ == "__main__":
    main()
