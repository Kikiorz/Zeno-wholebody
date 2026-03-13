#!/usr/bin/env python3
"""
VC-ACT Training Monitor — 实时训练仪表盘

用法:
  /venv/mult-act/bin/python3 monitor.py
  /venv/mult-act/bin/python3 monitor.py --refresh 2   # 2秒刷新
"""

import argparse
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

# ─── Config ───────────────────────────────────────────────────────
VC_ROOT = Path("/workspace/Mult-skill ACT/VC-ACT")
LOG_DIR = VC_ROOT / "logs"
SCHEDULER_LOG = VC_ROOT / "scheduler.log"

# ANSI colors
RESET   = "\033[0m"
BOLD    = "\033[1m"
DIM     = "\033[2m"
RED     = "\033[31m"
GREEN   = "\033[32m"
YELLOW  = "\033[33m"
BLUE    = "\033[34m"
MAGENTA = "\033[35m"
CYAN    = "\033[36m"
WHITE   = "\033[37m"
BG_BLUE = "\033[44m"
BG_GREEN  = "\033[42m"
BG_YELLOW = "\033[43m"
BG_RED    = "\033[41m"

# Task definitions
TASKS_DEF = [
    ("k10_c0", "act",  "vc_k10 cluster0",  "6 eps",  100010),
    ("k10_c1", "act",  "vc_k10 cluster1",  "16 eps", 100010),
    ("k10_c2", "act",  "vc_k10 cluster2",  "14 eps", 100010),
    ("k10_c3", "act",  "vc_k10 cluster3",  "5 eps",  100010),
    ("k10_c4", "act",  "vc_k10 cluster4",  "9 eps",  100010),
    ("k10_c5", "act",  "vc_k10 cluster5",  "4 eps",  100010),
    ("k10_c6", "act",  "vc_k10 cluster6",  "5 eps",  100010),
    ("k10_c7", "act",  "vc_k10 cluster7",  "15 eps", 100010),
    ("k10_c8", "act",  "vc_k10 cluster8",  "11 eps", 100010),
    ("k10_c9", "act",  "vc_k10 cluster9",  "15 eps", 100010),
    ("auto_c0","act",  "vc_auto cluster0",  "62 eps", 100010),
    ("auto_c1","act",  "vc_auto cluster1",  "38 eps", 100010),
    ("cls_k10","cls",  "classifier k=10",   "10 cls", 50),
    ("cls_auto","cls", "classifier k=2",    "2 cls",  50),
]


def read_tail(path: Path, nbytes: int = 32768) -> str:
    """Read last nbytes of a file as string."""
    if not path.exists():
        return ""
    try:
        size = path.stat().st_size
        with open(path, "rb") as f:
            f.seek(max(0, size - nbytes))
            return f.read().decode("utf-8", errors="replace")
    except Exception:
        return ""


def parse_scheduler_log():
    """Parse scheduler.log for task start/finish events and GPU assignment."""
    text = read_tail(SCHEDULER_LOG, 65536)
    if not text:
        return {}, {}

    # task -> gpu mapping (latest assignment)
    task_gpu = {}
    # task -> "started" | "finished"
    task_status = {}

    for line in text.split("\n"):
        # [03-04 14:00:00] Started ACT [k10_c0] on GPU 0, PID=12345
        m = re.search(r"Started (?:ACT|classifier) \[(\w+)\] on GPU (\d+), PID=(\d+)", line)
        if m:
            tid, gpu, pid = m.group(1), int(m.group(2)), int(m.group(3))
            task_gpu[tid] = {"gpu": gpu, "pid": pid}
            task_status[tid] = "running"

        # [03-04 14:30:00] GPU 0 finished: k10_c0 (PID 12345)
        m2 = re.search(r"GPU \d+ finished: (\w+)", line)
        if m2:
            tid = m2.group(1)
            task_status[tid] = "finished"

        # All tasks completed!
        if "All tasks completed" in line:
            pass  # handled by individual finish

    return task_status, task_gpu


def is_pid_alive(pid: int) -> bool:
    """Check if a process is still running."""
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def parse_tqdm_progress(text: str) -> dict | None:
    """Parse tqdm output from lerobot training log.

    Pattern: Training:  45%|...| 45000/100010 [1:23:45<1:42:30, 8.94step/s]
    """
    # Find all tqdm lines (they use \r)
    # Split by \r and find lines matching the pattern
    segments = text.split("\r")
    best = None

    for seg in reversed(segments):
        m = re.search(
            r"Training:\s+(\d+)%\|.*?\|\s*(\d+)/(\d+)\s*\[([^\]]*)\]",
            seg
        )
        if m:
            pct = int(m.group(1))
            current = int(m.group(2))
            total = int(m.group(3))
            time_info = m.group(4)

            elapsed_str = ""
            remaining_str = ""
            speed_str = ""
            if "<" in time_info:
                parts = time_info.split("<")
                elapsed_str = parts[0].strip()
                rest = parts[1]
                if "," in rest:
                    remaining_str = rest.split(",")[0].strip()
                    speed_str = rest.split(",")[1].strip()
                else:
                    remaining_str = rest.strip()

            best = {
                "current": current,
                "total": total,
                "pct": pct,
                "elapsed": elapsed_str,
                "remaining": remaining_str,
                "speed": speed_str,
            }
            break

    return best


def parse_checkpoint_progress(text: str) -> int | None:
    """Fallback: parse checkpoint step from INFO lines."""
    matches = re.findall(r"Checkpoint policy after step (\d+)", text)
    if matches:
        return int(matches[-1])
    return None


def parse_classifier_progress(text: str, total_epochs: int) -> dict | None:
    """Parse classifier training log.

    Pattern: Epoch 001/050 | Train Loss: ... Acc: ... | Val Loss: ... Acc: ...
    """
    matches = re.findall(
        r"Epoch\s+(\d+)/(\d+)\s*\|.*?Train.*?Acc:\s*([\d.]+).*?Val.*?Acc:\s*([\d.]+)",
        text
    )
    if matches:
        last = matches[-1]
        epoch = int(last[0])
        total = int(last[1])
        train_acc = float(last[2])
        val_acc = float(last[3])
        return {
            "current": epoch,
            "total": total,
            "pct": int(100 * epoch / max(total, 1)),
            "train_acc": train_acc,
            "val_acc": val_acc,
        }
    return None


def get_log_mtime(task_id: str) -> float | None:
    """Get log file modification time."""
    p = LOG_DIR / f"{task_id}.log"
    if p.exists():
        return p.stat().st_mtime
    return None


def get_task_progress(task_id: str, task_type: str, total: int) -> dict:
    """Get comprehensive progress info for a task."""
    log_path = LOG_DIR / f"{task_id}.log"
    result = {
        "current": 0,
        "total": total,
        "pct": 0,
        "elapsed": "",
        "remaining": "",
        "speed": "",
        "extra": "",
    }

    tail = read_tail(log_path)
    if not tail:
        return result

    if task_type == "act":
        prog = parse_tqdm_progress(tail)
        if prog:
            result.update(prog)
        else:
            # Fallback to checkpoint lines
            step = parse_checkpoint_progress(tail)
            if step:
                result["current"] = step
                result["total"] = total
                result["pct"] = int(100 * step / max(total, 1))

        # Check for errors
        if "Error" in tail[-500:] or "Traceback" in tail[-500:]:
            result["extra"] = f"{RED}ERROR{RESET}"

    elif task_type == "cls":
        prog = parse_classifier_progress(tail, total)
        if prog:
            result["current"] = prog["current"]
            result["total"] = prog["total"]
            result["pct"] = prog["pct"]
            result["extra"] = (
                f"train_acc={prog['train_acc']:.3f} "
                f"val_acc={prog['val_acc']:.3f}"
            )

    return result


def make_bar(pct: int, width: int = 30, color: str = GREEN) -> str:
    """Create a colored progress bar."""
    filled = int(width * pct / 100)
    empty = width - filled
    bar_fill = "█" * filled
    bar_empty = "░" * empty
    return f"{color}{bar_fill}{DIM}{bar_empty}{RESET}"


def format_time_short(s: str) -> str:
    """Keep time string short."""
    if not s:
        return ""
    # Already in format like 2:40:27 or 13:37 or 1:42:30
    return s


def get_terminal_width() -> int:
    """Get terminal width."""
    try:
        return os.get_terminal_size().columns
    except Exception:
        return 100


def get_gpu_info() -> dict:
    """Get GPU utilization, memory info, and running process names."""
    try:
        import subprocess
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        gpus = {}
        for line in r.stdout.strip().split("\n"):
            parts = [x.strip() for x in line.split(",")]
            if len(parts) >= 4:
                idx = int(parts[0])
                gpus[idx] = {
                    "util": int(parts[1]),
                    "mem_used": int(parts[2]),
                    "mem_total": int(parts[3]),
                    "external": False,
                    "ext_name": "",
                }

        # Check for external processes on each GPU
        r2 = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        r3 = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        # Map UUID -> GPU index
        uuid_to_idx = {}
        for line in r3.stdout.strip().split("\n"):
            parts = [x.strip() for x in line.split(",")]
            if len(parts) >= 2:
                uuid_to_idx[parts[1]] = int(parts[0])

        for line in r2.stdout.strip().split("\n"):
            parts = [x.strip() for x in line.split(",")]
            if len(parts) >= 3:
                gpu_uuid = parts[0]
                proc_name = parts[2]
                gid = uuid_to_idx.get(gpu_uuid, -1)
                if gid >= 0 and gid in gpus:
                    # Check if this is NOT a VC-ACT task
                    short = proc_name.split("/")[-1]
                    gpus[gid]["external"] = True
                    gpus[gid]["ext_name"] = short

        return gpus
    except Exception:
        return {}


def render(refresh_count: int):
    """Render the full dashboard."""
    tw = get_terminal_width()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    task_status, task_gpu = parse_scheduler_log()
    gpu_info = get_gpu_info()

    # Determine task states
    tasks_state = []
    for tid, ttype, desc, eps_str, total in TASKS_DEF:
        state = "pending"
        gpu = -1
        pid = 0

        if tid in task_status:
            if task_status[tid] == "finished":
                state = "done"
            elif task_status[tid] == "running":
                ginfo = task_gpu.get(tid, {})
                pid = ginfo.get("pid", 0)
                gpu = ginfo.get("gpu", -1)
                if pid and is_pid_alive(pid):
                    state = "running"
                else:
                    state = "done"

        log_exists = (LOG_DIR / f"{tid}.log").exists()
        progress = get_task_progress(tid, ttype, total) if log_exists else {
            "current": 0, "total": total, "pct": 0,
            "elapsed": "", "remaining": "", "speed": "", "extra": ""
        }

        # Detect error/crash: log exists, task "finished" but progress < 90%
        if state == "done" and progress["pct"] < 90 and log_exists:
            tail = read_tail(LOG_DIR / f"{tid}.log", 2048)
            if "Traceback" in tail or "Error" in tail or "RuntimeError" in tail:
                state = "error"

        if state == "done":
            progress["pct"] = 100
            progress["current"] = total

        tasks_state.append({
            "id": tid, "type": ttype, "desc": desc, "eps": eps_str,
            "total": total, "state": state, "gpu": gpu, "pid": pid,
            **progress,
        })

    # Build output lines
    lines = []
    sep = f"{DIM}{'─' * min(tw, 90)}{RESET}"

    # Header
    lines.append("")
    title = f"   VC-ACT Training Monitor  ·  {now}   "
    w = len(title)
    lines.append(f"  {BOLD}{CYAN}╔{'═' * w}╗{RESET}")
    lines.append(f"  {BOLD}{CYAN}║{title}║{RESET}")
    lines.append(f"  {BOLD}{CYAN}╚{'═' * w}╝{RESET}")
    lines.append("")

    # GPU Status
    lines.append(f"  {BOLD}GPU Status{RESET}")
    lines.append(sep)

    gpu_tasks = {}  # gpu_id -> task info
    for t in tasks_state:
        if t["state"] == "running" and t["gpu"] >= 0:
            gpu_tasks[t["gpu"]] = t

    for gid in range(4):
        gi = gpu_info.get(gid, {})
        util = gi.get("util", 0)
        mem_used = gi.get("mem_used", 0)
        mem_total = gi.get("mem_total", 1)
        mem_pct = int(100 * mem_used / max(mem_total, 1))

        if gid in gpu_tasks:
            t = gpu_tasks[gid]
            bar = make_bar(t["pct"], width=25)
            pct_str = f"{t['pct']:3d}%"

            if t["type"] == "act":
                step_str = f"{t['current']:>6d}/{t['total']}"
                time_str = ""
                if t["remaining"]:
                    time_str = f"ETA {YELLOW}{t['remaining']}{RESET}"
                elif t["elapsed"]:
                    time_str = f"elapsed {t['elapsed']}"
                speed_str = f" {DIM}{t['speed']}{RESET}" if t["speed"] else ""
                lines.append(
                    f"  GPU {gid}  {bar} {pct_str}  "
                    f"{BOLD}{t['id']}{RESET} {step_str} "
                    f"{time_str}{speed_str}"
                )
            else:
                lines.append(
                    f"  GPU {gid}  {bar} {pct_str}  "
                    f"{BOLD}{t['id']}{RESET} epoch {t['current']}/{t['total']} "
                    f"{t.get('extra', '')}"
                )
        elif gi.get("external") and gi.get("mem_used", 0) > 100:
            # GPU occupied by external process
            ext = gi.get("ext_name", "unknown")
            lines.append(
                f"  GPU {gid}  {DIM}{'░' * 25} occupied{RESET}  "
                f"{MAGENTA}{ext}{RESET}  "
                f"{DIM}mem {mem_used}M/{mem_total}M ({mem_pct}%){RESET}"
            )
        else:
            lines.append(
                f"  GPU {gid}  {DIM}{'░' * 25} idle{RESET}  "
                f"{DIM}mem {mem_used}M/{mem_total}M ({mem_pct}%){RESET}"
            )
    lines.append("")

    # Overall progress
    done_count = sum(1 for t in tasks_state if t["state"] == "done")
    running_count = sum(1 for t in tasks_state if t["state"] == "running")
    pending_count = sum(1 for t in tasks_state if t["state"] == "pending")
    error_count = sum(1 for t in tasks_state if t["state"] == "error")
    total_count = len(tasks_state)
    overall_pct = int(100 * done_count / total_count) if total_count > 0 else 0

    summary_parts = [
        f"{GREEN}{done_count} done{RESET}",
        f"{YELLOW}{running_count} running{RESET}",
        f"{DIM}{pending_count} pending{RESET}",
    ]
    if error_count:
        summary_parts.append(f"{RED}{error_count} failed{RESET}")

    lines.append(f"  {BOLD}Overall Progress{RESET}  "
                 f"{make_bar(overall_pct, 30)} {overall_pct}%  "
                 f"({' / '.join(summary_parts)})")
    lines.append("")

    # Task table
    lines.append(f"  {BOLD}Task Queue{RESET}")
    lines.append(sep)
    lines.append(
        f"  {DIM}{'Status':^8} {'Task':<9} {'Description':<20} {'Data':>7}  "
        f"{'Progress':<32} {'Time Info'}{RESET}"
    )
    lines.append(sep)

    for t in tasks_state:
        # Status icon
        if t["state"] == "done":
            icon = f"{GREEN}  ✓  {RESET}"
        elif t["state"] == "error":
            icon = f"{RED}  ✗  {RESET}"
        elif t["state"] == "running":
            spin = "▶▸"[refresh_count % 2]
            icon = f"{YELLOW}  {spin}  {RESET}"
        else:
            icon = f"{DIM}  ·  {RESET}"

        # Progress bar (compact)
        if t["state"] == "running":
            bar = make_bar(t["pct"], width=20)
            pct_s = f"{t['pct']:3d}%"
            prog_str = f"{bar} {pct_s}"
        elif t["state"] == "done":
            bar = make_bar(100, width=20, color=GREEN)
            prog_str = f"{bar} {GREEN}100%{RESET}"
        elif t["state"] == "error":
            bar = make_bar(t["pct"], width=20, color=RED)
            pct_s = f"{t['pct']:3d}%"
            prog_str = f"{bar} {pct_s}"
        else:
            prog_str = f"{DIM}{'·' * 20}    {RESET}"

        # Time/step info
        if t["state"] == "running":
            if t["type"] == "act":
                step_s = f"{t['current']:>6d}/{t['total']}"
                if t["remaining"]:
                    time_s = f"  ETA {YELLOW}{t['remaining']}{RESET}  {DIM}{t['speed']}{RESET}"
                elif t["elapsed"]:
                    time_s = f"  {t['elapsed']}  {DIM}{t['speed']}{RESET}"
                else:
                    time_s = ""
                info = f"{step_s}{time_s}"
            else:
                info = f"epoch {t['current']}/{t['total']}  {t.get('extra', '')}"
        elif t["state"] == "done":
            if t.get("elapsed"):
                info = f"{GREEN}completed{RESET} ({t['elapsed']})"
            else:
                info = f"{GREEN}completed{RESET}"
        elif t["state"] == "error":
            info = f"{RED}FAILED{RESET} at step {t.get('current', '?')}"
        else:
            info = ""

        lines.append(
            f"  {icon}{t['id']:<9} {t['desc']:<20} {t['eps']:>7}  "
            f"{prog_str}  {info}"
        )

    lines.append(sep)

    # Footer
    scheduler_running = SCHEDULER_LOG.exists() and (time.time() - SCHEDULER_LOG.stat().st_mtime < 120)
    if scheduler_running:
        sched_status = f"{GREEN}● running{RESET}"
    elif SCHEDULER_LOG.exists():
        sched_status = f"{DIM}○ stopped{RESET}"
    else:
        sched_status = f"{RED}○ not started{RESET}"

    lines.append(f"  Scheduler: {sched_status}  |  "
                 f"Log: {DIM}{SCHEDULER_LOG}{RESET}")
    lines.append(f"  {DIM}Press Ctrl+C to exit{RESET}")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="VC-ACT Training Monitor")
    parser.add_argument("--refresh", type=float, default=1.0,
                        help="Refresh interval in seconds (default: 1)")
    args = parser.parse_args()

    count = 0
    try:
        while True:
            # Clear screen and move cursor to top
            sys.stdout.write("\033[2J\033[H")
            sys.stdout.write(render(count))
            sys.stdout.flush()
            count += 1
            time.sleep(args.refresh)
    except KeyboardInterrupt:
        sys.stdout.write("\033[2J\033[H")
        print("Monitor stopped.")


if __name__ == "__main__":
    main()
