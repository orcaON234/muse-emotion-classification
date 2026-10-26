"""Record Muse 2 EEG from a BlueMuse LSL outlet on Windows.
.\venv\Scripts\activate.ps1
Commands:
    python muse_eeg.py start
    python muse_eeg.py stop
    python muse_eeg.py plot recordings/<recording>.csv

BlueMuse must be streaming EEG before `start`.
"""

import argparse
import csv
import json
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pylsl import StreamInlet, resolve_byprop
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "recordings"
SESSION_FILE = DATA_DIR / "current_session.json"
DEFAULT_CHANNELS = ["TP9", "AF7", "AF8", "TP10", "Right AUX"]


def paths(session_id):
    return (
        DATA_DIR / f"{session_id}.stop",
        DATA_DIR / f"{session_id}.ready",
        DATA_DIR / f"{session_id}.done.json",
    )


def write_json(path, obj):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    temp.replace(path)


def channel_labels(info):
    """Use LSL channel metadata, with standard Muse channel names as fallback."""
    n = info.channel_count()
    labels = []
    try:
        channel = info.desc().child("channels").child("channel")
        for i in range(n):
            label = channel.child_value("label") if not channel.empty() else ""
            labels.append(label or (DEFAULT_CHANNELS[i] if i < 5 else f"CH{i+1}"))
            channel = channel.next_sibling()
    except Exception:
        labels = [DEFAULT_CHANNELS[i] if i < 5 else f"CH{i+1}"
                  for i in range(n)]
    # Keep column names unique, even if source metadata repeats a label.
    return [f"{name}_{i+1}" if name in labels[:i] else name
            for i, name in enumerate(labels)]


def capture(session_id, csv_path):
    """Run in a separate background Python process."""
    stop_path, ready_path, done_path = paths(session_id)
    samples = 0
    error = None
    try:
        

        streams = resolve_byprop("type", "EEG", timeout=12)
        if not streams:
            raise RuntimeError("No EEG LSL stream found. Check BlueMuse is streaming EEG.")

        # Prefer a Muse stream if other EEG devices are connected.
        stream = next((s for s in streams if "muse" in s.name().lower()), streams[0])
        inlet = StreamInlet(stream, max_buflen=60)
        labels = channel_labels(inlet.info(timeout=5))
        first_time = None

        with csv_path.open("w", newline="", encoding="utf-8") as out:
            writer = csv.writer(out)
            writer.writerow(["lsl_timestamp", "time_s", *labels])
            while not stop_path.exists():
                sample, timestamp = inlet.pull_sample(timeout=0.25)
                if sample is None:
                    continue
                if first_time is None:
                    first_time = timestamp
                writer.writerow([timestamp, timestamp - first_time, *sample])
                samples += 1
                if samples == 1:
                    out.flush()
                    ready_path.write_text("ready", encoding="utf-8")
                elif samples % 256 == 0:
                    out.flush()

    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        write_json(done_path, {"samples": samples, "error": error})


def start():
    DATA_DIR.mkdir(exist_ok=True)
    if SESSION_FILE.exists():
        old = json.loads(SESSION_FILE.read_text(encoding="utf-8"))
        if not paths(old["id"])[2].exists():
            print("There may already be an active recording. Run 'stop' first.")
            print(f"If the worker crashed, inspect: {old['log']}")
            return 1

    session_id = uuid.uuid4().hex[:12]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = DATA_DIR / f"muse_{stamp}_{session_id}.csv"
    log_path = DATA_DIR / f"muse_{stamp}_{session_id}.log"
    stop_path, ready_path, done_path = paths(session_id)

    state = {"id": session_id, "csv": str(csv_path), "log": str(log_path)}
    write_json(SESSION_FILE, state)
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)  # Windows background mode
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "_capture",
             session_id, str(csv_path)],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=flags,
        )

    # Confirm a real EEG sample was received (not just an LSL stream listing).
    deadline = time.monotonic() + 16
    while time.monotonic() < deadline:
        if ready_path.exists():
            print("Recording started. Run: python muse_eeg.py stop")
            print(f"CSV: {csv_path}")
            return 0
        if done_path.exists():
            result = json.loads(done_path.read_text(encoding="utf-8"))
            print(f"Could not start recording: {result['error']}")
            return 1
        time.sleep(0.2)
    print("Worker launched, but no EEG samples received yet.")
    print(f"Check BlueMuse; see log: {log_path}")
    return 1


def plot(csv_path):
    import matplotlib.pyplot as plt
    import pandas as pd

    csv_path = Path(csv_path)
    data = pd.read_csv(csv_path)
    channels = [name for name in data.columns
                if name not in ("lsl_timestamp", "time_s")]
    if data.empty or not channels:
        print("No EEG samples to plot.")
        return 1

    fig, axes = plt.subplots(len(channels), 1, figsize=(13, 2.3 * len(channels)),
                             sharex=True, squeeze=False, layout="constrained")
    for ax, channel in zip(axes[:, 0], channels):
        ax.plot(data["time_s"], data[channel], lw=0.65)
        ax.set_ylabel(f"{channel}\n(µV)")
        ax.grid(True, alpha=0.3)
    axes[-1, 0].set_xlabel("Time (seconds)")
    fig.suptitle(f"Muse EEG | {csv_path.name}")

    png_path = csv_path.with_suffix(".png")
    fig.savefig(png_path, dpi=150)
    print(f"Plot saved: {png_path}")
    plt.show()
    return 0


def stop():
    if not SESSION_FILE.exists():
        print("No recording session found. Run: python muse_eeg.py start")
        return 1
    session = json.loads(SESSION_FILE.read_text(encoding="utf-8"))
    stop_path, _ready_path, done_path = paths(session["id"])
    if not done_path.exists():
        stop_path.touch()
        print("Stopping EEG recording and finishing CSV...")
        deadline = time.monotonic() + 20
        while not done_path.exists() and time.monotonic() < deadline:
            time.sleep(0.2)
    if not done_path.exists():
        print(f"Recorder did not finish. Check log: {session['log']}")
        return 1

    result = json.loads(done_path.read_text(encoding="utf-8"))
    print(f"Captured {result['samples']} EEG samples.")
    if result["error"]:
        print(f"Recorder error: {result['error']}")
    csv_path = Path(session["csv"])
    if not csv_path.exists() or result["samples"] == 0:
        return 1
    print(f"CSV saved: {csv_path}")
    return plot(csv_path)


def main():
    parser = argparse.ArgumentParser(description="Record Muse 2 EEG from BlueMuse LSL")
    parser.add_argument("command", choices=["start", "stop", "plot", "_capture"])
    parser.add_argument("args", nargs="*")
    args = parser.parse_args()
    if args.command == "start":
        return start()
    if args.command == "stop":
        return stop()
    if args.command == "plot":
        if len(args.args) != 1:
            parser.error("plot requires the path to a saved CSV file")
        return plot(args.args[0])
    if len(args.args) != 2:
        parser.error("internal _capture requires session ID and output CSV")
    capture(args.args[0], Path(args.args[1]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
