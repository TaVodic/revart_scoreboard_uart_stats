#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
REVART Timing Protocol - Time Integrity Benchmark (optimized for long capture)
=============================================================================
Purpose:
    Validate that when the "Clock running" flag (T) is 1, the time difference conveyed
    by consecutive protocol messages matches the elapsed wall-clock time.

Changes vs older version:
    - Serial reading is non-blocking (reduces bursty reads caused by timeouts).
    - Uses time.perf_counter() for recv timestamps (monotonic, better for dt).
    - Streams series CSV with auto-flush (no giant console arrays).
    - Removes --csv (no full per-sample detail).
    - Writes final stats into the beginning of series.csv on exit (Ctrl+C),
      by spooling samples into a temporary file and then assembling the final CSV.

Inputs:
    - Serial port: --port COM3 --baud 38400
    - File: --file frames.txt
    - STDIN (default): tail -f frames.txt | python bm.py
    python bm.py --port COM15 --baud 38400 --series series.csv --hist-wall wall.png --hist-game game.png --flush-every 50 --heartbeat-s 5

Output:
    - Console summary (compact)
    - --series series.csv (stats header + dt_wall_ms, dt_game_ms, error_ms rows)
    - Optional histograms: --hist-wall wall.png --hist-game game.png (100 bins, 0..max)

License: MIT
"""

import sys
import time
import re
import csv
import math
import os
import argparse
from dataclasses import dataclass
from typing import Optional, List, Iterable, Tuple, Union

STX = 0x02
ETX = 0x03
LF  = 0x0A
MESSAGE_LEN = 55

def get_app_dir() -> str:
    if getattr(sys, "frozen", False):
        # Running as bundled exe
        return os.path.dirname(sys.executable)
    # Running as script
    return os.path.dirname(os.path.abspath(__file__))

APP_DIR = get_app_dir()
DATA_DIR = os.path.join(APP_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

def xor_crc_ascii_hex(segment_bytes: bytes) -> str:
    acc = 0
    for b in segment_bytes:
        acc ^= b
    return f"{acc:02X}"


@dataclass
class Parsed:
    raw: str
    recv_ts: float     # monotonic clock (time.perf_counter())
    valid_crc: bool
    running: bool      # T flag
    mode: str          # P char
    clock_text: str    # "MM:SS:DD" (or HH:MM:SS for P=7/8)
    mm: int
    ss: int
    dd: int
    total_hundredths: Optional[int]  # None for P=7/8
    home_score: int
    away_score: int


def parse_message(line: str, recv_ts: Optional[float] = None) -> Parsed:
    if len(line) != MESSAGE_LEN:
        raise ValueError(f"Message length must be {MESSAGE_LEN}, got {len(line)}")
    if line[0] != chr(STX) or line[51] != chr(ETX) or line[-1] != chr(LF):
        raise ValueError("Bad framing (STX/ETX/LF)")
    body = line[:52].encode("ascii", errors="strict")
    crc_rx = line[52:54]
    crc_ok = (xor_crc_ascii_hex(body).upper() == crc_rx.upper())

    clock_text = line[1:9]
    if not re.match(r"^\d{2}:\d{2}:\d{2}$", clock_text):
        raise ValueError("Clock field malformed")

    T = line[9]
    P = line[10]
    running = (T == "1")

    mm, ss, dd = clock_text.split(":")
    MM = int(mm); SS = int(ss); DD = int(dd)

    sc = line[11:15]
    if not sc.isdigit():
        raise ValueError("Score field malformed")
    home = int(sc[:2]); away = int(sc[2:])

    if P in {"1","2","3","4","5","6"}:
        total_hundredths = (MM * 60 + SS) * 100 + DD
    else:
        total_hundredths = None

    return Parsed(
        raw=line,
        recv_ts=time.perf_counter() if recv_ts is None else recv_ts,
        valid_crc=crc_ok,
        running=running,
        mode=P,
        clock_text=clock_text,
        mm=MM, ss=SS, dd=DD,
        total_hundredths=total_hundredths,
        home_score=home, away_score=away
    )


def normalize_frame(line: str) -> Optional[str]:
    """
    Try to recover a full 55-char frame.

    Accepts:
      - exact 55-char frames (preferred)
      - lines >= 55 (takes last 55 chars)
      - 54-char frames missing LF (adds LF)
    Returns None if cannot normalize.
    """
    if not line:
        return None

    if len(line) == MESSAGE_LEN:
        return line

    # If capture stripped trailing '\n' (LF), we might have 54 chars.
    if len(line) == MESSAGE_LEN - 1:
        # If it looks like a frame missing LF, add it
        if line and line[0] == chr(STX) and len(line) > 51 and line[51] == chr(ETX):
            return line + chr(LF)
        return None

    if len(line) > MESSAGE_LEN:
        cand = line[-MESSAGE_LEN:]
        if cand and cand[0] == chr(STX) and cand[51] == chr(ETX) and cand[-1] == chr(LF):
            return cand
        # maybe missing LF inside longer line; try adding LF if last char isn't LF
        if len(cand) == MESSAGE_LEN and cand[0] == chr(STX) and cand[51] == chr(ETX):
            if cand[-1] != chr(LF):
                cand = cand[:-1] + chr(LF)
                if cand[-1] == chr(LF):
                    return cand
        return None

    return None


class SeriesWriter:
    """
    Stream samples into a temporary CSV file with auto-flush.
    On finalize(), assemble final CSV with a stats header at the beginning.
    """
    def __init__(self, final_path: str, flush_every: int = 1):
        self.final_path = final_path
        self.tmp_path = final_path + ".tmp"
        self.flush_every = max(1, int(flush_every))
        self._f = open(self.tmp_path, "w", newline="")
        self._w = csv.writer(self._f)
        self._count = 0
        # Only write data rows here; final file will get header + stats.
        # Keep temp as "pure rows" for easy concatenation.

    def write_row(self, dt_wall_ms: float, dt_game_ms: float, err_ms: float):
        self._w.writerow([f"{dt_wall_ms:.3f}", f"{dt_game_ms:.3f}", f"{err_ms:.3f}"])
        self._count += 1
        if (self._count % self.flush_every) == 0:
            self._f.flush()

    def close(self):
        try:
            self._f.flush()
        except Exception:
            pass
        try:
            self._f.close()
        except Exception:
            pass

    def finalize(self, stats_lines: List[str]):
        """
        Create final_path:
          - stats header as comment lines (starting with '#')
          - CSV header line
          - streamed rows copied from tmp
        """
        self.close()
        with open(self.final_path, "w", newline="") as out:
            for line in stats_lines:
                out.write(f"# {line}\n")
            out.write("#\n")
            out.write("dt_wall_ms,dt_game_ms,error_ms\n")
            with open(self.tmp_path, "r", newline="") as inp:
                for chunk in inp:
                    out.write(chunk)
        try:
            os.remove(self.tmp_path)
        except Exception:
            pass


class Benchmark:
    def __init__(self, tolerance_ms: float = 30.0, ignore_bad_crc: bool = True, series_writer: Optional[SeriesWriter] = None):
        self.tolerance_ms = tolerance_ms
        self.ignore_bad_crc = ignore_bad_crc
        self.prev: Optional[Parsed] = None

        # Keep arrays for stats + hist (reasonable even for ~1M samples)
        self.dt_wall_ms: List[float] = []
        self.dt_game_ms: List[float] = []
        self.err_ms: List[float] = []

        self.total_frames = 0
        self.bad_crc = 0
        self.skipped = 0
        self.within_tol = 0

        self.series_writer = series_writer

    def feed(self, frame: str, recv_ts: Optional[float] = None):
        self.total_frames += 1
        try:
            p = parse_message(frame, recv_ts=recv_ts)
        except Exception:
            self.skipped += 1
            return

        if not p.valid_crc:
            self.bad_crc += 1
            if self.ignore_bad_crc:
                return

        # Ignore non-running or non-game-time modes
        if (p.total_hundredths is None):
            self.prev = p
            return

        if self.prev is not None and self.prev.running and self.prev.total_hundredths is not None:
            dt_wall = (p.recv_ts - self.prev.recv_ts) * 1000.0

            diff = p.total_hundredths - self.prev.total_hundredths
            if diff == 0:
                dt_game = 0.0
            elif diff < 0:
                dt_game = (-diff) * 10.0
            else:
                dt_game = diff * 10.0

            err = dt_game - dt_wall

            self.dt_wall_ms.append(dt_wall)
            self.dt_game_ms.append(dt_game)
            self.err_ms.append(err)

            if abs(err) <= self.tolerance_ms:
                self.within_tol += 1

            if self.series_writer is not None:
                self.series_writer.write_row(dt_wall, dt_game, err)

        self.prev = p

    def stats(self):
        n = len(self.err_ms)
        if n == 0:
            return None

        errors = self.err_ms
        abs_err = [abs(e) for e in errors]

        mean_err = sum(errors) / n
        rms = (sum(e*e for e in errors) / n) ** 0.5
        p95_abs = sorted(abs_err)[max(0, int(0.95*n) - 1)]

        wall = self.dt_wall_ms
        game = self.dt_game_ms

        avg_period = sum(wall) / n
        wall_mean = sum(wall) / n
        game_mean = sum(game) / n
        wall_rms = (sum(x*x for x in wall) / n) ** 0.5
        game_rms = (sum(x*x for x in game) / n) ** 0.5
        wall_p95 = sorted(wall)[max(0, int(0.95*n) - 1)]
        game_p95 = sorted(game)[max(0, int(0.95*n) - 1)]

        pct_within = 100.0 * self.within_tol / n

        zeros_wall = sum(1 for x in wall if x == 0.0)
        zeros_game = sum(1 for x in game if x == 0.0)
        zeros_err  = sum(1 for x in errors if x == 0.0)

        return {
            "total_frames": self.total_frames,
            "used_samples": n,
            "bad_crc": self.bad_crc,
            "skipped": self.skipped,
            "avg_period_ms": avg_period,
            "mean_error_ms": mean_err,
            "rms_error_ms": rms,
            "p95_abs_error_ms": p95_abs,
            "within_tol_count": self.within_tol,
            "within_tol_pct": pct_within,
            "tolerance_ms": self.tolerance_ms,
            "wall_mean": wall_mean,
            "game_mean": game_mean,
            "wall_rms": wall_rms,
            "game_rms": game_rms,
            "wall_p95": wall_p95,
            "game_p95": game_p95,
            "zeros_wall": zeros_wall,
            "zeros_game": zeros_game,
            "zeros_err": zeros_err,
        }

    def print_summary(self):
        s = self.stats()
        if not s:
            print("No running clock samples collected.")
            return
        print(f"Frames total: {s['total_frames']}")
        print(f"Samples used (running & game-time modes): {s['used_samples']}")
        print(f"CRC bad: {s['bad_crc']} (ignored={self.ignore_bad_crc})  Skipped parse: {s['skipped']}")
        print(f"Avg wall period: {s['avg_period_ms']:.2f} ms")
        print(f"Mean error (game - wall): {s['mean_error_ms']:.2f} ms")
        print(f"RMS error: {s['rms_error_ms']:.2f} ms")
        print(f"95th % abs error: {s['p95_abs_error_ms']:.2f} ms")
        print(f"Within ±{s['tolerance_ms']:.1f} ms: {s['within_tol_count']}/{s['used_samples']} ({s['within_tol_pct']:.1f}%)")
        print(f"Mean wall dt: {s['wall_mean']:.2f} ms  RMS: {s['wall_rms']:.2f}  P95: {s['wall_p95']:.2f}")
        print(f"Mean game dt: {s['game_mean']:.2f} ms  RMS: {s['game_rms']:.2f}  P95: {s['game_p95']:.2f}")
        print(f"Zeros: wall={s['zeros_wall']} game={s['zeros_game']} err={s['zeros_err']}")

    def stats_lines_for_file(self) -> List[str]:
        s = self.stats()
        if not s:
            return ["No running clock samples collected."]
        return [
            "REVART time integrity benchmark results",
            f"frames_total={s['total_frames']}",
            f"samples_used={s['used_samples']}",
            f"bad_crc={s['bad_crc']} ignored_bad_crc={self.ignore_bad_crc}",
            f"skipped_parse={s['skipped']}",
            f"avg_wall_period_ms={s['avg_period_ms']:.3f}",
            f"mean_error_ms={s['mean_error_ms']:.3f}",
            f"rms_error_ms={s['rms_error_ms']:.3f}",
            f"p95_abs_error_ms={s['p95_abs_error_ms']:.3f}",
            f"within_tol={s['within_tol_count']} pct_within={s['within_tol_pct']:.2f} tol_ms={s['tolerance_ms']:.3f}",
            f"wall_mean_ms={s['wall_mean']:.3f} wall_rms_ms={s['wall_rms']:.3f} wall_p95_ms={s['wall_p95']:.3f}",
            f"game_mean_ms={s['game_mean']:.3f} game_rms_ms={s['game_rms']:.3f} game_p95_ms={s['game_p95']:.3f}",
            f"zeros_wall={s['zeros_wall']} zeros_game={s['zeros_game']} zeros_err={s['zeros_err']}",
            f"generated_at_epoch={time.time():.3f}",
        ]

    def _fixed_edges_0_to_max(self, values: List[float], bins: int = 100) -> List[float]:
        maxv = max(values) if values else 0.0
        if maxv <= 0.0:
            return [0.0, 1.0]
        step = maxv / bins
        return [i * step for i in range(bins + 1)]

    def plot_wall_hist(self, bins: int = 100, save_path: Optional[str] = None):
        wall = [x for x in self.dt_wall_ms if x >= 0.0]
        if not wall:
            print("No wall dt samples to plot.")
            return
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib is required to plot the histogram.", file=sys.stderr)
            return

        edges = self._fixed_edges_0_to_max(wall, bins=bins)

        plt.figure()
        plt.hist(wall, bins=edges)
        plt.xlabel("Wall interval dt [ms]")
        plt.ylabel("Count")
        plt.title(f"Histogram of wall-clock intervals (N={len(wall)})")
        plt.grid(True)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def plot_game_hist(self, bins: int = 100, save_path: Optional[str] = None):
        game = [x for x in self.dt_game_ms if x >= 0.0]
        if not game:
            print("No game dt samples to plot.")
            return
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib is required to plot the histogram.", file=sys.stderr)
            return

        edges = self._fixed_edges_0_to_max(game, bins=bins)

        plt.figure()
        plt.hist(game, bins=edges)
        plt.xlabel("Game interval dt [ms]")
        plt.ylabel("Count")
        plt.title(f"Histogram of game-clock intervals (N={len(game)})")
        plt.grid(True)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
        else:
            plt.show()


IterItem = Union[str, Tuple[str, float]]


def iter_source(args) -> Iterable[IterItem]:
    if args.file:
        # Keep line endings out; we normalize later.
        with open(args.file, "r", encoding="ascii", errors="ignore") as f:
            for line in f:
                yield line.rstrip("\n")
    elif args.port:
        try:
            import serial  # type: ignore
        except Exception:
            print("pyserial is required for --port", file=sys.stderr)
            sys.exit(2)

        ser = serial.Serial(args.port, baudrate=args.baud, timeout=0)  # non-blocking
        buf = bytearray()
        try:
            while True:
                n = ser.in_waiting
                if n:
                    buf += ser.read(n)
                    while True:
                        i = buf.find(b"\n")
                        if i < 0:
                            break
                        frame = bytes(buf[:i+1])
                        del buf[:i+1]

                        ts = time.perf_counter()
                        try:
                            s = frame.decode("ascii", errors="strict")
                        except Exception:
                            continue
                        yield (s, ts)
                else:
                    time.sleep(0.001)
        finally:
            try:
                ser.close()
            except Exception:
                pass
    else:
        for line in sys.stdin:
            yield line.rstrip("\n")


def main():
    ap = argparse.ArgumentParser(description="REVART time integrity benchmark (optimized)")
    ap.add_argument("--port", help="Serial port (e.g., COM5 or /dev/ttyUSB0)")
    ap.add_argument("--baud", type=int, default=115200, help="Baud rate")
    ap.add_argument("--file", help="Read frames from a file (one per line)")
    ap.add_argument("--series", help="Write dt_wall_ms, dt_game_ms, error_ms series to CSV with stats header")
    ap.add_argument("--tolerance-ms", type=float, default=30.0, help="Pass/fail tolerance in ms")
    ap.add_argument("--include-bad-crc", action="store_true", help="Include samples even if CRC is bad")
    ap.add_argument("--limit", type=int, default=0, help="Stop after N frames (0=unlimited)")
    ap.add_argument("--hist-wall", nargs="?", const="wall_hist.png", help="Create histogram of wall dt (optionally specify PNG path, default=wall_hist.png)")
    ap.add_argument("--hist-game", nargs="?", const="game_hist.png", help="Create histogram of game dt (optionally specify PNG path, default=game_hist.png)")
    ap.add_argument("--flush-every", type=int, default=1, help="Flush series temp file every N rows (default=1, safest)")
    ap.add_argument("--heartbeat-s", type=float, default=60.0, help="Heartbeat print period in seconds (0 = disable)"
)

    args = ap.parse_args()

    series_path = os.path.join(DATA_DIR, args.series) if args.series else None
    series_writer = SeriesWriter(series_path, flush_every=args.flush_every) if series_path else None

    bench = Benchmark(tolerance_ms=args.tolerance_ms,
                      ignore_bad_crc=not args.include_bad_crc,
                      series_writer=series_writer)
    
    start_ts = time.perf_counter()
    last_hb_ts = start_ts

    count = 0
    try:
        for item in iter_source(args):
            
            now = time.perf_counter()

            if args.heartbeat_s > 0 and (now - last_hb_ts) >= args.heartbeat_s:
                elapsed_h = (now - start_ts) / 3600.0
                used = len(bench.err_ms)

                print(
                    f"[HB] {elapsed_h:.2f} h | "
                    f"frames={bench.total_frames} "
                    f"samples={used} "
                )

                last_hb_ts = now
            
            if not item:
                continue

            if isinstance(item, tuple):
                line, ts = item
            else:
                line, ts = item, None

            frame = normalize_frame(line)
            if frame is None:
                continue

            bench.feed(frame, recv_ts=ts)

            count += 1
            if args.limit and count >= args.limit:
                break

    except KeyboardInterrupt:
        pass
    finally:
        # Assemble final series.csv with stats header at the top
        if series_writer is not None:
            stats_lines = bench.stats_lines_for_file()
            series_writer.finalize(stats_lines)
            print(f"Wrote series CSV: {args.series}")

    bench.print_summary()

    if args.hist_wall:
        wall_path = os.path.join(DATA_DIR, args.hist_wall)
        bench.plot_wall_hist(bins=100, save_path=wall_path)
        print(f"Wrote wall dt histogram: {wall_path}")

    if args.hist_game:
        game_path = os.path.join(DATA_DIR, args.hist_game)
        bench.plot_game_hist(bins=100, save_path=game_path)
        print(f"Wrote game dt histogram: {game_path}")



if __name__ == "__main__":
    main()
