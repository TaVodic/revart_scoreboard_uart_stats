#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
REVART Timing Protocol - Time Integrity Benchmark
=================================================
Purpose:
    Validate that when the "Clock running" flag (T) is 1, the time difference conveyed
    by consecutive protocol messages matches the elapsed wall-clock time.
    Useful for benchmarking the timekeeper and transport timings.

Input sources:
    - Serial port (requires pyserial): --port COM3 --baud 115200
    - File with captured frames (one message per line): --file frames.txt
    - STDIN (default): e.g., tail -f frames.txt | python benchmark.py
    - python revart_time_integrity_benchmark.py --port COM2 --baud 38400 --limit 1000 --hist-wall --hist-game

Output:
    - Console summary with jitter/drift stats
    - Optional CSV log via --csv out.csv for offline analysis

Assumptions:
    - Messages are 55 ASCII chars, LF-terminated.
    - CRC is 8-bit XOR over bytes STX..ETX inclusive, encoded as 2 uppercase hex chars.
    - Game clock is *usually* a countdown; script auto-detects direction on the fly.
    - Only evaluates periods/modes where MM:SS:DD encodes game time (P in {'1','2','3','4','5','6'}).
      For P in {'7','8'} (wall clock / pre-game), entries are ignored for benchmark.

Author: ChatGPT for Revart s.r.o.
License: MIT
"""

import sys
import time
import re
import csv
import math
import argparse
from dataclasses import dataclass
from typing import Optional, List

STX = 0x02
ETX = 0x03
LF  = 0x0A
MESSAGE_LEN = 55

def xor_crc_ascii_hex(segment_bytes: bytes) -> str:
    acc = 0
    for b in segment_bytes:
        acc ^= b
    return f"{acc:02X}"

@dataclass
class Parsed:
    raw: str
    recv_ts: float     # wall clock (time.time())
    valid_crc: bool
    running: bool      # T flag
    mode: str          # P char
    clock_text: str    # "MM:SS:DD" (or HH:MM:SS for P=7/8)
    mm: int
    ss: int
    dd: int            # hundredths for P 1..6; seconds for 7/8
    total_hundredths: Optional[int]  # None for P=7/8
    home_score: int
    away_score: int

def parse_message(line: str, recv_ts: Optional[float]=None) -> Parsed:
    if len(line) != MESSAGE_LEN:
        raise ValueError(f"Message length must be {MESSAGE_LEN}, got {len(line)}")
    if line[0] != chr(STX) or line[51] != chr(ETX) or line[-1] != chr(LF):
        raise ValueError("Bad framing (STX/ETX/LF)")
    body = line[:52].encode('ascii', errors='strict')
    crc_rx = line[52:54]
    crc_ok = (xor_crc_ascii_hex(body).upper() == crc_rx.upper())

    clock_text = line[1:9]
    if not re.match(r"^\d{2}:\d{2}:\d{2}$", clock_text):
        raise ValueError("Clock field malformed")

    T = line[9]
    P = line[10]
    running = (T == '1')

    mm, ss, dd = clock_text.split(":")
    MM = int(mm); SS = int(ss); DD = int(dd)

    sc = line[11:15]
    if not sc.isdigit():
        raise ValueError("Score field malformed")
    home = int(sc[:2]); away = int(sc[2:])

    if P in {'1','2','3','4','5','6'}:
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

@dataclass
class Sample:
    dt_wall_ms: float
    dt_game_ms: float
    error_ms: float
    direction: str     # 'down' / 'up' / 'none'
    mode: str
    crc_ok: bool
    t_prev: float
    t_curr: float

class Benchmark:
    def __init__(self, tolerance_ms: float=30.0, ignore_bad_crc: bool=True):
        self.tolerance_ms = tolerance_ms
        self.ignore_bad_crc = ignore_bad_crc
        self.prev: Optional[Parsed] = None
        self.samples: List[Sample] = []
        self.total_frames = 0
        self.bad_crc = 0
        self.skipped = 0

    def feed(self, line: str, recv_ts: Optional[float]=None):
        self.total_frames += 1
        try:
            p = parse_message(line, recv_ts=recv_ts)
        except Exception:
            self.skipped += 1
            return

        if not p.valid_crc:
            self.bad_crc += 1
            if self.ignore_bad_crc:
                return

        if not p.running or p.total_hundredths is None:
            self.prev = p
            return

        if self.prev is not None and self.prev.total_hundredths is not None and self.prev.running:
            dt_wall = (p.recv_ts - self.prev.recv_ts) * 1000.0
            diff = p.total_hundredths - self.prev.total_hundredths
            if diff == 0:
                direction = 'none'; dt_game_ms = 0.0
            elif diff < 0:
                direction = 'down'; dt_game_ms = (-diff) * 10.0
            else:
                direction = 'up'; dt_game_ms = (diff) * 10.0
            error = dt_game_ms - dt_wall
            self.samples.append(Sample(dt_wall, dt_game_ms, error, direction, p.mode, p.valid_crc, self.prev.recv_ts, p.recv_ts))

        self.prev = p

    def stats(self):
        n = len(self.samples)
        if n == 0:
            return None
        errors = [s.error_ms for s in self.samples]
        game_errors = [s.dt_game_ms for s in self.samples]
        wall_errors = [s.dt_wall_ms for s in self.samples]
        abs_err = [abs(e) for e in errors]
        mean_err = sum(errors)/n
        rms = (sum(e*e for e in errors)/n) ** 0.5
        p95 = sorted(abs_err)[int(0.95*n)-1] if n > 0 else float('nan')
        within = sum(1 for e in abs_err if e <= self.tolerance_ms)
        pct_within = 100.0 * within / n
        wall = [s.dt_wall_ms for s in self.samples if s.dt_wall_ms > 0]
        avg_period = sum(wall)/len(wall) if wall else float('nan')
        dt_wall = [s.dt_wall_ms for s in self.samples]
        dt_game = [s.dt_game_ms for s in self.samples]
        wall_mean = sum(wall_errors)/n
        game_mean = sum(game_errors)/n
        wall_rms = (sum(e*e for e in wall_errors)/n) ** 0.5
        game_rms = (sum(e*e for e in game_errors)/n) ** 0.5
        wall_p95 = sorted(wall_errors)[int(0.95*n)-1] if n > 0 else float('nan')
        game_p95 = sorted(game_errors)[int(0.95*n)-1] if n > 0 else float('nan')
        return {
            "total_frames": self.total_frames,
            "used_samples": n,
            "bad_crc": self.bad_crc,
            "skipped": self.skipped,
            "avg_period_ms": avg_period,
            "mean_error_ms": mean_err,
            "rms_error_ms": rms,
            "p95_abs_error_ms": p95,
            "within_tol_count": within,
            "within_tol_pct": pct_within,
            "tolerance_ms": self.tolerance_ms,
            "all_errors": errors,
            "dt_wall": dt_wall, 
            "dt_game": dt_game, 
            "wall_mean": wall_mean,
            "game_mean": game_mean,
            "wall_rms": wall_rms,
            "game_rms": game_rms,
            "wall_p95": wall_p95,
            "game_p95": game_p95,
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
        print(f"All errors: {s['all_errors']}")
        
        print(f"Mean wall dt: {s['wall_mean']:.2f} ms")
        print(f"RMS wall dt: {s['wall_rms']:.2f} ms")
        print(f"95th % wall dt: {s['wall_p95']:.2f} ms")
        print(f"All wall: {s['dt_wall']}")
        
        print(f"Mean game dt: {s['game_mean']:.2f} ms")
        print(f"RMS game dt: {s['game_rms']:.2f} ms")
        print(f"95th % game dt: {s['game_p95']:.2f} ms")
        print(f"All game: {s['dt_game']}")

    def write_csv(self, path: str):
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(["t_prev","t_curr","dt_wall_ms","dt_game_ms","error_ms","direction","mode","crc_ok"])
            for s in self.samples:
                w.writerow([f"{s.t_prev:.6f}", f"{s.t_curr:.6f}", f"{s.dt_wall_ms:.3f}",
                            f"{s.dt_game_ms:.3f}", f"{s.error_ms:.3f}", s.direction, s.mode, int(s.crc_ok)])
    
    def plot_wall_hist(self, bins: int = 100, save_path: Optional[str] = None):
        wall = [s.dt_wall_ms for s in self.samples if s.dt_wall_ms >= 0]
        if not wall:
            print("No wall dt samples to plot.")
            return

        import matplotlib.pyplot as plt

        maxv = bins
        # 1 ms bins from 0..maxv (so 0 is always a bin edge)
        edges = list(range(0, int(math.ceil(maxv)) + 2))  # +2 to include last edge

        plt.figure()
        plt.hist(wall, bins=edges)
        plt.xlabel("Wall interval dt [ms]")
        plt.ylabel("Count")
        plt.title("Histogram of wall-clock intervals")
        plt.grid(True)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
        else:
            plt.show()


    def plot_game_hist(self, bins: int = 100, save_path: Optional[str] = None):
        game = [s.dt_game_ms for s in self.samples if s.dt_game_ms >= 0]
        if not game:
            print("No game dt samples to plot.")
            return

        import matplotlib.pyplot as plt

        maxv = bins
        edges = list(range(0, int(math.ceil(maxv)) + 2))

        plt.figure()
        plt.hist(game, bins=edges)
        plt.xlabel("Game interval dt [ms]")
        plt.ylabel("Count")
        plt.title("Histogram of game-clock intervals")
        plt.grid(True)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
        else:
            plt.show()




def iter_source(args):
    if args.file:
        with open(args.file, 'r', encoding='ascii', errors='ignore') as f:
            for line in f:
                yield line
    elif args.port:
        try:
            import serial  # type: ignore
        except Exception:
            print("pyserial is required for --port", file=sys.stderr)
            sys.exit(2)

        # Non-blocking serial
        ser = serial.Serial(args.port, baudrate=args.baud, timeout=0)
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

                        # Timestamp as close as possible to frame boundary
                        ts = time.perf_counter()

                        try:
                            s = frame.decode("ascii", errors="strict")
                        except Exception:
                            continue

                        # Yield both frame and timestamp
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
            yield line

def main():
    ap = argparse.ArgumentParser(description="REVART time integrity benchmark")
    ap.add_argument("--port", help="Serial port (e.g., COM5 or /dev/ttyUSB0)")
    ap.add_argument("--baud", type=int, default=115200, help="Baud rate")
    ap.add_argument("--file", help="Read frames from a file (one per line)")
    ap.add_argument("--csv", help="Write detailed samples to CSV")
    ap.add_argument("--tolerance-ms", type=float, default=30.0, help="Pass/fail tolerance in ms")
    ap.add_argument("--include-bad-crc", action="store_true", help="Include samples even if CRC is bad")
    ap.add_argument("--limit", type=int, default=0, help="Stop after N frames (0=unlimited)")
    ap.add_argument("--hist-wall", nargs='?', const="wall_hist.png", help="Create a histogram of wall dt intervals (optionally specify PNG path, default=wall_hist.png)")
    ap.add_argument("--hist-game", nargs='?', const="game_hist.png", help="Create a histogram of game dt intervals (optionally specify PNG path, default=game_hist.png)")


    args = ap.parse_args()

    bench = Benchmark(tolerance_ms=args.tolerance_ms, ignore_bad_crc=not args.include_bad_crc)

    count = 0
    try:
        for item in iter_source(args):
            if not item:
                continue

            if isinstance(item, tuple):
                line, ts = item
            else:
                line, ts = item, None

            if len(line) >= MESSAGE_LEN:
                frame = line[-MESSAGE_LEN:]
                bench.feed(frame, recv_ts=ts)
                
            count += 1
            if args.limit and count >= args.limit:
                break
    except KeyboardInterrupt:
        pass

    bench.print_summary()
    if args.csv:
        bench.write_csv(args.csv)
        print(f"Wrote CSV: {args.csv}")
    if args.hist_wall:
        bench.plot_wall_hist(save_path=args.hist_wall)
        print(f"Wrote wall dt histogram: {args.hist_wall}")
    if args.hist_game:
        bench.plot_game_hist(save_path=args.hist_game)
        print(f"Wrote game dt histogram: {args.hist_game}")


if __name__ == "__main__":
    main()
