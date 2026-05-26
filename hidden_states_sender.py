#!/usr/bin/env python3
"""TCP sender for hidden_states payload latency benchmarks."""

from __future__ import annotations

import argparse
import socket
import statistics
import struct
import time
from typing import Sequence


ACK_BYTES = b"OK"
HEADER_STRUCT = struct.Struct("!Q")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send hidden_states-sized payloads over TCP and report latency statistics."
    )
    parser.add_argument("--host", required=True, help="Receiver host/IP.")
    parser.add_argument("--port", type=int, default=5001, help="Receiver port. Default: 5001")
    parser.add_argument(
        "--payload-bytes",
        type=int,
        default=None,
        help="Explicit payload size in bytes. Overrides hidden_states-derived size.",
    )
    parser.add_argument(
        "--payload-kb",
        type=int,
        default=None,
        help="Explicit payload size in KB. Overrides hidden_states-derived size.",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Default: 1")
    parser.add_argument("--seq-len", type=int, default=260, help="Default: 260")
    parser.add_argument("--hidden-size", type=int, default=2048, help="Default: 2048")
    parser.add_argument(
        "--hidden-divisor",
        type=int,
        default=1,
        help="Divide hidden_size by this value before computing payload bytes. Use 2 for Ndiv=2 half-hidden transfer.",
    )
    parser.add_argument(
        "--bytes-per-element",
        type=int,
        default=2,
        help="fp16/bf16=2, fp32=4. Default: 2",
    )
    parser.add_argument("--warmup-runs", type=int, default=5, help="Default: 5")
    parser.add_argument("--measure-runs", type=int, default=20, help="Default: 20")
    parser.add_argument(
        "--recv-ack-bytes",
        type=int,
        default=len(ACK_BYTES),
        help="Expected ack size. Default: 2",
    )
    parser.add_argument(
        "--tcp-nodelay",
        action="store_true",
        help="Enable TCP_NODELAY on the sender socket.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print one line per measured payload.",
    )
    return parser.parse_args()


def resolve_payload_size(args: argparse.Namespace) -> int:
    if args.payload_bytes is not None:
        return args.payload_bytes
    if args.payload_kb is not None:
        return args.payload_kb * 1024
    if args.hidden_divisor <= 0:
        raise ValueError("--hidden-divisor must be positive.")
    if args.hidden_size % args.hidden_divisor != 0:
        raise ValueError("--hidden-size must be divisible by --hidden-divisor.")
    return (
        args.batch_size
        * args.seq_len
        * (args.hidden_size // args.hidden_divisor)
        * args.bytes_per_element
    )


def recv_exact(sock: socket.socket, num_bytes: int) -> bytes:
    parts = bytearray()
    while len(parts) < num_bytes:
        chunk = sock.recv(num_bytes - len(parts))
        if not chunk:
            raise ConnectionError("Peer closed connection while waiting for ACK.")
        parts.extend(chunk)
    return bytes(parts)


def summarize(times_ms: Sequence[float]) -> tuple[float, float, float, float]:
    avg_ms = statistics.fmean(times_ms)
    min_ms = min(times_ms)
    max_ms = max(times_ms)
    std_ms = statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0
    return avg_ms, min_ms, max_ms, std_ms


def human_mib(num_bytes: int) -> float:
    return num_bytes / (1024.0 * 1024.0)


def main() -> None:
    args = parse_args()
    payload_size = resolve_payload_size(args)
    total_runs = args.warmup_runs + args.measure_runs
    payload = b"\0" * payload_size
    header = HEADER_STRUCT.pack(payload_size)

    print("=" * 79)
    print("hidden_states send benchmark")
    print(f"receiver: {args.host}:{args.port}")
    print(f"payload_bytes: {payload_size}")
    print(f"payload_mib: {human_mib(payload_size):.4f}")
    if args.payload_bytes is None and args.payload_kb is None:
        print(
            "derived_from: "
            f"batch={args.batch_size}, seq_len={args.seq_len}, "
            f"hidden={args.hidden_size}, hidden_divisor={args.hidden_divisor}, "
            f"bytes_per_element={args.bytes_per_element}"
        )
    print(f"warmup_runs: {args.warmup_runs}")
    print(f"measure_runs: {args.measure_runs}")
    print("=" * 79)

    measurements_ms: list[float] = []
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        if args.tcp_nodelay:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.connect((args.host, args.port))
        for run_idx in range(total_runs):
            t0_ns = time.perf_counter_ns()
            sock.sendall(header)
            sock.sendall(payload)
            recv_exact(sock, args.recv_ack_bytes)
            elapsed_ms = (time.perf_counter_ns() - t0_ns) / 1_000_000.0
            if run_idx >= args.warmup_runs:
                measurements_ms.append(elapsed_ms)
                if args.verbose:
                    measured_idx = run_idx - args.warmup_runs + 1
                    print(f"measure_run={measured_idx:03d} latency_ms={elapsed_ms:.3f}")
        sock.sendall(HEADER_STRUCT.pack(0))

    avg_ms, min_ms, max_ms, std_ms = summarize(measurements_ms)
    print("results (measure runs only):")
    print(f"  avg_ms: {avg_ms:.3f}")
    print(f"  min_ms: {min_ms:.3f}")
    print(f"  max_ms: {max_ms:.3f}")
    print(f"  std_ms: {std_ms:.3f}")


if __name__ == "__main__":
    main()
