#!/usr/bin/env python3
"""TCP receiver for hidden_states payload latency benchmarks."""

from __future__ import annotations

import argparse
import socket
import struct
from typing import Tuple


ACK_BYTES = b"OK"
HEADER_STRUCT = struct.Struct("!Q")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Receive fixed-size payloads over TCP and ack each one."
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind host. Default: 0.0.0.0")
    parser.add_argument("--port", type=int, default=5001, help="Bind port. Default: 5001")
    parser.add_argument(
        "--listen-backlog",
        type=int,
        default=1,
        help="listen() backlog. Default: 1",
    )
    parser.add_argument(
        "--recv-chunk-bytes",
        type=int,
        default=1024 * 1024,
        help="Chunk size for socket.recv while draining a payload. Default: 1MiB",
    )
    parser.add_argument(
        "--tcp-nodelay",
        action="store_true",
        help="Enable TCP_NODELAY on accepted client sockets.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print one line per payload received.",
    )
    return parser.parse_args()


def recv_exact(sock: socket.socket, num_bytes: int, chunk_bytes: int) -> bytes:
    parts = bytearray()
    while len(parts) < num_bytes:
        to_read = min(chunk_bytes, num_bytes - len(parts))
        chunk = sock.recv(to_read)
        if not chunk:
            raise ConnectionError("Peer closed connection while receiving payload.")
        parts.extend(chunk)
    return bytes(parts)


def recv_header(sock: socket.socket, chunk_bytes: int) -> int:
    header = recv_exact(sock, HEADER_STRUCT.size, chunk_bytes)
    return HEADER_STRUCT.unpack(header)[0]


def handle_client(
    conn: socket.socket,
    peer: Tuple[str, int],
    recv_chunk_bytes: int,
    verbose: bool,
) -> None:
    print(f"[receiver] client connected: {peer[0]}:{peer[1]}")
    message_count = 0
    total_bytes = 0
    try:
        while True:
            try:
                payload_size = recv_header(conn, recv_chunk_bytes)
            except ConnectionError:
                break
            if payload_size == 0:
                break
            recv_exact(conn, payload_size, recv_chunk_bytes)
            conn.sendall(ACK_BYTES)
            message_count += 1
            total_bytes += payload_size
            if verbose:
                print(
                    f"[receiver] message={message_count:04d} "
                    f"payload_bytes={payload_size}"
                )
    finally:
        print(
            f"[receiver] client disconnected: {peer[0]}:{peer[1]} "
            f"messages={message_count} total_bytes={total_bytes}"
        )


def main() -> None:
    args = parse_args()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((args.host, args.port))
        server.listen(args.listen_backlog)
        print(
            f"[receiver] listening on {args.host}:{args.port} "
            f"(recv_chunk_bytes={args.recv_chunk_bytes})"
        )
        while True:
            conn, peer = server.accept()
            with conn:
                if args.tcp_nodelay:
                    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                handle_client(conn, peer, args.recv_chunk_bytes, args.verbose)


if __name__ == "__main__":
    main()
