#!/usr/bin/env python3
"""Descarga un archivo público de Drive mediante rangos HTTP verificables."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

RANGE_NAME = re.compile(r"^(\d+)-(\d+)\.part$")


def fetch_range(
    url: str,
    start: int,
    end: int,
    total: int,
    destination: Path,
    retries: int,
) -> tuple[int, int]:
    expected_bytes = end - start + 1
    if destination.is_file() and destination.stat().st_size == expected_bytes:
        return start, expected_bytes

    temporary = destination.with_name(f"{destination.name}.download")
    temporary.unlink(missing_ok=True)
    expected_range = f"bytes {start}-{end}/{total}"
    range_url = f"{url}&range_start={start}"
    headers = destination.with_name(f"{destination.name}.headers")
    for attempt in range(1, retries + 1):
        try:
            result = subprocess.run(
                [
                    "curl",
                    "-sS",
                    "-L",
                    "--fail",
                    "--max-time",
                    "300",
                    "--range",
                    f"{start}-{end}",
                    "--dump-header",
                    str(headers),
                    "--output",
                    str(temporary),
                    "--write-out",
                    "%{http_code}",
                    range_url,
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=330,
            )
            header_text = headers.read_text(encoding="latin-1") if headers.exists() else ""
            content_ranges = re.findall(r"(?im)^content-range:\s*(.+?)\s*$", header_text)
            content_range = content_ranges[-1] if content_ranges else None
            if result.returncode != 0 or result.stdout != "206" or content_range != expected_range:
                raise RuntimeError(
                    f"respuesta inválida: curl={result.returncode} status={result.stdout!r} "
                    f"content_range={content_range!r} stderr={result.stderr.strip()!r}"
                )
            received = temporary.stat().st_size if temporary.exists() else 0
            if received != expected_bytes:
                raise RuntimeError(
                    f"bloque incompleto: recibido={received} esperado={expected_bytes}"
                )
            temporary.replace(destination)
            headers.unlink(missing_ok=True)
            return start, expected_bytes
        except (RuntimeError, subprocess.TimeoutExpired) as exc:
            temporary.unlink(missing_ok=True)
            headers.unlink(missing_ok=True)
            if attempt == retries:
                raise RuntimeError(f"rango {start}-{end}: {exc}") from exc
            time.sleep(min(5 * attempt, 30))
    raise AssertionError("bucle de reintentos inalcanzable")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("file_id")
    parser.add_argument("output", type=Path)
    parser.add_argument("expected_size", type=int)
    parser.add_argument("--chunk-mib", type=float, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--retries", type=int, default=10)
    args = parser.parse_args()

    if min(args.expected_size, args.chunk_mib, args.workers, args.retries) <= 0:
        parser.error("expected_size, chunk-mib, workers y retries deben ser positivos")

    output = args.output.resolve()
    if output.is_file() and output.stat().st_size == args.expected_size:
        print(f"download_ok output={output} bytes={args.expected_size}")
        return

    prefix = output.with_name(f"{output.name}.part")
    ranges = output.with_name(f"{output.name}.ranges")
    ranges.mkdir(parents=True, exist_ok=True)
    chunk_bytes = int(args.chunk_mib * 1024 * 1024)
    if chunk_bytes <= 0:
        parser.error("chunk-mib produce un bloque vacío")

    prefix_size = prefix.stat().st_size if prefix.exists() else 0
    if prefix_size > args.expected_size:
        raise SystemExit(f"ERROR: parcial mayor al tamaño esperado: {prefix_size}")
    complete_prefix = min((prefix_size // chunk_bytes) * chunk_bytes, args.expected_size)
    if prefix_size != complete_prefix:
        with prefix.open("r+b") as handle:
            handle.truncate(complete_prefix)
        prefix_size = complete_prefix

    url = (
        "https://drive.usercontent.google.com/download"
        f"?id={args.file_id}&export=download&confirm=t"
    )
    for temporary in ranges.glob("*.download"):
        temporary.unlink()

    existing: list[tuple[int, int, Path]] = []
    for path in ranges.glob("*.part"):
        match = RANGE_NAME.fullmatch(path.name)
        if match is None:
            raise SystemExit(f"ERROR: nombre de rango inválido: {path}")
        start, end = map(int, match.groups())
        if start < prefix_size or end < start or end >= args.expected_size:
            raise SystemExit(f"ERROR: límites de rango inválidos: {path}")
        if path.stat().st_size != end - start + 1:
            raise SystemExit(f"ERROR: tamaño de rango inválido: {path}")
        existing.append((start, end, path))
    existing.sort()

    specifications: list[tuple[int, int, Path]] = []
    current = prefix_size
    for existing_start, existing_end, existing_path in existing:
        if existing_start < current:
            raise SystemExit(f"ERROR: rangos solapados en {existing_path}")
        while current < existing_start:
            end = min(current + chunk_bytes - 1, existing_start - 1)
            path = ranges / f"{current:012d}-{end:012d}.part"
            specifications.append((current, end, path))
            current = end + 1
        specifications.append((existing_start, existing_end, existing_path))
        current = existing_end + 1
    while current < args.expected_size:
        end = min(current + chunk_bytes - 1, args.expected_size - 1)
        path = ranges / f"{current:012d}-{end:012d}.part"
        specifications.append((current, end, path))
        current = end + 1

    completed = prefix_size + sum(
        path.stat().st_size for _, _, path in specifications if path.is_file()
    )
    pending = [(start, end, path) for start, end, path in specifications if not path.is_file()]
    print(
        f"download_start bytes={completed}/{args.expected_size} workers={args.workers} "
        f"pending_ranges={len(pending)}",
        flush=True,
    )
    report_bytes = max(64 * 1024 * 1024, args.expected_size // 1000)
    next_report = completed + report_bytes

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                fetch_range,
                url,
                start,
                end,
                args.expected_size,
                destination,
                args.retries,
            ): (start, end)
            for start, end, destination in pending
        }
        try:
            for future in as_completed(futures):
                _, received = future.result()
                completed += received
                if completed >= next_report or completed == args.expected_size:
                    percent = completed * 100 / args.expected_size
                    print(
                        f"download_progress bytes={completed}/{args.expected_size} "
                        f"percent={percent:.2f}",
                        flush=True,
                    )
                    next_report = completed + report_bytes
        except Exception:
            for future in futures:
                future.cancel()
            raise

    assembled = output.with_name(f"{output.name}.assembled")
    with assembled.open("wb") as target:
        if prefix_size:
            with prefix.open("rb") as source:
                shutil.copyfileobj(source, target, length=1024 * 1024)
        for start, end, part in specifications:
            expected_bytes = end - start + 1
            if not part.is_file() or part.stat().st_size != expected_bytes:
                raise SystemExit(f"ERROR: rango faltante o inválido: {part}")
            with part.open("rb") as source:
                shutil.copyfileobj(source, target, length=1024 * 1024)
        target.flush()
        os.fsync(target.fileno())

    if assembled.stat().st_size != args.expected_size:
        raise SystemExit("ERROR: tamaño ensamblado inválido")
    assembled.replace(output)
    prefix.unlink(missing_ok=True)
    for _, _, part in specifications:
        part.unlink()
    ranges.rmdir()
    print(f"download_ok output={output} bytes={args.expected_size}")


if __name__ == "__main__":
    main()
