"""Shared metadata helpers for tiny_*.py fixture generators (fiche T-1.3)."""
import datetime
import json
import struct
import subprocess
from pathlib import Path

from safetensors.torch import save_file

_YUE_ROOT = Path(__file__).resolve().parents[2] / "reference" / "yue"


def upstream_sha():
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=_YUE_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


META = {"upstream_sha": upstream_sha(), "generated": datetime.date.today().isoformat()}


def save_file_deterministic(tensors, path, metadata):
    """`safetensors.torch.save_file`, then canonicalize the header's JSON key order.

    The Rust HashMap backing the header (tensor entries + ``__metadata__``) iterates in
    a per-process randomized order, so two back-to-back runs write byte-different files
    even though every tensor and every metadata value is identical (verified by content,
    not just size). Re-serializing the already-written header with sorted keys makes the
    file reproducible without touching the data section (``data_offsets`` are relative
    offsets into that section, unaffected by JSON key order).
    """
    save_file(tensors, path, metadata=metadata)
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
        data = f.read()
    canonical = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(canonical)))
        f.write(canonical)
        f.write(data)
