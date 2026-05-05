from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import numpy as np


def write_viewer_manifest(
    cache_root: str | Path,
    title: str,
    items: list[tuple[str, np.ndarray]],
    start_index: int = 0,
) -> Path:
    cache_root = Path(cache_root)
    cache_dir = cache_root / datetime.now().strftime("%Y%m%d_%H%M%S") / uuid4().hex[:8]
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "title": title,
        "start_index": int(max(0, min(start_index, max(len(items) - 1, 0)))),
        "items": [],
    }
    colors = [
        [0.18, 0.80, 0.44],
        [0.95, 0.65, 0.25],
        [0.37, 0.69, 1.00],
        [0.91, 0.34, 0.34],
        [0.77, 0.58, 0.96],
        [0.95, 0.86, 0.34],
    ]
    for index, (name, mask) in enumerate(items):
        array_path = cache_dir / f"mask_{index:03d}.npy"
        np.save(array_path, np.asarray(mask, dtype=np.uint8))
        manifest["items"].append(
            {
                "name": str(name),
                "array_path": str(array_path),
                "color": colors[index % len(colors)],
                "opacity": 1.0,
            }
        )
    manifest_path = cache_dir / "viewer_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path
