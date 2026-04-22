from __future__ import annotations

import re
import zlib


def scene_track_id_to_int(track_id: str) -> int:
    """Canonical integer track identifier derived from SceneActor.track_id."""
    match = re.search(r"(\d+)$", track_id)
    if match is not None:
        return int(match.group(1))
    return int(zlib.crc32(track_id.encode("utf-8")) & 0x7FFFFFFF)
