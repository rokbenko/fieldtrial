"""UUIDv7 string primary keys (RFC 9562): time-ordered, so rows sort by creation."""

import os
import time
import uuid


def uuid7() -> str:
    """A new UUIDv7 as a string: 48-bit Unix milliseconds, version 7, then random bits."""
    millis = time.time_ns() // 1_000_000
    rand = int.from_bytes(os.urandom(10), "big")
    value = (millis & ((1 << 48) - 1)) << 80
    value |= 0x7 << 76  # version
    value |= ((rand >> 68) & 0xFFF) << 64  # 12 random bits (rand_a)
    value |= 0b10 << 62  # RFC 4122 variant
    value |= rand & ((1 << 62) - 1)  # 62 random bits (rand_b)
    return str(uuid.UUID(int=value))
