"""Blind codes: short, unambiguous labels that hide arm identities from the operator."""

from fieldtrial.design._rng import STREAM_BLINDING, StableRng

# No letters or digits that are easy to confuse (B/8, G/6, I/1/L, O/0/Q, S/5, Z/2).
_LETTERS = "ACDEFHJKMNPRTUVWXY"
_DIGITS = "3479"


def blind_codes(arm_ids: list[str], seed: int) -> dict[str, str]:
    """Assign each arm a distinct two-character code such as ``K7``, derived from the seed.

    The same arms and seed always give the same codes. Codes reveal nothing about the order
    in which the arms are listed.
    """
    pool = [letter + digit for letter in _LETTERS for digit in _DIGITS]
    if len(arm_ids) > len(pool):
        raise ValueError(f"at most {len(pool)} arms can be blinded")
    rng = StableRng(seed, STREAM_BLINDING)
    rng.shuffle(pool)
    return dict(zip(arm_ids, pool, strict=False))
