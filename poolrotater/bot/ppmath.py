"""osu! performance-point weighting.

A profile's pp is the weighted sum of its top plays: the best score counts
fully, the second at 0.95, the third at 0.95^2, and so on. So a new score is
worth far more than its raw pp suggests when it lands high, and almost
nothing when it lands near #100.

    total = sum(pp_i * 0.95^i) + bonus

The bonus term depends only on total clear count, so it cancels out of a
gain calculation and is ignored here.
"""
from __future__ import annotations

DECAY = 0.95


def weighted_total(pps: list[float]) -> float:
    """Weighted sum of a descending list of top-play pp values."""
    return sum(pp * (DECAY ** i) for i, pp in enumerate(pps))


def insert_score(top: list[tuple[float, int]], pp: float, beatmap_id: int,
                 cap: int = 100) -> tuple[list[tuple[float, int]], int | None]:
    """Insert a new score into a cached top-100 and report its rank.

    `top` is [(pp, beatmap_id), ...] sorted descending.

    Handles the case that matters most in a farm lobby: a new score on a map
    the player ALREADY has in their top 100 replaces the old one rather than
    stacking, and only counts if it is actually better. Getting this wrong
    would overstate gains on repeat plays of the same map, which is exactly
    what a rotating pool produces.

    Returns (new_list, rank) where rank is 1-based, or None if the score
    does not make the top `cap`.
    """
    existing = next((i for i, (_p, b) in enumerate(top) if b == beatmap_id), None)
    if existing is not None:
        if pp <= top[existing][0]:
            return top, None          # not an improvement on this map
        top = top[:existing] + top[existing + 1:]

    rank = None
    for i, (p, _b) in enumerate(top):
        if pp > p:
            rank = i
            break
    if rank is None:
        if len(top) < cap:
            rank = len(top)
        else:
            return top, None          # below their #100

    new = top[:rank] + [(pp, beatmap_id)] + top[rank:]
    return new[:cap], rank + 1


def gain_from(top: list[tuple[float, int]], pp: float,
              beatmap_id: int, cap: int = 100):
    """How much total profile pp a new score adds, and what rank it takes.

    Returns (gain, rank, new_top) -- gain 0.0 and rank None if it doesn't
    make the list.
    """
    before = weighted_total([p for p, _b in top])
    new_top, rank = insert_score(top, pp, beatmap_id, cap)
    if rank is None:
        return 0.0, None, top
    after = weighted_total([p for p, _b in new_top])
    return after - before, rank, new_top
