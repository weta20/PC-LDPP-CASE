"""Unchanged local heading and weighted Manhattan connection helpers."""


def heading(a, b):
    dx, dy = b - a
    return 1 if dx > 0 else -1 if dx < 0 else 2 if dy > 0 else -2


def connection(a, b, before, after, turn_weight=1.0):
    dx, dy = b - a
    h = 1 if dx > 0 else -1
    v = 2 if dy > 0 else -2
    candidates = []
    for axes in [(0, 1), (1, 0)]:
        directions = (
            [before]
            + [(h if axis == 0 else v) for axis in axes if (dx if axis == 0 else dy) != 0]
            + [after]
        )
        candidates.append(sum(x != y for x, y in zip(directions, directions[1:])))
    return abs(dx) + abs(dy) + turn_weight * min(candidates)
