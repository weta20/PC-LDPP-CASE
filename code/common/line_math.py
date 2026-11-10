import numpy as np


def base_turn_block(points, start, stop):
    """Minimum signed heading changes over the two rectilinear L connectors.

    A reversal counts as one change. Exact coordinate comparisons determine
    whether a connector leg has zero length.
    """
    p = np.asarray(points, dtype=np.float64)
    n = len(p)
    ids = np.arange(n)
    tangent = np.sign(p - p[ids ^ 1])
    outgoing = np.where(
        tangent[:, 0] > 0, 0, np.where(tangent[:, 1] > 0, 1, np.where(tangent[:, 0] < 0, 2, 3))
    )
    incoming = (outgoing + 2) % 4
    dx = p[None, :, 0] - p[start:stop, None, 0]
    dy = p[None, :, 1] - p[start:stop, None, 1]
    h = np.where(dx > 0, 0, 2)
    v = np.where(dy > 0, 1, 3)
    current = outgoing[start:stop, None]
    result = np.minimum(
        (current != h).astype(np.uint8) + 1 + (v != incoming),
        (current != v).astype(np.uint8) + 1 + (h != incoming),
    )
    result = np.where(dx == 0, (current != v).astype(np.uint8) + (v != incoming), result)
    result = np.where(dy == 0, (current != h).astype(np.uint8) + (h != incoming), result)
    result = np.where((dx == 0) & (dy == 0), (current != incoming).astype(np.uint8), result)
    local = np.arange(stop - start)
    result[local, np.arange(start, stop)] = 0
    result[local, np.arange(start, stop) ^ 1] = 0
    return result.astype(np.uint8)
