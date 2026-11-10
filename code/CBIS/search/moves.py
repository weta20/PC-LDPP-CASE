"""Constant-boundary move evaluation; scan lines flip direction on reversal."""

import numpy as np


def swap_delta(e, i, j, model):
    """Score a two-line swap from the affected connector edges."""
    m = len(e)
    affected = sorted({(i - 1) % m, i, (j - 1) % m, j})

    def new(k):
        return e[j] if k == i else e[i] if k == j else e[k]

    return float(
        sum(
            model.edge(new(k), new((k + 1) % m)) - model.edge(e[k], e[(k + 1) % m])
            for k in affected
        )
    )


def reverse_delta(e, i, j, model):
    """Score a segment reversal, reversing each scan direction as well."""
    m = len(e)
    if i == 0 and j == m - 1:
        return 0.0
    return float(
        model.edge(e[i - 1], model.flip(e[j]))
        + model.edge(model.flip(e[i]), e[(j + 1) % m])
        - model.edge(e[i - 1], e[i])
        - model.edge(e[j], e[(j + 1) % m])
    )


def relocate_delta(e, i, length, j, flip, model):
    """Score moving a line or block after position j; reject overlapping moves."""
    m = len(e)
    end = i + length
    if end > m or i <= j < end or j == (i - 1) % m:
        return None
    a, b, z, d, u, v = e[i - 1], e[i], e[end - 1], e[end % m], e[j], e[(j + 1) % m]
    entry, last = (model.flip(z), model.flip(b)) if flip else (b, z)
    return float(
        model.edge(a, d)
        - model.edge(a, b)
        - model.edge(z, d)
        + model.edge(u, entry)
        + model.edge(last, v)
        - model.edge(u, v)
    )


def apply(e, descriptor, model):
    """Apply the order and direction change encoded by a move descriptor."""
    kind, i, j, length, flip = descriptor
    result = e.copy()
    if kind == "swap":
        result[i], result[j] = result[j], result[i]
    elif kind == "reverse":
        result[i : j + 1] = model.flip(result[i : j + 1][::-1])
    else:
        block = e[i : i + length]
        if flip:
            block = model.flip(block[::-1])
        rest = np.r_[e[:i], e[i + length :]]
        pos = j + 1 - (length if j >= i + length else 0)
        result = np.r_[rest[:pos], block, rest[pos:]].astype(np.int32)
    return result


def ni(e, i, model):
    """Remove one unit, select the minimum full insertion delta, keep its state."""
    if len(e) < 3:
        return e.copy()
    target = e[i]
    rest = np.delete(e, i)
    nxt = np.roll(rest, -1)
    increase = model.edge(rest, target) + model.edge(target, nxt) - model.edge(rest, nxt)
    j = int(np.argmin(increase))
    return np.insert(rest, j + 1, target).astype(np.int32)


def shake(e, rng, model, count, enabled, stats):
    """Randomly choose direction change, swap or best insertion at each step."""
    current = e.copy()
    for _ in range(count):
        kind = str(rng.choice(enabled))
        stats["shake_attempts"][kind] += 1
        before = current.copy()
        i = int(rng.integers(len(e)))
        if kind == "change":
            options = model.groups[model.owner[current[i]]]
            options = options[options != current[i]]
            if len(options):
                current[i] = int(rng.choice(options))
        elif kind == "swap":
            if len(e) > 1:
                j = int(rng.choice(np.flatnonzero(np.arange(len(e)) != i)))
                current[i], current[j] = current[j], current[i]
        elif kind == "NI":
            current = ni(current, i, model)
        else:
            raise ValueError("Unknown shake operator: " + kind)
        stats["shake_changed"][kind] += int(not np.array_equal(before, current))
    return current
