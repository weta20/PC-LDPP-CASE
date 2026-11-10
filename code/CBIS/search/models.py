"""Paired endpoint states for a closed scan-line route."""

import math

import numpy as np

from ..line_search import orient


class Deadline(Exception):
    """Signal that search must return the best route retained so far."""

    def __init__(self, route=None):
        super().__init__("Search time limit reached")
        self.route = route


class LineModel:
    """Each line has two entry states; its exit is the opposite endpoint."""

    def __init__(self, matrix):
        self.c = matrix
        self.groups = [
            np.asarray(group, np.int32)
            for group in np.arange(len(matrix), dtype=np.int32).reshape(-1, 2)
        ]
        self.m = len(self.groups)
        self.owner = np.arange(len(matrix), dtype=np.int32) // 2
        if not self.m or not np.allclose(matrix, matrix.T, atol=1e-10, rtol=0):
            raise ValueError("Expected a nonempty symmetric endpoint connection matrix")

    def exit(self, state):
        return state ^ 1

    def edge(self, left, right):
        return self.c[self.exit(left), right]

    def value(self, route):
        return math.fsum(map(float, self.edge(route, np.roll(route, -1))))

    def flip(self, states):
        return states ^ 1

    def reoptimize(self, route, deadline=float("inf")):
        return orient(self.owner[route], self.c)

    def audit(self, route):
        if len(route) != self.m or not np.array_equal(
            np.sort(self.owner[route]), np.arange(self.m)
        ):
            raise ValueError("A scan line was duplicated or omitted")
        return self.value(route)

    def changed_delta(self, before, after, positions):
        positions = np.asarray(positions, dtype=np.int32) % self.m
        edges = np.unique(np.r_[positions, (positions - 1) % self.m])
        following = (edges + 1) % self.m
        return float(
            np.sum(
                self.edge(after[edges], after[following])
                - self.edge(before[edges], before[following])
            )
        )

    def state_window(self, route, start, length=1):
        """Optimize one or two entry directions while fixing the outside route."""
        length = min(length, self.m)
        if length == self.m:
            return self.reoptimize(route)
        positions = (start + np.arange(length)) % self.m
        choices = [self.groups[self.owner[route[i]]] for i in positions]
        left = route[(start - 1) % self.m]
        right = route[(start + length) % self.m]
        previous = self.edge(left, choices[0])
        backs = []
        for before, nodes in zip(choices[:-1], choices[1:]):
            cost = previous[:, None] + self.edge(before[:, None], nodes[None, :])
            back = np.argmin(cost, axis=0)
            previous = cost[back, np.arange(len(nodes))]
            backs.append(back)
        index = int(np.argmin(previous + self.edge(choices[-1], right)))
        result = route.copy()
        for k in range(length - 1, -1, -1):
            result[positions[k]] = choices[k][index]
            if k:
                index = int(backs[k - 1][index])
        return result

    def repair(self, route, positions):
        """Choose the cheaper entry direction at each affected position."""
        result = route.copy()
        for pos in dict.fromkeys(int(i) % self.m for i in positions):
            nodes = self.groups[self.owner[result[pos]]]
            if self.m == 1:
                cost = self.edge(nodes, nodes)
            else:
                cost = self.edge(result[pos - 1], nodes) + self.edge(
                    nodes, result[(pos + 1) % self.m]
                )
            result[pos] = nodes[int(np.argmin(cost))]
        return result

    def neighbor_units(self, count):
        """Rank distinct lines by connection cost from each endpoint state."""
        neighbors = []
        count = min(count, self.m - 1)
        for state in range(len(self.c)):
            picked = []
            seen = {int(self.owner[state])}
            for target in np.argsort(self.c[self.exit(state)], kind="stable"):
                group = int(self.owner[target])
                if group not in seen:
                    picked.append(group)
                    seen.add(group)
                    if len(picked) == count:
                        break
            neighbors.append(np.asarray(picked, np.int32))
        return neighbors


def line_model(matrix):
    return LineModel(matrix)
