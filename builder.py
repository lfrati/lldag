from itertools import product
import math
from string import ascii_uppercase
from typing import Iterator

import numpy as np


def make_names(N: int) -> Iterator[str]:
    """
    Generate N unique strings, lexicographically ordered
    """
    l = math.ceil(math.log(N, len(ascii_uppercase)))

    for i, line in enumerate(product(ascii_uppercase, repeat=l)):
        if i >= N:
            return
        yield "".join(line)


def make_dag(
    n_nodes: int = 32,
    n_inputs: int = 8,
    min_edges: int = 3,
    max_edges: int = 6,
    seed: int = 1337,
) -> dict[str, list[str]]:
    np.random.seed(seed)
    pool = [str(i) for i in range(n_inputs)]
    wiring = {}
    for node in make_names(n_nodes):
        n_edges = np.random.randint(min_edges, max_edges)
        edges = np.random.choice(pool, size=n_edges, replace=False).tolist()
        wiring[node] = [int(e) if e.isdigit() else e for e in edges]
        pool.append(node)
    return wiring


def show(wiring, weights):
    for neuron, edges in wiring.items():
        ws = weights[neuron]
        assert len(ws) == len(edges)
        weight_info = f"[{ws.min():+.3f}, {ws.max():+.3f}]"
        print(f'"{neuron}": {weight_info} | { [f"{e:>2}" for e in edges]}')


def main():
    wiring, weights = make_dag(n_nodes=32, n_inputs=16, min_edges=4, max_edges=5)
    show(wiring, weights)


if __name__ == "__main__":
    main()
