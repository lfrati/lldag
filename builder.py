import numpy as np
from string import ascii_uppercase
from itertools import product
from typing import Iterator


def make_names(N: int) -> Iterator[str]:
    """
    Generate N unique strings, lexicographically ordered
    """
    l = N // len(ascii_uppercase) + 1
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
) -> tuple[dict[str, list[str]], dict[str, np.ndarray]]:
    np.random.seed(seed)
    pool = [str(i) for i in range(n_inputs)]
    wiring = {}
    for node in make_names(n_nodes):
        n_edges = np.random.randint(min_edges, max_edges)
        edges = np.random.choice(pool, size=n_edges, replace=False).tolist()
        wiring[node] = [int(e) if e.isdigit() else e for e in edges]
        pool.append(node)
    weights = {
        out: np.random.uniform(-1, 1, size=(len(ins))) for out, ins in wiring.items()
    }
    return wiring, weights


def main():
    wiring, weights = make_dag(n_nodes=32, n_inputs=16, min_edges=4, max_edges=5)
    for neuron, edges in wiring.items():
        ws = weights[neuron]
        print(ws.sum())
        assert len(ws) == len(edges)
        weight_info = f"[{ws.min():+.3f}, {ws.max():+.3f}]"
        print(f'"{neuron}": {weight_info} | { [f"{e:>2}" for e in edges]}')


if __name__ == "__main__":
    main()
