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
) -> dict[str, list[str]]:
    np.random.seed(seed)
    pool = [str(i) for i in range(n_inputs)]
    wiring = {}
    for node in make_names(n_nodes):
        edges = np.random.randint(min_edges, max_edges)
        wiring[node] = np.random.choice(pool, size=(edges), replace=False).tolist()
        pool.append(node)

    return wiring


def main():
    wiring = make_dag(n_nodes=32, n_inputs=16, min_edges=4, max_edges=5)
    print("{")
    for neuron, edges in wiring.items():
        print(f'"{neuron}": { [int(e) if e.isdigit() else e for e in edges]},')
    print("}")


if __name__ == "__main__":
    main()
