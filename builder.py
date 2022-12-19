import numpy as np
from string import ascii_uppercase


def make_dag(n_inputs=8, min_edges=3, max_edges=6, seed=1337):
    np.random.seed(seed)
    pool = [str(i) for i in range(n_inputs)]
    wiring = {}
    for node in ascii_uppercase:
        edges = np.random.randint(min_edges, max_edges)
        wiring[node] = np.random.choice(pool, size=(edges), replace=False).tolist()
        pool.append(node)

    return wiring


def main():
    wiring = make_dag(n_inputs=16, min_edges=6, max_edges=10)
    for neuron, edges in wiring.items():
        print(f'"{neuron}": { [int(e) if e.isdigit() else e for e in edges]},')


if __name__ == "__main__":
    main()
