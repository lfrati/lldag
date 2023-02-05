from time import monotonic
import os

SEED = int(os.getenv("SEED", "4"))


def relu(x):
    return max(0, x)


def run(inputs, outputs, wiring, weights):
    context = {}

    def query(output, inputs, context=None):
        if not context:
            context = {}
        _query(output, inputs, context)
        return context

    def _query(node, inputs, context):
        if type(node) == int and not node in context:
            context[node] = inputs[node]
            return

        # only evaluate a common subtree once
        if not node in context:
            inps = wiring[node]
            op = ops[node]
            for inp in inps:
                _query(inp, inputs, context)

            values = [w * context[pred] for w, pred in zip(weights[node], inps)]
            pre = sum(values)
            context[node] = op(pre)

        return context[node]

    for out in outputs:
        context = query(out, inputs, context)

    return [context[out] for out in outputs]


#%%

if __name__ == "__main__":
    import numpy as np
    from builder import make_dag

    NNODES = 16
    NINPS = 8
    NOUTS = 4
    MINED = 4
    MAXED = 5

    generator = np.random.default_rng(SEED)
    inputs = generator.random(NINPS).astype(np.float32)
    wiring, weights = make_dag(
        n_nodes=NNODES,
        n_inputs=len(inputs),
        min_edges=MINED,
        max_edges=MAXED,
        seed=SEED,
    )
    outputs = list(wiring.keys())[-NOUTS:]

    ops = {neuron: relu for neuron in wiring.keys()}

    CNT = 1_000
    start = monotonic()
    for _ in range(CNT):
        retvals = run(inputs, outputs, wiring, weights)
    end = monotonic()
    print(f"{(end - start)/CNT:.9f}")
    print(retvals)
