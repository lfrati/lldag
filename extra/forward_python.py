from time import monotonic


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

    np.random.seed(4)

    inputs = np.random.rand(4).astype(np.float32)
    outputs = ["B", "C"]
    wiring = {"A": [0, 1, 2], "B": [2, 3, "A"], "C": [0, "A", "B"]}

    # inputs = np.random.rand(16).astype(np.float32).tolist()
    # outputs = ["Y", "Z"]
    # wiring = {
    #     "A": [14, 3, 15, 12, 13, 9, 8],
    #     "B": [13, 2, 10, 7, 9, "A"],
    #     "C": [8, 5, 0, 14, 3, 15, 10, 11],
    #     "D": [11, 3, 15, 1, "A", "C"],
    #     "E": ["B", 15, 13, 14, 10, 6, 11],
    #     "F": [8, 11, 1, 5, 14, "E", "B", 4],
    #     "G": [0, 7, 9, 8, "A", 10, "E", 4, 12],
    #     "H": ["A", 10, 0, 11, "E", 5, "B", "F", 8],
    #     "I": ["D", 7, "C", 2, "F", 3, 11],
    #     "J": ["D", 15, 14, 11, 7, 12, "B", "F", 3],
    #     "K": ["B", 8, 9, 3, 4, 10, "G", 15],
    #     "L": ["I", 3, "K", "H", 1, 10, 8, 14, 5],
    #     "M": [2, 0, "D", 1, 5, "C", 4],
    #     "N": [13, "J", 10, 15, "F", "A", "G"],
    #     "O": ["D", 5, "F", "K", 1, 0, 15],
    #     "P": [7, 5, 10, "M", "B", 0, "D", "G"],
    #     "Q": ["L", "J", 4, "K", "N", "H", "G"],
    #     "R": [11, 0, "Q", "F", 4, 3],
    #     "S": ["G", "C", "F", 1, "B", 8, 4, 5, 10],
    #     "T": ["R", 13, 1, 4, "J", 2, "L"],
    #     "U": [1, "R", 7, "C", "O", "F", "N", 0, "L"],
    #     "V": ["G", 7, "S", 4, 6, 12, "O", "Q", "L"],
    #     "W": ["P", 11, 7, "F", 14, 0, "L", "C", "I"],
    #     "X": [5, "M", "H", "B", 10, "F", "R", "W", "N"],
    #     "Y": ["R", "D", "T", "Q", 5, 3, "U", "F"],
    #     "Z": ["R", "K", "A", 5, "S", 1, 12, 10, "T"],
    # }

    ops = {neuron: relu for neuron in wiring.keys()}
    weights = {
        key: np.random.rand(len(edges)).astype(dtype=np.float32).tolist()
        for key, edges in wiring.items()
    }

    CNT = 1_000
    start = monotonic()
    for _ in range(CNT):
        retvals = run(inputs, outputs, wiring, weights)
    end = monotonic()
    print(f"{(end - start)/CNT:.9f}")
    print(retvals)
