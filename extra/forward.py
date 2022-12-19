import math
from random import random
from time import monotonic


class Input:
    def __init__(self, value, label=None):
        self.const = value
        self.label = label
        self.value = None

    def query(self):
        if not self.value:
            self.value = self.const

    def __repr__(self):
        return f"Input : {self.value}"


class Node:
    def __init__(self, op, inputs, label=None):
        self.op = op
        self.inputs = inputs
        self.value = None
        self.weights = [random() * 2 - 1 for _ in range(len(inputs))]
        self.label = label

    def hash(self):
        return self.label if self.label else self.__hash__()

    def query(self):
        if not self.value:
            for input in self.inputs:
                input.query()

            values = [w * input.value for w, input in zip(self.weights, self.inputs)]
            pre = sum(values)
            self.value = self.op(pre)
        return self.value

    def __repr__(self):
        return f"{self.hash()} : {self.op}({[self.inputs]})\n"


class Op:
    def __repr__(self):
        return self.__class__.__name__


class ReLU(Op):
    def __call__(self, x):
        return max(0, x)


class Sin(Op):
    def __call__(self, x):
        return math.sin(x)


# A = Node(ReLU(), [inputs[0], inputs[2]], "A")
# B = Node(ReLU(), [inputs[1], inputs[2]], "B")
# C = Node(ReLU(), [inputs[1], inputs[2]], "C")
# D = Node(ReLU(), [A, B, C], "D")
# print(A)
# print(B)
# print(C)
# print(D)
# print(D.query())

#%%

"""
Ok the pytorch/micrograd style implementation is fun, but I'm going to mutate the structure of
this graph a lot, I don't want to have to go fish for information inside nodes and change it.
Instead I want a single place were to store information about:
- edges
- ops

Take #2:
"""

inputs = [random() * 2 - 1 for _ in range(16)]
outputs = ["Y", "Z"]
wiring = {
    "A": [14, 3, 15, 12, 13, 9, 8],
    "B": [13, 2, 10, 7, 9, "A"],
    "C": [8, 5, 0, 14, 3, 15, 10, 11],
    "D": [11, 3, 15, 1, "A", "C"],
    "E": ["B", 15, 13, 14, 10, 6, 11],
    "F": [8, 11, 1, 5, 14, "E", "B", 4],
    "G": [0, 7, 9, 8, "A", 10, "E", 4, 12],
    "H": ["A", 10, 0, 11, "E", 5, "B", "F", 8],
    "I": ["D", 7, "C", 2, "F", 3, 11],
    "J": ["D", 15, 14, 11, 7, 12, "B", "F", 3],
    "K": ["B", 8, 9, 3, 4, 10, "G", 15],
    "L": ["I", 3, "K", "H", 1, 10, 8, 14, 5],
    "M": [2, 0, "D", 1, 5, "C", 4],
    "N": [13, "J", 10, 15, "F", "A", "G"],
    "O": ["D", 5, "F", "K", 1, 0, 15],
    "P": [7, 5, 10, "M", "B", 0, "D", "G"],
    "Q": ["L", "J", 4, "K", "N", "H", "G"],
    "R": [11, 0, "Q", "F", 4, 3],
    "S": ["G", "C", "F", 1, "B", 8, 4, 5, 10],
    "T": ["R", 13, 1, 4, "J", 2, "L"],
    "U": [1, "R", 7, "C", "O", "F", "N", 0, "L"],
    "V": ["G", 7, "S", 4, 6, 12, "O", "Q", "L"],
    "W": ["P", 11, 7, "F", 14, 0, "L", "C", "I"],
    "X": [5, "M", "H", "B", 10, "F", "R", "W", "N"],
    "Y": ["R", "D", "T", "Q", 5, 3, "U", "F"],
    "Z": ["R", "K", "A", 5, "S", 1, 12, 10, "T"],
}
ops = {neuron: ReLU() for neuron in wiring.keys()}
weights = {
    key: [random() * 2 - 1 for _ in range(len(wiring[key]))] for key in wiring.keys()
}
outputs = ["Y", "Z"]


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


CNT = 1_000
start = monotonic()
for _ in range(CNT):
    _ = run(inputs, outputs, wiring, weights)
end = monotonic()
print(f"{(end - start)/CNT:.9f}")
