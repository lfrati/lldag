import math
from random import random

"""
I want to implement a llvmlite implementation of some neat-like graph.
Pros:
- Fast
Cons:
- ???
"""


class Input:
    def __init__(self, value, label=None):
        self.const = value
        self.label = label
        self.value = None

    def query(self):
        if not self.value:
            self.value = self.const
            print(f"   {self.label} = {self.value}")

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
            print(f"{self.label}: values = sum({values})")
            self.value = self.op(pre)
            print(f"{self.label}: pre = {self.op}({pre})")
            print(f"{self.label}: {self.label} = {self.value}")
        return self.value

    def __repr__(self):
        return f"{self.hash()} : {self.op}({[self.inputs]})\n"


inputs = [Input(1, "X0"), Input(-1, "X1"), Input(2, "X2")]
print(inputs)


class Op:
    def __repr__(self):
        return self.__class__.__name__


class ReLU(Op):
    def __call__(self, x):
        return max(0, x)


class Sin(Op):
    def __call__(self, x):
        return math.sin(x)


A = Node(ReLU(), [inputs[0], inputs[2]], "A")
B = Node(ReLU(), [inputs[1], inputs[2]], "B")
C = Node(ReLU(), [inputs[1], inputs[2]], "C")
D = Node(ReLU(), [A, B, C], "D")
print(A)
print(B)
print(C)
print(D)

print(D.query())

#%%

"""
Ok the pytorch/micrograd style implementation is fun, but I'm going to mutate the structure of
this graph a lot, I don't want to have to go fish for information inside nodes and change it.
Instead I want a single place were to store information about:
- edges
- ops

Take #2:
"""

wiring = {
    "A": ([0, 2], ReLU()),
    "B": ([1, 2], Sin()),
    "C": ([1, "B"], ReLU()),
    "D": (["A", "B", "C"], Sin()),
    "E": (["A", 0, "C"], ReLU()),
}

weights = {
    key: [random() * 2 - 1 for _ in range(len(wiring[key]))] for key in wiring.keys()
}


def query(output, inputs, context=None):
    if not context:
        context = {}
    _query(output, inputs, context)
    return context


def _query(node, inputs, context):
    if type(node) == int and not node in context:
        context[node] = inputs[node]
        print(f"   X{node}={context[node]}")
        return

    # only evaluate a common subtree once
    if not node in context:
        inps, op = wiring[node]
        for inp in inps:
            _query(inp, inputs, context)

        values = [w * context[pred] for w, pred in zip(weights[node], inps)]
        pre = sum(values)
        context[node] = op(pre)
        print(f"{node}: pre = sum({values})")
        print(f"{node}: {node} = {op}(pre) --> {context[node]}")

    return context[node]


inputs = [1, -1, 2]
outputs = ["D", "E"]
context = {}
for out in outputs:
    context = query(out, inputs, context)

print(f"   return [{','.join(outputs)}]")