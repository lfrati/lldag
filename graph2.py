import math
import numpy as np
from random import random
from rich import print

VALS = -1  # Values counter
ACTS = -1  # Values counter
NEUS = -1  # Neurons counter


class Op:
    def __repr__(self):
        return self.__class__.__name__


class ReLU(Op):
    def __call__(self, pre):
        global ACTS
        ACTS += 1
        label = f"a{ACTS}"
        print(f"[red]# {self.__repr__()}")
        print(
            f"   cmp = builder.fcmp_ordered('<=', ir.Constant(ir.FloatType(), 0), {pre}, flags=('fast',))"
        )
        print(f"   {label} = builder.select(cmp, x, ir.Constant(ir.FloatType(), 0))")
        print("---")
        return Value(label)


class Sin(Op):
    def __call__(self, x):
        return math.sin(x.data)


class Nop(Op):
    def __call__(self, x):
        return x


class Value:
    def __init__(self, label):
        self.label = label

    def query(self):
        return self.label

    def __repr__(self):
        return self.label

    def __mul__(self, other):
        global VALS
        VALS += 1
        label = f"v{VALS}"
        print(f"   {label} = builder.fmul({self.label},{other.label})")
        return Value(label)

    def __add__(self, other):
        global VALS
        VALS += 1
        label = f"v{VALS}"
        print(f"   {label} = builder.fadd({self.label},{other.label})")
        return Value(label)

    def __radd__(self, other):
        return self.__add__(other)

    def __rmul__(self, other):
        return self.__mul__(other)


class Const(Value):
    def __init__(self, c):
        self.label = f"ir.Constant(ir.FloatType(), {c})"


class Input(Value):
    def __init__(self, off=0):
        self.off = off
        self.label = f"inp_{off}"  # for builder
        self.name = f"I{off}"  # for human

    def query(self):
        print(
            f"   {self.label} = builder.load(builder.gep(inp_ptr, ir.Constant(ir.IntType(32),{self.off}))"
        )
        return self.label


class Weight(Value):
    def __init__(self, ix=0, off=0):
        self.ix = ix
        self.off = off
        self.label = f"w{ix}_{off}"  # for builder
        self.label = f"w{ix}_{off}"  # for human

    def query(self):
        print(
            f"   {self.label} = builder.load(builder.gep(w{self.ix}_ptr, ir.Constant(ir.IntType(32),{self.off}))"
        )
        return self.label


NEUS = -1


class Neuron(Value):
    def __init__(self, op=None, name=None):
        global NEUS
        NEUS += 1
        self.ix = NEUS
        if op:
            self.op = op
        else:
            self.op = Nop()
        self.inputs = []
        self.weights = []
        self.label = None
        self.name = name

    def query(self):
        if not self.label:
            print(f"[green]# Neuron {self.name} START")
            pre = Const(0)
            for _, (inp, w) in enumerate(zip(self.inputs, self.weights)):
                inp.query()
                w.query()
                pre = pre + (inp * w)
            self.label = self.op(pre)
            print(f"[green]# Neuron {self.name} END")
        else:
            print(f"[green]# Neuron {self.name} CACHED")
        return self.label

    def __repr__(self):
        return f"{self.name} : {self.op}({[el.name for el in self.inputs]})"


# print(D.query())

ops = {
    "A": ReLU(),
    "B": ReLU(),
    "C": ReLU(),
    "D": ReLU(),
    "E": ReLU(),
}
# outputs = ["D", "E"]
# wiring = {
#     "A": [0, 2],
#     "B": [1, 2],
#     "C": [1, "B"],
#     "D": ["A", "B", "C"],
#     "E": ["A", 0, "C"],
# }

inputs = np.array([1, -1, 2])
outputs = ["B", "C"]
wiring = {
    "A": [0, 1],
    "B": ["A", 2],
    "C": [1, "B"],
}
weights = {
    node: [random() * 2 - 1 for _ in range(len(wiring[node]))] for node in wiring.keys()
}
print(wiring)
print(weights)


def build(wiring, ops):
    neurons = {neuron: Neuron(name=neuron) for neuron in wiring.keys()}
    for name, neuron in neurons.items():
        neuron.op = ops[name] if name in ops else Nop()
        for i, edge in enumerate(wiring[name]):
            if type(edge) == int:
                neuron.inputs.append(Input(off=edge))
            else:
                neuron.inputs.append(neurons[edge])
            neuron.weights.append(Weight(ix=neuron.ix, off=i))
    return neurons


neurons = build(wiring, ops)

for neuron in neurons.values():
    # for x in node.inputs:
    print(neuron)
print("------------")

retvals = []
for output in outputs:
    if output in neurons:
        retvals.append(neurons[output].query())
print("TO RETURN")
print(retvals)
