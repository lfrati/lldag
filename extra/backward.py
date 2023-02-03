import os
from time import monotonic
from typing import Union

import numpy as np
from rich import print as pprint

from draw import draw_micro

DEBUG = int(os.getenv("DEBUG", "0"))
SEED = int(os.getenv("SEED", "4"))


class Value:
    """stores a single scalar value and its gradient"""

    __slots__ = ("data", "grad", "_backward", "_prev", "_op", "name")

    def __init__(self, data, _children=(), _op="", name="V"):
        self.data = data
        self.grad = 0
        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op  # the op that produced this node, for graphviz / debugging / etc
        self.name = f"{name}({data if data is not None else 0:+.4f})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")
        if DEBUG:
            pprint(f"{self.data:+.3f} + {other.data:+.3f}")

        def _backward():
            if DEBUG:
                pprint("[red]ADD")
                pprint(f"  [red]{self.name}.grad = {self.grad:.3f} + {out.grad:.3f}")
                pprint(f"  [red]{other.name}.grad = {other.grad:.3f} + {out.grad:.3f}")
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")
        if DEBUG:
            pprint(f"{self.data:+.3f} x {other.data:+.3f}")

        def _backward():
            if DEBUG:
                pprint("[green]MUL")
                pprint(
                    f"  [green]{self.name}.grad = {self.grad:.3f} + {other.data:.3f} * {out.grad:.3f}"
                )
                pprint(
                    f"  [green]{other.name}.grad = {other.grad:.3f} + {self.data:.3f} * {out.grad:.3f}"
                )
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(
            other, (int, float)
        ), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f"**{other}")

        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad

        out._backward = _backward

        return out

    def __neg__(self):  # -self
        return self * -1

    def __radd__(self, other):  # other + self
        return self.__add__(other)

    def __sub__(self, other):  # self - other
        return self + (-other)

    def __rsub__(self, other):  # other - self
        return self.__sub__(other)

    def __rmul__(self, other):  # other * self
        return self.__mul__(other)

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), "ReLU")

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward

        return out

    def backward(self):

        # topological order all of the children in the graph
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()

    def __repr__(self):
        return f"Value(data={self.data:+.5f}, grad={self.grad:+.5f}, op={self._op})"


class Xs(Value):
    def __init__(self, xs: np.ndarray, off: int):
        super().__init__(data=None)
        self.xs = xs
        self.off = off
        self.name = f"I[{off}]"

    @property
    def data(self):
        value = self.xs[self.off]
        self.name = f"I({value:+.3f})"
        return value

    @data.setter
    def data(self, _):
        pass

    def query(self):
        return self


class Neuron:
    __slots__ = ("inputs_refs", "w", "b", "name", "inputs", "data")

    def __init__(self, name, inputs_refs):
        self.inputs_refs = inputs_refs
        self.w = [
            Value(np.random.random() * 2 - 1, name="W")
            for _ in range(len(self.inputs_refs))
        ]
        self.b = Value(0, name="B")
        self.name = name
        self.inputs = []
        self.data = None

    def query(self):
        if not self.data:
            if DEBUG:
                print(self)

            act = self.b
            for wi, xi in zip(self.w, self.inputs):
                # inputs can be Inputs or Neuron
                xi = xi.query()
                act += wi * xi
            self.data = act.relu()
        else:
            if DEBUG:
                pprint(f"[yellow]Cached {self.name}={self.data}")
        return self.data

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"ReLU({self.inputs_refs})[{self.data}]"


class SLP:
    def __init__(self, wiring: dict[str, Union[str, int]], nin: int):
        self.wiring = wiring
        self.neurons = {}
        self.inputs = np.empty(shape=nin)
        self.Xs = [Xs(self.inputs, i) for i in range(nin)]
        # first create the neurons
        for neuron, inputs in wiring.items():
            self.neurons[neuron] = Neuron(name=neuron, inputs_refs=inputs)
        # then wire up the bits
        for neuron in self.neurons.values():
            for to in neuron.inputs_refs:
                if type(to) == str:
                    neuron.inputs.append(self.neurons[to])
                else:
                    neuron.inputs.append(self.Xs[to])

    def _set_inputs(self, inputs: np.ndarray):
        """
        Sets the inputs to be read by Input instances.
        Input objects contain a pointer to self.input,
        and an offset to know which value to read.
        """
        self.inputs[:] = inputs

    def _reset(self):
        """
        Reset the caching of intermediate values.
        Called before a new forward pass.
        """
        for neuron in self.neurons.values():
            neuron.data = None

    def __call__(self, inps: np.ndarray, outs: list[str]):
        self._set_inputs(inps)
        self._reset()
        outputs = [self.neurons[out].query() for out in outs]
        return outputs

    def parameters(self):
        return [p for neuron in self.neurons.values() for p in neuron.parameters()]

    def __repr__(self):
        return f"SLP{self.wiring}"

    def zero_grad(self):
        for p in self.Xs:
            p.grad = 0
        for p in self.parameters():
            p.grad = 0


#%%


np.random.seed(SEED)

inputs = np.random.rand(4).astype(np.float32)
outputs = ["B", "C"]
wiring = {"A": [0, 1, 2], "B": [2, 3, "A"], "C": [0, "A", "B"]}

net = SLP(wiring=wiring, nin=len(inputs))

pprint(net)
st = monotonic()
out = net(inputs, outputs)
loss1 = 0.5 * sum([(1 - o) ** 2 for o in out])
et = monotonic()

pprint(f"[red]Elapsed {et - st: .7f}")
print("Out", out)
print("Loss", loss1.data)

loss1.name = "Loss"
loss1.backward()

if DEBUG:
    draw_micro(loss1)

lr = 0.1
for p in net.parameters():
    if DEBUG:
        print("Params", p)
    p.data = p.data - lr * p.grad

net.zero_grad()
out = net(inputs, outputs)
loss2 = 0.5 * sum([(1 - o) ** 2 for o in out])
print("Loss", loss2.data)

assert loss2.data <= loss1.data, f"Loss has increased! {loss1.data} > {loss2.data}"
