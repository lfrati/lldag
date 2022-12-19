from ctypes import POINTER, c_float

from llvmlite import ir
import numpy as np
from rich import print as pprint

from llvm import LLVM

VALS = -1  # Values counter
ACTS = -1  # Values counter
NEUS = -1  # Neurons counter
CONSTANTS = {}

c_float_p = POINTER(c_float)


class Op:
    def __repr__(self):
        return self.__class__.__name__


class ReLU(Op):
    def __call__(self, pre):
        global ACTS
        ACTS += 1
        label = f"a{ACTS}"
        cmp = builder.fcmp_ordered(
            "<=", ir.Constant(ir.FloatType(), 0), pre.label, flags=("fast",)
        )
        label = builder.select(cmp, pre.label, ir.Constant(ir.FloatType(), 0))
        return label


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
        label = builder.fmul(self.label, other.label)
        return Value(label)

    def __add__(self, other):
        global VALS
        VALS += 1
        label = f"v{VALS}"
        label = builder.fadd(self.label, other.label)
        return Value(label)

    def __radd__(self, other):
        return self.__add__(other)

    def __rmul__(self, other):
        return self.__mul__(other)


class Const(Value):
    def __init__(self, c):
        if c in CONSTANTS:
            self.label = CONSTANTS[c]
        else:
            self.label = ir.Constant(ir.FloatType(), c)
            CONSTANTS[c] = self.label


class Input(Value):
    def __init__(self, off=0):
        self.off = off
        self.label = f"inp_{off}"  # for builder

    def query(self):
        self.label = builder.load(
            builder.gep(fargs[0], [ir.Constant(ir.IntType(32), self.off)])
        )
        return self.label


class Weight(Value):
    def __init__(self, ix=0, off=0):
        self.ix = ix
        self.off = off
        self.label = f"w{ix}_{off}"  # for builder

    def query(self):
        self.label = builder.load(
            builder.gep(fargs[self.ix], [ir.Constant(ir.IntType(32), self.off)])
        )
        return self.label


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
            pre = Const(0)
            for _, (inp, w) in enumerate(zip(self.inputs, self.weights)):
                inp.query()
                w.query()
                pre = pre + (inp * w)
            self.label = self.op(pre)
        else:
            pprint(f"#[green] Neuron {self.name} CACHED")
        return self.label

    def __repr__(self):
        return f"{self.name} : {self.op}({[el.label for el in self.inputs]})"


# print(D.query())

ops = {
    "A": ReLU(),
    "B": ReLU(),
    "C": ReLU(),
    "D": ReLU(),
    "E": ReLU(),
}
outputs = ["D", "E"]
wiring = {
    "A": [0, 2],
    "B": [1, 2],
    "C": [1, "B"],
    "D": ["A", "B", "C"],
    "E": ["A", 0, "C"],
}
neuron2idx = {neuron: i for i, neuron in enumerate(wiring.keys())}


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


llvm_manager = LLVM()

neurons = build(wiring, ops)

for neuron in neurons.values():
    # for x in node.inputs:
    print(neuron)
print("------------")


void = ir.VoidType()
float_ptr = ir.PointerType(ir.FloatType())
# there must be a pointer for each neuron
NARGS = 1 + len(wiring.keys())  # +1 for input
fnty = ir.FunctionType(ir.FloatType(), (float_ptr for _ in range(NARGS)))
# Create an empty module...
module = ir.Module(name=__file__)
# and declare a function named "fpadd" inside it
func = ir.Function(module, fnty, name="grapher")
# Now implement the function
block = func.append_basic_block(name="entry")
builder = ir.IRBuilder(block)
print(len(func.args))
fargs = func.args


retvals = []
for output in outputs:
    if output in neurons:
        retvals.append(neurons[output].query())
print("TO RETURN")
builder.ret(retvals[0])

print(">>> LLVM IR ================================")
print(module)
print("============================================\n")

opt_module = llvm_manager.optimize_ir(module)

print(">>> OPTIMIZED ==============================")
print(opt_module)
print("============================================\n")

comp_mod = llvm_manager.compile_ir(opt_module)

print(">>> ASM ====================================")
llvm_manager.show_asm(comp_mod)
print("============================================\n")

weights = [np.random.rand(len(edges)) for edges in wiring.values()]

# prepare arrays to pass them as pointers
# print([w.ctypes.data_as(c_float_p) for w in weights])
# args = []
