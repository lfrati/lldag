from ctypes import CFUNCTYPE, POINTER, c_float, c_int
from time import monotonic

from llvmlite import ir
import numpy as np
from rich import print as pprint

from llvm import LLVM

CNT = 1_000  # runs to time
VALS = -1  # Values counter
ACTS = -1  # Values counter
NEUS = -1  # Neurons counter
CONSTANTS = {}

c_float_p = POINTER(c_float)

zeroi = ir.Constant(ir.IntType(32), 0)
zerof = ir.Constant(ir.FloatType(), 0)


def index(i):
    return ir.Constant(ir.IntType(32), i)


class Op:
    def __repr__(self):
        return self.__class__.__name__


class ReLU(Op):
    def __call__(self, pre):
        global ACTS
        ACTS += 1
        data = f"a{ACTS}"
        cmp = builder.fcmp_ordered("<=", zerof, pre.data, flags=("fast",))
        data = builder.select(cmp, pre.data, zerof)
        return data


class Nop(Op):
    def __call__(self, x):
        return x


class Value:
    def __init__(self, data):
        self.data = data

    def query(self):
        return self.data

    def __repr__(self):
        return self.data

    def __mul__(self, other):
        global VALS
        VALS += 1
        data = f"v{VALS}"
        data = builder.fmul(self.data, other.data)
        return Value(data)

    def __add__(self, other):
        global VALS
        VALS += 1
        data = f"v{VALS}"
        data = builder.fadd(self.data, other.data)
        return Value(data)

    def __radd__(self, other):
        return self.__add__(other)

    def __rmul__(self, other):
        return self.__mul__(other)


class Const(Value):
    def __init__(self, c):
        if c in CONSTANTS:
            self.data = CONSTANTS[c]
        else:
            self.data = ir.Constant(ir.FloatType(), c)
            CONSTANTS[c] = self.data


class Input(Value):
    def __init__(self, off=0):
        self.off = off
        self.data = f"inp_{off}"  # for builder

    def query(self):
        self.data = builder.load(builder.gep(fargs[0], [index(self.off)]))
        return self.data


class Weight(Value):
    def __init__(self, ix=0, off=0):
        self.ix = ix
        self.off = off
        self.data = f"w{ix}_{off}"  # for builder

    def query(self):
        self.data = builder.load(builder.gep(fargs[self.ix], [index(self.off)]))
        return self.data


class Neuron(Value):
    def __init__(self, name):
        self.ix = neuron2idx[name]
        self.op = Nop()
        self.inputs = []
        self.weights = []
        self.data = None
        self.name = name

    def query(self):
        if not self.data:
            pre = Const(0)
            for _, (inp, w) in enumerate(zip(self.inputs, self.weights)):
                inp.query()
                w.query()
                pre = pre + (inp * w)
            self.data = self.op(pre)
        return self.data

    def __repr__(self):
        return f"{self.name} : {self.op}({[el.data for el in self.inputs]})"


ops = {
    "A": ReLU(),
    "B": ReLU(),
    "C": ReLU(),
    "D": ReLU(),
    "E": ReLU(),
}
inputs = np.array([1.0, -1.0, 2.0], dtype=np.float32)
outputs = ["D", "E"]
results = np.zeros(len(outputs), dtype=np.float32)
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
    print(neuron)
print("------------")


float_ptr = ir.PointerType(ir.FloatType())
# there must be a pointer for each neuron
NARGS = 2 + len(wiring.keys())  # +1 for input +1 for outputs
fnty = ir.FunctionType(ir.IntType(32), (float_ptr for _ in range(NARGS)))
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


for i, retval in enumerate(retvals):
    builder.store(retval, builder.gep(fargs[-1], [index(i)]))
builder.ret(zeroi)

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

args = (
    [inputs]
    + [np.random.rand(len(edges)).astype(dtype=np.float32) for edges in wiring.values()]
    + [results]
)
pprint(args)

# prepare arrays to pass them as pointers
args = [w.ctypes.data_as(c_float_p) for w in args]

func_ptr = llvm_manager.get_fptr("grapher")

# Run the function via ctypes
c_float_p = POINTER(c_float)
cfunc = CFUNCTYPE(c_int, *([c_float_p] * NARGS))(func_ptr)
start = monotonic()
for _ in range(CNT):
    ret = cfunc(*args)
end = monotonic()
print(f"{(end - start)/CNT:.9f}")
# rint("  retval =", ret)
print(results)

# print("expected =", (inputs * weights).sum())
# assert ret == (inputs * weights).sum()
