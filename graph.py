from ctypes import CFUNCTYPE, POINTER, c_float, c_int
from time import monotonic

from llvmlite import ir
import numpy as np
from rich import print as pprint

from llvm import LLVM

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
        cmp = builder.fcmp_ordered("<=", zerof, pre.data, flags=("fast",))
        data = builder.select(cmp, pre.data, zerof)
        return data


class Nop(Op):
    def __call__(self, x):
        return x


class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = data
        self.grad = 0
        self._children = _children
        self._op = _op
        self._backward = lambda: None

    def query(self):
        return self.data

    def __mul__(self, other):
        data = builder.fmul(self.data, other.data)
        out = Value(data, _children=(self, other), _op="*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    def __add__(self, other):
        data = builder.fadd(self.data, other.data)
        out = Value(data, _children=(self, other), _op="+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

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

    def query(self):
        self.data = builder.load(
            builder.gep(fargs[0], [index(self.off)], inbounds=True)
        )
        return self.data


class Weight(Value):
    def __init__(self, ix=0, off=0):
        self.ix = ix
        self.off = off

    def query(self):
        self.data = builder.load(
            builder.gep(fargs[self.ix], [index(self.off)], inbounds=True)
        )
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


if __name__ == "__main__":
    np.random.seed(4)

    inputs = np.random.rand(16).astype(np.float32)
    outputs = ["Y", "Z"]
    results = np.zeros(len(outputs), dtype=np.float32)
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

    float_ptr = ir.PointerType(ir.FloatType())
    # there must be a pointer for each neuron
    NARGS = 2 + len(wiring.keys())  # +1 for input +1 for outputs
    fnty = ir.FunctionType(ir.IntType(32), (float_ptr for _ in range(NARGS)))

    module = ir.Module(name=__file__)
    func = ir.Function(module, fnty, name="grapher")
    block = func.append_basic_block(name="entry")
    builder = ir.IRBuilder(block)
    fargs = func.args

    retvals = []
    for output in outputs:
        if output in neurons:
            retvals.append(neurons[output].query())

    for i, retval in enumerate(retvals):
        builder.store(retval, builder.gep(fargs[-1], [index(i)], inbounds=True))
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
        + [
            np.random.rand(len(edges)).astype(dtype=np.float32)
            for edges in wiring.values()
        ]
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
    ret = cfunc(*args)
    end = monotonic()
    assert ret == 0
    print(f"{end - start:.9f}")
    print(results)
