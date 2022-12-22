from ctypes import CFUNCTYPE, POINTER, c_float, c_int
import os
from time import monotonic
from types import SimpleNamespace

from llvmlite import ir
import numpy as np
from rich import print as pprint

from draw import draw_dot
from llvm import LLVM

DRAW = int(os.getenv("DRAW", "0"))
DEBUG = bool(int(os.getenv("DEBUG", "0")))
CONSTANTS = {}

c_float_p = POINTER(c_float)

zeroi = ir.Constant(ir.IntType(32), 0)
zerof = ir.Constant(ir.FloatType(), 0)


def index(i):
    return ir.Constant(ir.IntType(32), i)


class Value:
    def __init__(self, binfo, data=None, _children=(), _op=""):
        self.data = data
        self.grad = 0
        self._prev = set(_children)
        self._op = _op
        self.name = f"V({self._op})"
        self.binfo = binfo

    def query(self):
        return self.data

    def __mul__(self, other):
        data = self.binfo.builder.fmul(self.data, other.data)
        out = Value(binfo=self.binfo, data=data, _children=(self, other), _op="*")
        return out

    def __add__(self, other):
        data = self.binfo.builder.fadd(self.data, other.data)
        out = Value(binfo=self.binfo, data=data, _children=(self, other), _op="+")
        return out

    def relu(self):
        cmp = self.binfo.builder.fcmp_ordered("<=", zerof, self.data, flags=("fast",))
        data = self.binfo.builder.select(cmp, self.data, zerof)
        return Value(binfo=self.binfo, data=data, _children=(self,), _op="ReLU")

    def __radd__(self, other):
        return self.__add__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __repr__(self):
        return f"{self.__class__.__name__}:{self.data.__class__.__name__}"


class Const(Value):
    def __init__(self, binfo, c):
        super().__init__(binfo=binfo)
        self.name = f"c{c}"
        if not c in CONSTANTS:
            CONSTANTS[c] = ir.Constant(ir.FloatType(), c)
        self.data = CONSTANTS[c]


class Input(Value):
    def __init__(self, binfo, off=0):
        super().__init__(binfo=binfo)
        self.off = off
        self.name = f"inp{off}"

    def query(self):
        ptr = self.binfo.builder.gep(
            self.binfo.fargs[0], [index(self.off)], inbounds=True
        )
        self.data = self.binfo.builder.load(ptr)
        return self.data


class Weight(Value):
    def __init__(self, binfo, ix=0, off=0):
        super().__init__(binfo=binfo)
        self.ix = ix
        self.off = off
        self.name = f"w{ix}_{off}"

    def query(self):
        ptr = self.binfo.builder.gep(
            self.binfo.fargs[self.ix], [index(self.off)], inbounds=True
        )
        self.data = self.binfo.builder.load(ptr)
        return self.data


class Neuron(Value):
    def __init__(self, binfo, name):
        super().__init__(binfo=binfo)
        self.ix = binfo.neuron2idx[name]
        self.inputs = []
        self.weights = []
        self.data = None
        self.value = None

    def query(self):
        if not self.data:
            pre = Const(binfo=self.binfo, c=0)
            for _, (inp, w) in enumerate(zip(self.inputs, self.weights)):
                inp.query()
                w.query()
                pre = pre + (inp * w)
            value = pre.relu()
            self.data = value.data
            self.value = value
        return self.value


class CFUNC:
    def __init__(self, bitcode, fname, args_types):
        import llvmlite.binding as llvm

        self.bitcode = bitcode
        self.fname = fname
        target_machine = llvm.Target.from_triple(
            llvm.get_process_triple()
        ).create_target_machine()
        self.args_types = args_types
        self.module = llvm.parse_bitcode(bitcode)
        self.engine = llvm.create_mcjit_compiler(self.module, target_machine)
        self.faddr = self.engine.get_function_address(fname)
        self.cfunc = CFUNCTYPE(*self.args_types)(self.faddr)

    def __call__(self, *args, **kwargs):
        return self.cfunc(*args, **kwargs)


def build(wiring, fname):

    llvm_manager = LLVM()

    float_ptr = ir.PointerType(ir.FloatType())
    # there must be a pointer for each neuron
    NARGS = 2 + len(wiring.keys())  # +1 for input +1 for outputs
    fnty = ir.FunctionType(ir.IntType(32), (float_ptr for _ in range(NARGS)))

    module = ir.Module(name=__file__)
    func = ir.Function(module, fnty, name=fname)
    block = func.append_basic_block(name="entry")
    builder = ir.IRBuilder(block)

    fargs = func.args
    neuron2idx = {neuron: i for i, neuron in enumerate(wiring.keys())}
    build_info = {"builder": builder, "fargs": fargs, "neuron2idx": neuron2idx}
    binfo = SimpleNamespace(**build_info)

    neurons = {neuron: Neuron(binfo=binfo, name=neuron) for neuron in wiring.keys()}
    for name, neuron in neurons.items():
        for i, edge in enumerate(wiring[name]):
            if type(edge) == int:
                neuron.inputs.append(Input(binfo=binfo, off=edge))
            else:
                neuron.inputs.append(neurons[edge])
            neuron.weights.append(Weight(binfo=binfo, ix=neuron.ix, off=i))

    retvals = []
    for output in outputs:
        if output in neurons:
            retvals.append(neurons[output].query())

    for i, retval in enumerate(retvals):
        binfo.builder.store(
            retval.data, binfo.builder.gep(fargs[-1], [index(i)], inbounds=True)
        )
    binfo.builder.ret(zeroi)

    if DEBUG:
        print(">>> LLVM IR ================================")
        print(module)
        print("============================================\n")

    opt_module = llvm_manager.optimize_ir(module)

    if DEBUG:
        print(">>> OPTIMIZED ==============================")
        print(opt_module)
        print("============================================\n")

    comp_mod = llvm_manager.compile_ir(opt_module)

    if DEBUG:
        print(">>> ASM ====================================")
        llvm_manager.show_asm(comp_mod)
        print("============================================\n")

    bitcode = comp_mod.as_bitcode()

    # func_ptr = llvm_manager.get_fptr(fname)

    args_types = [c_int, *([c_float_p] * NARGS)]

    # Run the function via ctypes
    # cfunc = CFUNCTYPE(*args_types)(func_ptr)

    return CFUNC(bitcode=bitcode, fname=fname, args_types=args_types), retvals


if __name__ == "__main__":

    np.random.seed(4)

    inputs = np.random.rand(4).astype(np.float32)
    outputs = ["B", "C"]
    wiring = {"A": [0, 1, 2], "B": [2, 3, "A"], "C": [0, "A", "B"]}

    # inputs = np.random.rand(8).astype(np.float32)
    # outputs = ["C", "D"]
    # results = np.zeros(len(outputs), dtype=np.float32)
    # wiring = {
    #     "A": [14, 3, 15, 8],
    #     "B": [13, 2, "A"],
    #     "C": [8, 5, 15, 10, 11],
    #     "D": [11, 3, "A", "C"],
    # }

    # inputs = np.random.rand(16).astype(np.float32)
    # outputs = ["Y", "Z"]
    # results = np.zeros(len(outputs), dtype=np.float32)
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

    cfunc, retvals = build(wiring, "grapher")

    results = np.zeros(len(outputs), dtype=np.float32)
    args = (
        [inputs]
        + [
            np.random.rand(len(edges)).astype(dtype=np.float32)
            for edges in wiring.values()
        ]
        + [results]
    )

    if DRAW:
        print("drawing", retvals[0])
        dot = draw_dot(retvals[0])

    # prepare arrays to pass them as pointers
    args = [w.ctypes.data_as(c_float_p) for w in args]

    start = monotonic()
    ret = cfunc(*args)
    end = monotonic()
    assert ret == 0
    print(f"{end - start:.9f}")
    print(results)
