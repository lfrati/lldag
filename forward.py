from ctypes import CFUNCTYPE, POINTER, c_float, c_int
import os
from time import monotonic
from types import SimpleNamespace

from llvmlite import ir
from rich import print as pprint

from llvm import LLVM

DEBUG = int(os.getenv("DEBUG", "0"))
FMA = int(os.getenv("FMA", "0"))
SEED = int(os.getenv("SEED", "4"))
CONSTANTS = {}

c_float_p = POINTER(c_float)

zeroi = ir.Constant(ir.IntType(32), 0)
zerof = ir.Constant(ir.FloatType(), 0)


def index(i):
    return ir.Constant(ir.IntType(32), i)


class Value:
    def __init__(self, binfo, data=None, _op=""):
        self.data = data
        self.grad = 0
        self._op = _op
        self.name = f"V({self._op})"
        self.binfo = binfo

    def query(self):
        return self.data

    def __mul__(self, other):
        data = self.binfo.builder.fmul(
            self.data, other.data, name=f"{self.name}x{other.name}"
        )
        out = Value(binfo=self.binfo, data=data, _op="*")
        return out

    def __add__(self, other):
        data = self.binfo.builder.fadd(
            self.data, other.data, name=f"{self.name}+{other.name}"
        )
        out = Value(binfo=self.binfo, data=data, _op="+")
        return out

    def fma(self, other, other2):
        data = self.binfo.builder.fma(self.data, other.data, other2.data)
        out = Value(binfo=self.binfo, data=data, _op="fma")
        return out

    def relu(self):
        cmp = self.binfo.builder.fcmp_ordered("<=", zerof, self.data, flags=("fast",))
        data = self.binfo.builder.select(cmp, self.data, zerof, name="relu")
        out = Value(binfo=self.binfo, data=data, _op="ReLU")
        return out

    def __radd__(self, other):
        return self.__add__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __repr__(self):
        return f"{self.__class__.__name__}:{self.data.__class__.__name__}"


class Const(Value):
    def __init__(self, binfo, c):
        super().__init__(binfo=binfo)
        self.name = f"C{c}"
        if not c in CONSTANTS:
            CONSTANTS[c] = ir.Constant(ir.FloatType(), c)
        self.data = CONSTANTS[c]


class Input(Value):
    def __init__(self, binfo, off=0):
        super().__init__(binfo=binfo)
        self.off = off
        self.name = f"I{off}"

    def query(self):
        ptr = self.binfo.builder.gep(
            self.binfo.fargs[0], [index(self.off)], inbounds=False
        )
        self.data = self.binfo.builder.load(ptr, self.name)
        return self.data


class Weight(Value):
    def __init__(self, binfo, ix=0, off=0):
        super().__init__(binfo=binfo)
        self.ix = ix
        self.off = off
        self.name = f"W{ix}_{off}"

    def query(self):
        ptr = self.binfo.builder.gep(
            self.binfo.fargs[self.ix], [index(self.off)], inbounds=False
        )
        self.data = self.binfo.builder.load(ptr, name=self.name)
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
                if FMA:
                    pre = inp.fma(w, pre)
                else:
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
            retval.data, binfo.builder.gep(fargs[-1], [index(i)], inbounds=False)
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

    args_types = [c_int, *([c_float_p] * NARGS)]

    return CFUNC(bitcode=bitcode, fname=fname, args_types=args_types)


if __name__ == "__main__":
    import numpy as np
    from builder import make_dag, show

    # NNODES = 16
    # NINPS = 8
    # NOUTS = 4
    # MINED = 4
    # MAXED = 5

    NNODES = 4
    NINPS = 3
    NOUTS = 2
    MINED = 2
    MAXED = 3

    generator = np.random.default_rng(SEED)
    inputs = generator.random(NINPS).astype(np.float32)
    wiring = make_dag(
        n_nodes=NNODES,
        n_inputs=len(inputs),
        min_edges=MINED,
        max_edges=MAXED,
        seed=SEED,
    )
    outputs = list(wiring.keys())[-NOUTS:]

    weights = {
        out: np.random.uniform(-1, 1, size=(len(ins))) for out, ins in wiring.items()
    }

    if DEBUG:
        show(wiring, weights)

    cfunc = build(wiring, "grapher")

    results = np.zeros(len(outputs), dtype=np.float32)
    args = [inputs] + list(weights.values()) + [results]

    # prepare arrays to pass them as pointers
    args = [w.ctypes.data_as(c_float_p) for w in args]

    start = monotonic()
    ret = cfunc(*args)
    end = monotonic()
    assert ret == 0
    pprint(f"Elapsed {end - start:.9f}")
    print(results)
