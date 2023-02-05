from ctypes import CFUNCTYPE, POINTER, c_float
import os
from time import monotonic

from llvmlite import ir
import numpy as np
from rich import print as pprint

from llvm import LLVM


def index(i):
    return ir.Constant(ir.IntType(32), i)


float_ptr = ir.PointerType(ir.FloatType())


def make_fun(fma=False):
    fnty = ir.FunctionType(ir.FloatType(), (float_ptr, float_ptr))

    # Create an empty module...
    module = ir.Module(name=__file__)

    func = ir.Function(module, fnty, name="node")

    block = func.append_basic_block(name="entry")
    builder = ir.IRBuilder(block)
    accum = ir.Constant(ir.FloatType(), 0)
    xs, ws = func.args
    for i in range(N):
        x = builder.load(builder.gep(xs, [index(i)]))
        w = builder.load(builder.gep(ws, [index(i)]))

        if fma:
            accum = builder.fma(x, w, accum)
        else:
            value = builder.fmul(x, w)
            accum = builder.fadd(accum, value)

    builder.ret(accum)

    return module


llvm_manager = LLVM()

FMA = bool(int(os.getenv("FMA", "1")))
N = int(os.getenv("N", "8"))

print(f"{FMA=}")
print(f"{N=}")

module = make_fun(FMA)

if N <= 32:
    print(">>> LLVM IR ================================")
    pprint(module)
    print("============================================\n")

opt_module = llvm_manager.optimize_ir(module)

if N <= 32:
    print(">>> OPTIMIZED ==============================")
    pprint(opt_module)
    print("============================================\n")

comp_mod = llvm_manager.compile_ir(module)

if N <= 32:
    print(">>> ASM ====================================")
    llvm_manager.show_asm(comp_mod)
    print("============================================\n")

# Look up the function pointer (a Python int)
func_ptr = llvm_manager.get_fptr("node")


inputs = np.random.rand(N).astype(np.float32)
weights = np.random.rand(N).astype(np.float32)

c_float_p = POINTER(c_float)
cfunc = CFUNCTYPE(c_float, c_float_p, c_float_p)(func_ptr)
inps, ws = inputs.ctypes.data_as(c_float_p), weights.ctypes.data_as(c_float_p)

st = monotonic()
ret = cfunc(inps, ws)
et = monotonic()
expected = (inputs * weights).sum()
print(f"Elapsed {et - st:.9f}")
print("  retval =", ret)
print("expected =", expected)
assert np.allclose(ret, expected)
