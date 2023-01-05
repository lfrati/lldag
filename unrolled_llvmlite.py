from ctypes import CFUNCTYPE, POINTER, c_float
import os
from time import monotonic

from llvmlite import ir
import numpy as np

from llvm import LLVM


def index(i):
    return ir.Constant(ir.IntType(32), i)


def make_fun_arr(fma=False):
    arr = ir.ArrayType(ir.FloatType(), N)
    fnty = ir.FunctionType(ir.FloatType(), (arr, arr))

    # Create an empty module...
    module = ir.Module(name=__file__)

    func = ir.Function(module, fnty, name="node")

    block = func.append_basic_block(name="entry")
    builder = ir.IRBuilder(block)
    accum = ir.Constant(ir.FloatType(), 0)
    xs, ws = func.args
    for i in range(N):
        x = builder.extract_value(xs, i)
        w = builder.extract_value(ws, i)

        if fma:
            accum = builder.fma(x, w, accum)
        else:
            value = builder.fmul(x, w)
            accum = builder.fadd(accum, value)

    builder.ret(accum)

    return module


def make_fun_ptr(fma=False):
    float_ptr = ir.PointerType(ir.FloatType())
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
ARR = bool(int(os.getenv("ARR", "1")))
N = int(os.getenv("N", "8"))

print(f"{FMA=}")
print(f"{ARR=}")
print(f"{N=}")

if ARR:
    module = make_fun_arr(FMA)
else:
    module = make_fun_ptr(FMA)

if N <= 32:
    print(">>> LLVM IR ================================")
    print(module)
    print("============================================\n")

opt_module = llvm_manager.optimize_ir(module)

if N <= 32:
    print(">>> OPTIMIZED ==============================")
    print(opt_module)
    print("============================================\n")

comp_mod = llvm_manager.compile_ir(module)

if N <= 32:
    print(">>> ASM ====================================")
    llvm_manager.show_asm(comp_mod)
    print("============================================\n")

# Look up the function pointer (a Python int)
func_ptr = llvm_manager.get_fptr("node")


inputs = np.ones(N, dtype=np.float32)
weights = np.ones(N, dtype=np.float32) * 2

if ARR:
    c_float_arr = c_float * N
    cfunc = CFUNCTYPE(c_float, c_float_arr, c_float_arr)(func_ptr)
    inps, ws = (c_float_arr)(*inputs), (c_float_arr)(*weights)
else:
    c_float_p = POINTER(c_float)
    cfunc = CFUNCTYPE(c_float, c_float_p, c_float_p)(func_ptr)
    inps, ws = inputs.ctypes.data_as(c_float_p), weights.ctypes.data_as(c_float_p)

st = monotonic()
ret = cfunc(inps, ws)
et = monotonic()
expected = 2 * N
print(f"Elapsed {et - st:.9f}")
print("  retval =", ret)
print("expected =", expected)
assert ret == expected
