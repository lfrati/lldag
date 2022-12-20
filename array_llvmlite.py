from ctypes import CFUNCTYPE, c_float
from time import monotonic

from llvmlite import ir
import numpy as np

from llvm import LLVM

N = 4


def make_fun():
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
        value = builder.fmul(x, w)
        accum = builder.fadd(accum, value)
        # accum = builder.fma(x, w, accum)

    builder.ret(accum)

    return module


llvm_manager = LLVM()

module = make_fun()

print(">>> LLVM IR ================================")
print(module)
print("============================================\n")

opt_module = llvm_manager.optimize_ir(module)

print(">>> OPTIMIZED ==============================")
print(opt_module)
print("============================================\n")

comp_mod = llvm_manager.compile_ir(module)

print(">>> ASM ====================================")
llvm_manager.show_asm(comp_mod)
print("============================================\n")

# Look up the function pointer (a Python int)
func_ptr = llvm_manager.get_fptr("node")


inputs = np.arange(N, dtype=np.float32)
weights = np.arange(N, dtype=np.float32)

c_float_arr = c_float * N
cfunc = CFUNCTYPE(c_float, c_float_arr, c_float_arr)(func_ptr)
inps, ws = (c_float_arr)(*inputs), (c_float_arr)(*weights)
st = monotonic()
ret = cfunc(inps, ws)
et = monotonic()
print(f"Elapsed {et - st:.9f}")
print("  retval =", ret)
print("expected =", (inputs * weights).sum())
assert ret == (inputs * weights).sum()
