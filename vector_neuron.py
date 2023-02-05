from ctypes import CFUNCTYPE, c_float, c_int
from time import monotonic

from llvmlite import ir

# import llvmlite.binding as llvm
import numpy as np
import os

from llvm import LLVM

N = 64

FMA = bool(os.getenv("FMA", 0))


def i32(i):
    return ir.Constant(ir.IntType(32), i)


def f32(i):
    return ir.Constant(ir.FloatType(), i)


def neuron(N):

    float_ptr = ir.PointerType(ir.FloatType())

    # void = ir.VoidType()
    fnty = ir.FunctionType(ir.FloatType(), (float_ptr, float_ptr, ir.IntType(32)))

    # Create an empty module...
    module = ir.Module(name=__file__)

    func = ir.Function(module, fnty, name="node")

    # the entry block allows phi to distinguish the first iteration
    bb_entry = func.append_basic_block(name="entry")
    bb_body = func.append_basic_block(name="body")
    bb_exit = func.append_basic_block(name="exit")

    builder = ir.IRBuilder()

    xs, ws, n = func.args

    builder.position_at_end(bb_entry)
    # shameful ptr to vector loading
    xs_vec = ir.Constant(ir.VectorType(ir.FloatType(), N), None)
    ws_vec = ir.Constant(ir.VectorType(ir.FloatType(), N), None)
    for i in range(N):
        xs_ptr = builder.gep(xs, [i32(i)])
        ws_ptr = builder.gep(ws, [i32(i)])
        x = builder.load(xs_ptr)
        w = builder.load(ws_ptr)
        xs_vec = builder.insert_element(vector=xs_vec, value=x, idx=i32(i))
        ws_vec = builder.insert_element(vector=ws_vec, value=w, idx=i32(i))
    builder.branch(bb_body)

    builder.position_at_end(bb_body)
    # prepare looping: use phi to init indeces and accumulator on entry into block
    index = builder.phi(ir.IntType(32), name="index")
    # phi nodes use .add_incoming to specigy the values to take depending on the provenance.
    # set to 0 if coming from bb_entry
    index.add_incoming(ir.Constant(index.type, 0), bb_entry)
    accum = builder.phi(ir.FloatType(), name="acc_in")
    accum.add_incoming(ir.Constant(accum.type, 0), bb_entry)

    x = builder.extract_element(xs_vec, index)
    w = builder.extract_element(ws_vec, index)

    # xs_ptr = builder.gep(xs, [index])
    # ws_ptr = builder.gep(ws, [index])
    # x = builder.load(xs_ptr)
    # w = builder.load(ws_ptr)

    value = builder.fmul(x, w)
    added = builder.fadd(accum, value, name="acc_out")

    accum.add_incoming(added, bb_body)

    indexp1 = builder.add(index, ir.Constant(index.type, 1))
    index.add_incoming(indexp1, bb_body)  # increment by one if coming from bb_body

    cond = builder.icmp_unsigned("<", indexp1, n)
    builder.cbranch(cond, bb_body, bb_exit)

    builder.position_at_end(bb_exit)
    builder.ret(added)

    return module


llvm_manager = LLVM()

mod = neuron(N)

print(">>> LLVM IR ================================")
print(mod)
print("============================================\n")

mod = llvm_manager.optimize_ir(mod)
print(">>> OPTIMIZED ==============================")
print(mod)
print("============================================\n")

mod = llvm_manager.compile_ir(mod)

print(">>> ASM ====================================")
llvm_manager.show_asm(mod)
print("============================================\n")

# Look up the function pointer (a Python int)
func_ptr = llvm_manager.get_fptr("node")

inputs = np.arange(N, dtype=np.float32)
weights = np.arange(N, dtype=np.float32)

c_float_arr = np.ctypeslib.ndpointer(
    dtype=np.float32, ndim=1, shape=(N), flags="C_CONTIGUOUS"
)
cfunc = CFUNCTYPE(c_float, c_float_arr, c_float_arr, c_int)(func_ptr)

print("Running...")
st = monotonic()

inputs = np.arange(N, dtype=np.float32)
weights = np.arange(N, dtype=np.float32)

ret = cfunc(inputs, weights, N)
expected = (inputs * weights).sum()
et = monotonic()
print(f" RUN: {et - st: .8f}s")

assert ret == expected, f"Unexpected retval: {ret=} != {expected=}"
print("MATCH: OK")
