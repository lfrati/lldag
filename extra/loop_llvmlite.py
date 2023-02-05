from ctypes import CFUNCTYPE, POINTER, c_float, c_int
from time import monotonic

from llvmlite import ir

# import llvmlite.binding as llvm
import numpy as np
import os

from llvm import LLVM

N = 4
NLOOPS = 2

FMA = bool(os.getenv("FMA", 0))


def i32(i):
    return ir.Constant(ir.IntType(32), i)


def f32(i):
    return ir.Constant(ir.FloatType(), i)


def neuron(
    builder, inputs, weights, n, outputs, i, prev, curr_body, curr_exit, next, module
):

    builder.position_at_end(curr_body)

    # prepare looping: use phi to init indeces and accumulator on entry into block
    index = builder.phi(ir.IntType(32))
    # phi nodes use .add_incoming to specigy the values to take depending on the provenance.
    # set to 0 if coming from bb_entry
    index.add_incoming(ir.Constant(index.type, 0), prev)
    accum = builder.phi(ir.FloatType(), name="accum")
    accum.add_incoming(ir.Constant(accum.type, 0), prev)

    xs_ptr = builder.gep(inputs, [index])
    ws_ptr = builder.gep(weights, [index])
    input = builder.load(xs_ptr)
    weight = builder.load(ws_ptr)

    if FMA:
        added = builder.fma(input, weight, accum)
    else:
        value = builder.fmul(input, weight)
        added = builder.fadd(accum, value)

    accum.add_incoming(added, curr_body)

    indexp1 = builder.add(index, ir.Constant(index.type, 1))
    index.add_incoming(indexp1, curr_body)  # increment by one if coming from bb_loop

    cond = builder.icmp_unsigned("<", indexp1, n)
    loop = builder.cbranch(cond, curr_body, curr_exit)
    # from https://github.com/numba/llvmlite/issues/270#issuecomment-304376876
    # https://llvm.org/docs/LangRef.html#llvm-loop
    # or "llvm.loop.unroll.enable"
    loop.set_metadata("llvm.loop", module.add_metadata(["llvm.loop.vectorize.enable"]))

    builder.position_at_end(curr_exit)
    outputs = builder.insert_element(outputs, added, i32(i))
    builder.branch(next)
    return outputs


def make_multi_loop():

    # Create some useful types
    float_ptr = ir.PointerType(ir.FloatType())
    # void = ir.VoidType()
    fnty = ir.FunctionType(ir.FloatType(), (float_ptr, float_ptr, ir.IntType(32)))

    # Create an empty module...
    module = ir.Module(name=__file__)

    func = ir.Function(module, fnty, name="node")
    vec = ir.Constant(ir.VectorType(ir.FloatType(), NLOOPS), None)
    print(vec)
    print(vec.type)

    # the entry block allows phi to distinguish the first iteration
    bb_entry = func.append_basic_block(name="entry")
    bb_loops = [func.append_basic_block(name=f"loop_body_{i}") for i in range(NLOOPS)]
    bb_exits = [func.append_basic_block(name=f"loop_exit_{i}") for i in range(NLOOPS)]
    bb_exit = func.append_basic_block(name="exit")

    builder = ir.IRBuilder()

    # with multiple block I set the builder position like a cursor
    builder.position_at_end(bb_entry)

    builder.branch(bb_loops[0])

    xs, ws, n = func.args

    for i in range(NLOOPS):
        prev = bb_entry if i == 0 else bb_exits[i - 1]
        curr_body = bb_loops[i]
        curr_exit = bb_exits[i]
        next = bb_loops[i + 1] if i < len(bb_loops) - 1 else bb_exit

        vec = neuron(
            builder, xs, ws, n, vec, i, prev, curr_body, curr_exit, next, module
        )

    builder.position_at_end(bb_exit)
    val = reduce_sum(builder, vec, NLOOPS)
    builder.ret(val)

    return module


def reduce_sum(builder, ptr, N):
    acc = None
    for i in range(N):
        value = builder.extract_element(ptr, i32(i))
        if i == 0:
            acc = value
        else:
            acc = builder.fadd(acc, value)
    return acc


llvm_manager = LLVM()

module = make_multi_loop()

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

# Look up the function pointer (a Python int)
func_ptr = llvm_manager.get_fptr("node")

inputs = np.arange(N, dtype=np.float32)
weights = np.arange(N, dtype=np.float32)

# Run the function via ctypes
c_float_p = POINTER(c_float)
cfunc = CFUNCTYPE(c_float, c_float_p, c_float_p, c_int)(func_ptr)
st = monotonic()
inps = inputs.ctypes.data_as(c_float_p)
ws = weights.ctypes.data_as(c_float_p)
et = monotonic()
print(f"CAST: {et - st: .8f}s")

print("Running...")
st = monotonic()
ret = cfunc(inps, ws, N)
expected = (inputs * weights).sum() * NLOOPS
et = monotonic()
print(f" RUN: {et - st: .8f}s")

assert ret == expected, f"Unexpected retval: {ret=} != {expected=}"
print("MATCH: OK")
