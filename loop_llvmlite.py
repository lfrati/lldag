from ctypes import CFUNCTYPE, POINTER, c_float, c_int
from time import monotonic

from llvmlite import ir

# import llvmlite.binding as llvm
import numpy as np

from llvm import LLVM

N = 64
NLOOPS = 2


def i32(i):
    return ir.Constant(ir.IntType(32), i)


def f32(i):
    return ir.Constant(ir.FloatType(), i)


def make_single_loop():
    # Create some useful types
    float_ptr = ir.PointerType(ir.FloatType())
    # void = ir.VoidType()
    fnty = ir.FunctionType(ir.FloatType(), (float_ptr, float_ptr, ir.IntType(32)))

    # Create an empty module...
    module = ir.Module(name=__file__)

    func = ir.Function(module, fnty, name="node")

    # the entry block allows phi to distinguish the first iteration
    bb_entry = func.append_basic_block(name="entry")
    bb_loop = func.append_basic_block(name="loop")
    bb_exit = func.append_basic_block(name="exit")

    builder = ir.IRBuilder()

    # with multiple block I set the builder position like a cursor
    builder.position_at_end(bb_entry)
    builder.branch(bb_loop)
    builder.position_at_end(bb_loop)

    # prepare looping: use phi to init indeces and accumulator on entry into block
    index = builder.phi(ir.IntType(32))
    # phi nodes use .add_incoming to specigy the values to take depending on the provenance.
    index.add_incoming(
        ir.Constant(index.type, 0), bb_entry
    )  # set to 0 if coming from bb_entry
    accum = builder.phi(ir.FloatType())
    accum.add_incoming(ir.Constant(accum.type, 0), bb_entry)

    xs, ws, n = func.args

    xs_ptr = builder.gep(xs, [index])
    ws_ptr = builder.gep(ws, [index])
    input = builder.load(xs_ptr)
    weight = builder.load(ws_ptr)

    value = builder.fmul(input, weight)

    added = builder.fadd(accum, value)
    accum.add_incoming(added, bb_loop)

    indexp1 = builder.add(index, ir.Constant(index.type, 1))
    index.add_incoming(indexp1, bb_loop)  # increment by one if coming from bb_loop

    cond = builder.icmp_unsigned("<", indexp1, n)
    builder.cbranch(cond, bb_loop, bb_exit)

    builder.position_at_end(bb_exit)
    builder.ret(added)

    return module


def make_multi_loop():

    # Create some useful types
    float_ptr = ir.PointerType(ir.FloatType())
    # void = ir.VoidType()
    fnty = ir.FunctionType(ir.FloatType(), (float_ptr, float_ptr, ir.IntType(32)))

    # Create an empty module...
    module = ir.Module(name=__file__)

    func = ir.Function(module, fnty, name="node")

    # the entry block allows phi to distinguish the first iteration
    bb_entry = func.append_basic_block(name="entry")
    bb_exit = func.append_basic_block(name="exit")
    bb_loops = [func.append_basic_block(name=f"loop_{i}") for i in range(NLOOPS)]
    blocks = [bb_entry] + bb_loops + [bb_exit]

    builder = ir.IRBuilder()

    # with multiple block I set the builder position like a cursor
    builder.position_at_end(bb_entry)
    results = builder.alloca(ir.FloatType(), size=NLOOPS, name="results")
    builder.branch(bb_loops[0])

    # # results = builder.alloca(ir.FloatType(), size=NLOOPS, name="results")
    # results_ptr = ir.GlobalVariable(
    #     module=module,
    #     typ=ir.ArrayType(element=ir.FloatType(), count=NLOOPS),
    #     name="results",
    # )

    for i, (prev, curr, next) in enumerate(zip(blocks, blocks[1:], blocks[2:])):
        builder.position_at_end(curr)

        # prepare looping: use phi to init indeces and accumulator on entry into block
        index = builder.phi(ir.IntType(32))
        # phi nodes use .add_incoming to specigy the values to take depending on the provenance.
        index.add_incoming(
            ir.Constant(index.type, 0), prev
        )  # set to 0 if coming from bb_entry
        accum = builder.phi(ir.FloatType(),name="accum")
        accum.add_incoming(ir.Constant(accum.type, 0), prev)

        xs, ws, n = func.args

        xs_ptr = builder.gep(xs, [index])
        ws_ptr = builder.gep(ws, [index])
        input = builder.load(xs_ptr)
        weight = builder.load(ws_ptr)

        # value = builder.fmul(input, weight)
        # added = builder.fadd(accum, value)

        added = builder.fma(input, weight, accum)

        # val = builder.store(added, ptr)
        # builder.insert_value(added, results_ptr, int32(0))

        # builder.insert_value(added, all, int32(0))

        accum.add_incoming(added, curr)

        indexp1 = builder.add(index, ir.Constant(index.type, 1))
        index.add_incoming(indexp1, curr)  # increment by one if coming from bb_loop

        cond = builder.icmp_unsigned("<", indexp1, n)
        builder.store(added, builder.gep(results, [i32(i)]))
        builder.cbranch(cond, curr, next)

    builder.position_at_end(bb_exit)
    val = reduce_sum(builder, results, NLOOPS)
    builder.ret(val)

    return module


def reduce_sum(builder, ptr, N):
    acc = None
    for i in range(N):
        value = builder.load(builder.gep(ptr, [i32(i)]))
        if i == 0:
            acc = builder.fadd(f32(0), value)
        else:
            acc = builder.fadd(acc, value)
    return acc


llvm_manager = LLVM()

# module = make_single_loop()
module = make_multi_loop()

print(">>> LLVM IR ================================")
print(module)
print("============================================\n")

opt_module = llvm_manager.optimize_ir(module)

# print(">>> OPTIMIZED ==============================")
# print(opt_module)
# print("============================================\n")

comp_mod = llvm_manager.compile_ir(opt_module)

# print(">>> ASM ====================================")
# llvm_manager.show_asm(comp_mod)
# print("============================================\n")

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
print(f"CAST: {et - st: .8f}")

st = monotonic()
ret = cfunc(inps, ws, N)
expected = (inputs * weights).sum() * NLOOPS
et = monotonic()
print(f" RUN: {et - st: .8f}")

assert ret == expected, f"Unexpected retval: {ret=} != {expected=}"
print("MATCH: OK")
