from ctypes import CFUNCTYPE, c_int
import os
from time import monotonic

from llvmlite import ir
import numpy as np
from numpy.ctypeslib import ndpointer

from llvm import LLVM

N_IN = 64
N_OUT = 32

FMA = bool(os.getenv("FMA", 0))


def i32(i):
    return ir.Constant(ir.IntType(32), i)


def f32(i):
    return ir.Constant(ir.FloatType(), i)


def relu(builder, value):
    cmp = builder.fcmp_ordered("<=", f32(0), value)
    return builder.select(cmp, value, f32(0), name="relu")


def neuron(module, func, builder, bb_entry, xs, ws, n, outputs):

    bb_body = func.append_basic_block(name=f"body")
    bb_exit = func.append_basic_block(name=f"exit")

    builder.branch(bb_body)
    builder.position_at_end(bb_body)
    # prepare looping: use phi to init indeces and accumulator on entry into block
    index = builder.phi(ir.IntType(32), name="index")
    # phi nodes use .add_incoming to specigy the values to take depending on the provenance.
    # set to 0 if coming from bb_entry
    index.add_incoming(ir.Constant(index.type, 0), bb_entry)
    accum = builder.phi(ir.FloatType(), name="acc_in")
    accum.add_incoming(ir.Constant(accum.type, 0), bb_entry)

    x = builder.load(builder.gep(xs, [index]))
    w = builder.load(builder.gep(ws, [index]))

    value = builder.fmul(x, w)
    added = builder.fadd(accum, value, name="acc_out")

    accum.add_incoming(added, bb_body)

    indexp1 = builder.add(index, ir.Constant(index.type, 1))
    index.add_incoming(indexp1, bb_body)  # increment by one if coming from bb_body

    cond = builder.icmp_unsigned("<", indexp1, i32(n))
    loop = builder.cbranch(cond, bb_body, bb_exit)

    # or "llvm.loop.unroll.enable"
    loop.set_metadata("llvm.loop", module.add_metadata(["llvm.loop.vectorize.enable"]))

    builder.position_at_end(bb_exit)
    result = relu(builder, added)
    for ptr, index in outputs:
        builder.store(result, builder.gep(ptr, [i32(index)], name=f"outputs"))

    return bb_exit


def forward(n_in, n_out):

    float_ptr = ir.PointerType(ir.FloatType())

    fnty = ir.FunctionType(ir.VoidType(), [float_ptr] * (n_out + 2))

    # Create an empty module...
    module = ir.Module(name=__file__)

    func = ir.Function(module, fnty, name="node")
    builder = ir.IRBuilder()

    x_ptr = func.args[0]
    ws_ptrs = func.args[1:-1]
    y_ptr = func.args[-1]

    bb_entry = func.append_basic_block(name="entry")
    builder.position_at_end(bb_entry)
    for i in range(n_out):
        ws_ptr = ws_ptrs[i]
        bb_entry = neuron(
            module=module,
            func=func,
            builder=builder,
            bb_entry=bb_entry,
            xs=x_ptr,
            ws=ws_ptr,
            n=n_in,
            outputs=[(y_ptr, i)],
        )

        if i == n_out - 1:
            builder.ret_void()

    return module


llvm_manager = LLVM()

mod = forward(n_in=N_IN, n_out=N_OUT)

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


inputs = np.random.randn(N_IN).astype(np.float32)
weights = np.random.randn(N_OUT, N_IN).astype(np.float32)
outputs = np.zeros(N_OUT, dtype=np.float32)


def to_ndptr(arr):
    return ndpointer(
        dtype=arr.dtype, ndim=arr.ndim, shape=arr.shape, flags="C_CONTIGUOUS"
    )


arr_ptrs = [to_ndptr(inputs)] + [to_ndptr(w) for w in weights] + [to_ndptr(outputs)]
cfunc = CFUNCTYPE(c_int, *arr_ptrs)(func_ptr)

print("Running...")
st = monotonic()
cfunc(inputs, *weights, outputs)
et = monotonic()
print(f" RUN: {et - st: .8f}s")
expected = np.maximum((inputs * weights).sum(axis=1), 0)

print(outputs)
assert np.allclose(outputs, expected), f"Unexpected retval: {outputs=} != {expected=}"
print("MATCH: OK")
