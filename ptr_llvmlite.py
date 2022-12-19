from ctypes import CFUNCTYPE, POINTER, c_float, c_int

from llvmlite import ir

# import llvmlite.binding as llvm
import numpy as np

from llvm import LLVM

N = 64


def make_fun():
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
    index.add_incoming(ir.Constant(index.type, 0), bb_entry)
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
    index.add_incoming(indexp1, bb_loop)

    cond = builder.icmp_unsigned("<", indexp1, n)
    builder.cbranch(cond, bb_loop, bb_exit)

    builder.position_at_end(bb_exit)
    builder.ret(added)

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
ret = cfunc(inputs.ctypes.data_as(c_float_p), weights.ctypes.data_as(c_float_p), N)
print("  retval =", ret)
print("expected =", (inputs * weights).sum())
assert ret == (inputs * weights).sum()
