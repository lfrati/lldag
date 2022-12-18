from ctypes import CFUNCTYPE, c_float, POINTER, c_int

import llvmlite.binding as llvm

import numpy as np

N = 64

# All these initializations are required for code generation!
llvm.initialize()
llvm.initialize_native_target()
llvm.initialize_native_asmprinter()  # yes, even this one

from llvmlite import ir

# Create some useful types
float_ptr = ir.PointerType(ir.FloatType())
void = ir.VoidType()
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


target_machine = None


def create_execution_engine():
    """
    Create an ExecutionEngine suitable for JIT code generation on
    the host CPU.  The engine is reusable for an arbitrary number of
    modules.
    """
    global target_machine
    # Create a target machine representing the host
    target = llvm.Target.from_triple(llvm.get_process_triple())
    target_machine = target.create_target_machine()
    target_machine.set_asm_verbosity(True)
    # And an execution engine with an empty backing module
    backing_mod = llvm.parse_assembly("")
    engine = llvm.create_mcjit_compiler(backing_mod, target_machine)
    return engine


def optimize_ir(llvm_ir):
    mod = llvm.parse_assembly(llvm_ir)
    ######## OPTIMIZATION PASS ########
    # from https://blog.christianperone.com/2022/09/tutorial-on-using-llvm-to-jit-pytorch-fx-graphs-to-native-code-x86-arm-risc-v-wasm-part-i-scalars/
    pass_manager = llvm.PassManagerBuilder()
    pass_manager.opt_level = 3
    pass_manager.size_level = 0
    pass_manager.loop_vectorize = True
    pass_manager.slp_vectorize = True
    pass_module = llvm.ModulePassManager()
    pass_manager.populate(pass_module)
    pass_module.run(mod)
    ###################################
    return str(mod)


def compile_ir(engine, llvm_ir):
    """
    Compile the LLVM IR string with the given engine.
    The compiled module object is returned.
    """
    # Create a LLVM module object from the IR
    mod = llvm.parse_assembly(llvm_ir)

    mod.verify()
    # Now add the module and make sure it is ready for execution
    engine.add_module(mod)
    engine.finalize_object()
    engine.run_static_constructors()
    return mod


engine = create_execution_engine()
module.triple = llvm.get_process_triple()
# TODO: Why this warning that I cannot write to it?
module.data_layout = engine.target_data

llvm_ir = str(module)
print(">>> LLVM IR ================================")
print(llvm_ir)
print("============================================\n")

opt_llvm_ir = optimize_ir(llvm_ir)

print(">>> OPTIMIZED ==============================")
print(opt_llvm_ir)
print("============================================\n")

mod = compile_ir(engine, opt_llvm_ir)

print(">>> ASM ====================================")
print(target_machine.emit_assembly(llvm.parse_assembly(str(mod))))
print("============================================\n")

# Look up the function pointer (a Python int)
func_ptr = engine.get_function_address("node")

inputs = np.arange(N, dtype=np.float32)
weights = np.arange(N, dtype=np.float32)

# Run the function via ctypes
c_float_p = POINTER(c_float)
cfunc = CFUNCTYPE(c_float, c_float_p, c_float_p, c_int)(func_ptr)
ret = cfunc(inputs.ctypes.data_as(c_float_p), weights.ctypes.data_as(c_float_p), N)
print("  retval =", ret)
print("expected =", (inputs * weights).sum())
assert ret == (inputs * weights).sum()
