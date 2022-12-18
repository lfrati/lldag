from ctypes import CFUNCTYPE, c_float, POINTER, c_int

import llvmlite.binding as llvm

import numpy as np


# All these initializations are required for code generation!
llvm.initialize()
llvm.initialize_native_target()
llvm.initialize_native_asmprinter()  # yes, even this one

from llvmlite import ir

# Create some useful types
float_ptr = ir.PointerType(ir.FloatType())
void = ir.VoidType()
fnty = ir.FunctionType(void, (float_ptr, float_ptr, float_ptr))

const_0 = ir.Constant(ir.FloatType(), 0)

# Create an empty module...
module = ir.Module(name=__file__)

# and declare a function named "fpadd" inside it
func = ir.Function(module, fnty, name="process")

# Now implement the function
block = func.append_basic_block(name="entry")
builder = ir.IRBuilder(block)
xs, ws, out = func.args
a = builder.load(xs)
b = builder.load(ws)
results = builder.fadd(a, b, name="result")
builder.store(results, out)
builder.ret_void()


def create_execution_engine():
    """
    Create an ExecutionEngine suitable for JIT code generation on
    the host CPU.  The engine is reusable for an arbitrary number of
    modules.
    """
    # Create a target machine representing the host
    target = llvm.Target.from_triple(llvm.get_process_triple())
    target_machine = target.create_target_machine()
    # And an execution engine with an empty backing module
    backing_mod = llvm.parse_assembly("")
    engine = llvm.create_mcjit_compiler(backing_mod, target_machine)
    return engine


def compile_ir(engine, llvm_ir):
    """
    Compile the LLVM IR string with the given engine.
    The compiled module object is returned.
    """
    # Create a LLVM module object from the IR
    mod = llvm.parse_assembly(llvm_ir)

    ######## OPTIMIZATION PASS ########
    # from https://blog.christianperone.com/2022/09/tutorial-on-using-llvm-to-jit-pytorch-fx-graphs-to-native-code-x86-arm-risc-v-wasm-part-i-scalars/
    pass_manager = llvm.PassManagerBuilder()
    pass_manager.opt_level = 2
    pass_module = llvm.ModulePassManager()
    pass_manager.populate(pass_module)
    pass_module.run(mod)
    print(">>> OPTIMIZED ==============================")
    print(mod)
    print("============================================\n")
    ###################################

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

mod = compile_ir(engine, llvm_ir)

# Look up the function pointer (a Python int)
func_ptr = engine.get_function_address("process")

inputs = np.arange(40, dtype=np.float32) + 1
weights = np.arange(40, dtype=np.float32) + 2
results = np.zeros(40, dtype=np.float32)

# Run the function via ctypes
c_float_p = POINTER(c_float)
cfunc = CFUNCTYPE(c_int, c_float_p, c_float_p, c_float_p)(func_ptr)
ret = cfunc(
    inputs.ctypes.data_as(c_float_p),
    weights.ctypes.data_as(c_float_p),
    results.ctypes.data_as(c_float_p),
)
print("ret", ret)
print("fpadd(...) =", results[0])
print("expected =", inputs[0] + weights[0])
