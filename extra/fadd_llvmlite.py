from ctypes import CFUNCTYPE, c_double

import llvmlite.binding as llvm


# All these initializations are required for code generation!
llvm.initialize()
llvm.initialize_native_target()
llvm.initialize_native_asmprinter()  # yes, even this one

# llvm_ir = """
#    ; ModuleID = "examples/ir_fpadd.py"
#    target triple = "unknown-unknown-unknown"
#    target datalayout = ""
#
#    define double @"fpadd"(double %".1", double %".2")
#    {
#    entry:
#      %"res" = fadd double %".1", %".2"
#      ret double %"res"
#    }
#    """

from llvmlite import ir

# Create some useful types
double = ir.DoubleType()
fnty = ir.FunctionType(double, (double, double))

# Create an empty module...
module = ir.Module(name=__file__)
# and declare a function named "fpadd" inside it
func = ir.Function(module, fnty, name="fpadd")

# Now implement the function
block = func.append_basic_block(name="entry")
builder = ir.IRBuilder(block)
a, b = func.args
result = builder.fadd(a, b, name="res")
builder.ret(result)


def create_execution_engine():
    """
    Create an ExecutionEngine suitable for JIT code generation on
    the host CPU.  The engine is reusable for an arbitrary number of
    modules.
    """
    # Create a target machine representing the host
    target = llvm.Target.from_default_triple()
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
    pass_manager.opt_level = 3
    pass_module = llvm.ModulePassManager()
    pass_manager.populate(pass_module)
    pass_module.run(mod)
    print("OPTIMIZED ==============================")
    print(mod)
    print("========================================")
    ###################################

    mod.verify()
    # Now add the module and make sure it is ready for execution
    engine.add_module(mod)
    engine.finalize_object()
    engine.run_static_constructors()
    return mod


llvm_ir = str(module)
print("LLVM IR ================================")
print(llvm_ir)
print("========================================")

engine = create_execution_engine()
mod = compile_ir(engine, llvm_ir)

# Look up the function pointer (a Python int)
func_ptr = engine.get_function_address("fpadd")

# Run the function via ctypes
cfunc = CFUNCTYPE(c_double, c_double, c_double)(func_ptr)
res = cfunc(1.0, 3.5)
print("fpadd(...) =", res)
