from ctypes import CFUNCTYPE, POINTER, c_float, c_int

from llvmlite import ir
import llvmlite.binding as llvm
import numpy as np

N = 64


class LLVM:
    def __init__(self):
        # All these initializations are required for code generation!
        llvm.initialize()
        llvm.initialize_native_target()
        llvm.initialize_native_asmprinter()  # yes, even this one

        # Create a target machine representing the host
        """
        Create an ExecutionEngine suitable for JIT code generation on
        the host CPU.  The engine is reusable for an arbitrary number of
        modules.
        """
        self.target = llvm.Target.from_triple(llvm.get_process_triple())
        self.target_machine = self.target.create_target_machine()
        self.target_machine.set_asm_verbosity(True)
        # And an execution engine with an empty backing module
        backing_mod = llvm.parse_assembly("")
        self.engine = llvm.create_mcjit_compiler(backing_mod, self.target_machine)

        ######## OPTIMIZATION PASS ########
        # from https://blog.christianperone.com/2022/09/tutorial-on-using-llvm-to-jit-pytorch-fx-graphs-to-native-code-x86-arm-risc-v-wasm-part-i-scalars/
        self.pass_manager = llvm.PassManagerBuilder()
        self.pass_manager.opt_level = 3
        self.pass_manager.size_level = 0
        self.pass_manager.loop_vectorize = True
        self.pass_manager.slp_vectorize = True
        self.pass_module = llvm.ModulePassManager()
        self.pass_manager.populate(self.pass_module)
        ###################################

    def get_fptr(self, name):
        # Look up the function pointer (a Python int)
        func_ptr = llvm_manager.engine.get_function_address(name)
        return func_ptr

    def show_asm(self, mod):
        print(self.target_machine.emit_assembly(llvm.parse_assembly(str(mod))))

    def optimize_ir(self, module):
        module.triple = llvm.get_process_triple()
        module.data_layout = self.engine.target_data
        llvm_ir = str(module)
        mod = llvm.parse_assembly(llvm_ir)
        self.pass_module.run(mod)
        return str(mod)

    def compile_ir(self, llvm_ir):
        """
        Compile the LLVM IR string with the given engine.
        The compiled module object is returned.
        """
        # Create a LLVM module object from the IR
        mod = llvm.parse_assembly(llvm_ir)
        mod.verify()

        # Now add the module and make sure it is ready for execution
        self.engine.add_module(mod)
        self.engine.finalize_object()
        self.engine.run_static_constructors()
        return mod


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
