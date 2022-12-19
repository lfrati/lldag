import llvmlite.binding as llvm


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
        func_ptr = self.engine.get_function_address(name)
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
