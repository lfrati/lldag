from ctypes import CFUNCTYPE, c_int, c_float, POINTER
import sys

try:
    from time import perf_counter as time
except ImportError:
    from time import time

import numpy as np

try:
    import faulthandler

    faulthandler.enable()
except ImportError:
    pass

import llvmlite.ir as ll
import llvmlite.binding as llvm


def i32(i):
    return ll.Constant(ll.IntType(32), i)


def f32(i):
    return ll.Constant(ll.FloatType(), i)


llvm.initialize()
llvm.initialize_native_target()
llvm.initialize_native_asmprinter()

N = 10

t1 = time()

fnty = ll.FunctionType(ll.FloatType(), [ll.ArrayType(ll.FloatType(), N).as_pointer()])
module = ll.Module()

func = ll.Function(module, fnty, name="sum")

bb_entry = func.append_basic_block()

builder = ll.IRBuilder()
builder.position_at_end(bb_entry)
print(func.args[0])
ptr = builder.gep(func.args[0], [i32(0)])
arr = builder.load(ptr)
print(arr)


added = f32(0)
for i in range(N):
    value = builder.extract_value(arr, i)
    added = builder.fadd(added, value)

builder.ret(added)

strmod = str(module)

t2 = time()

print("-- generate IR:", t2 - t1)

t3 = time()

llmod = llvm.parse_assembly(strmod)

t4 = time()

print("-- parse assembly:", t4 - t3)

print(llmod)

pmb = llvm.create_pass_manager_builder()
pmb.opt_level = 2
pm = llvm.create_module_pass_manager()
pmb.populate(pm)

t5 = time()

pm.run(llmod)

t6 = time()

print("-- optimize:", t6 - t5)

t7 = time()

target_machine = llvm.Target.from_default_triple().create_target_machine()


def to_ndptr(arr):
    return np.ctypeslib.ndpointer(
        dtype=arr.dtype, ndim=arr.ndim, shape=arr.shape, flags="C_CONTIGUOUS"
    )


with llvm.create_mcjit_compiler(llmod, target_machine) as ee:
    ee.finalize_object()
    cfptr = ee.get_function_address("sum")

    t8 = time()
    print("-- JIT compile:", t8 - t7)

    print(target_machine.emit_assembly(llmod))

    A = np.arange(N, dtype=np.float32)
    cfunc = CFUNCTYPE(c_float, to_ndptr(A))(cfptr)
    res = cfunc(A)

    print(cfunc.argtypes)
    print(res, A.sum())
