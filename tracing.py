from collections import defaultdict
from ctypes import CFUNCTYPE, c_int
from graphlib import TopologicalSorter
import os
from time import monotonic

from llvmlite import ir
import numpy as np
import numpy as np
from numpy.ctypeslib import ndpointer
from rich import print as pprint

from builder import make_dag
from llvm import LLVM

DEBUG = int(os.getenv("DEBUG", "0"))
SEED = int(os.getenv("SEED", "4"))


def check_ready(subset, slots):
    computed = set()

    requestors = defaultdict(list)
    for node in nodes:
        for pred in subset[node]:
            requestors[pred].append(node + str(slots[node][pred][1]))

    def ready(node, subset):
        return all(type(pred) == int or pred in computed for pred in subset[node])

    print("                      INCOMING  NODE  OUTGOING")
    for node in nodes:
        pprint(
            f"[green]{str(subset[node]):>30}   [violet]{node}   [red]{requestors[node]}"
        )
        assert ready(node, subset)
        computed.add(node)


def show(wiring):
    for neuron, edges in wiring.items():
        print(f'"{neuron}" : { [f"{e:>2}" for e in edges]}')


def i32(i):
    return ir.Constant(ir.IntType(32), i)


def f32(i):
    return ir.Constant(ir.FloatType(), i)


def relu(builder, value):
    cmp = builder.fcmp_ordered("<=", f32(0), value)
    return builder.select(cmp, value, f32(0), name="relu")


# Not all nodes and inputs take part in the computation
# of the required outputs, select only the needed ones
def extract_subset(outputs, wiring):
    seen = set()
    inputs = set()

    def trace(node):
        if type(node) == str:
            if node in seen:
                return
            seen.add(node)
            for pred in wiring[node]:
                trace(pred)
        else:
            inputs.add(node)

    for node in outputs:
        trace(node)
    subset = {node: wiring[node] for node in seen}
    return subset, sorted(list(inputs)), sorted(list(seen))


NNODES = 16
NINPS = 8
NOUTS = 8
MINED = 4
MAXED = 6

st = monotonic()
generator = np.random.default_rng(SEED)
wiring = make_dag(
    n_nodes=NNODES,
    n_inputs=NINPS,
    min_edges=MINED,
    max_edges=MAXED,
    seed=SEED,
)
outputs = list(wiring.keys())[-NOUTS:]
output2ix = {node: i for i, node in enumerate(outputs)}
et = monotonic()
print(f"DAG preparation took: {et - st:.5f}seconds")

show(wiring)

st = monotonic()
subset, leaves, nodes = extract_subset(outputs, wiring)
et = monotonic()
print(f"subset extraction took: {et - st:.5f}seconds")
show(subset)

ts = TopologicalSorter(subset)
topo = list(ts.static_order())

print(f"{outputs=}")
print(f" inputs={leaves}")
print()


def op(x):
    return max(0, x)


sensors = [el for el in topo if type(el) == int]
neurons = [el for el in topo if type(el) == str]

inputs = generator.random(NINPS).astype(np.float32)
weights = [
    generator.normal(size=(len(wiring[neuron]))).astype(np.float32) * 0.1
    for neuron in neurons
]
results = np.zeros(shape=len(outputs), dtype=np.float32)

float_ptr = ir.PointerType(ir.FloatType())
fnty = ir.FunctionType(ir.VoidType(), [float_ptr] * (len(weights) + 2))
mod = ir.Module(name=__file__)
func = ir.Function(mod, fnty, name="node")
builder = ir.IRBuilder()

inputs_ptr = func.args[0]
weights_ptr = func.args[1:-1]
results_ptr = func.args[-1]

bb_entry = func.append_basic_block(name="entry")
builder.position_at_end(bb_entry)

py_cache = {}
ll_cache = {}
for sensor in sensors:
    py_cache[sensor] = inputs[sensor]
    ll_cache[sensor] = builder.load(builder.gep(inputs_ptr, [i32(sensor)]))

assert len(neurons) == len(weights)

for i, neuron in enumerate(neurons):
    py_incoming = [py_cache[val] for val in wiring[neuron]]
    ll_incoming = [ll_cache[val] for val in wiring[neuron]]
    py_ws = weights[i]
    ll_ws = weights_ptr[i]

    py_acc = 0
    for j in range(len(py_ws)):
        py_acc += py_ws[j] * py_incoming[j]
    py_z = op(py_acc)

    ll_acc = f32(0)
    for j in range(len(py_ws)):
        w = builder.load(builder.gep(ll_ws, [i32(j)]))
        x = ll_incoming[j]
        val = builder.fmul(w, x)
        ll_acc = builder.fadd(ll_acc, val)
        # ll_acc = builder.fma(ll_acc, w, x)
    ll_z = relu(builder, ll_acc)

    py_cache[neuron] = py_z
    ll_cache[neuron] = ll_z

py_results = []
for i, out in enumerate(outputs):
    py_results.append(py_cache[out])
    builder.store(ll_cache[out], builder.gep(results_ptr, [i32(i)], name=f"outputs"))

builder.ret_void()

st = monotonic()
llvm_manager = LLVM()

print(">>> LLVM IR ================================")
print(mod)
print("============================================\n")

mod = llvm_manager.optimize_ir(mod)
print(">>> OPTIMIZED ==============================")
print(mod)
print("============================================\n")

mod = llvm_manager.compile_ir(mod)

# print(">>> ASM ====================================")
# llvm_manager.show_asm(mod)
# print("============================================\n")

# Look up the function pointer (a Python int)
func_ptr = llvm_manager.get_fptr("node")


def to_ndptr(arr):
    return ndpointer(dtype=arr.dtype, ndim=arr.ndim, shape=arr.shape)


arr_ptrs = [to_ndptr(inputs)] + [to_ndptr(w) for w in weights] + [to_ndptr(results)]
cfunc = CFUNCTYPE(c_int, *arr_ptrs)(func_ptr)

print("Running...")
st = monotonic()
cfunc(inputs, *weights, results)
et = monotonic()
elapsed = et - st
print(f" RUN: {elapsed: .8f}s")

tot_edges = sum([weight.size for weight in weights])
print(f"TOT EDGES: {tot_edges} ({tot_edges/elapsed/1e6:.2f}M edges/second)")

print(" ".join([f"{val:>6.3f}" for val in results]))
print(" ".join([f"{val:>6.3f}" for val in py_results]))
