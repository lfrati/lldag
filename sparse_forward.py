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

    ts = TopologicalSorter(subset)
    topo = list(ts.static_order())
    sensors = [el for el in topo if type(el) == int]
    neurons = [el for el in topo if type(el) == str]

    return subset, sensors, neurons


def op(x):
    # return max(0, x)
    return np.sin(x)


def forward(inputs, sensors, neurons, wiring, weights, outputs):
    py_cache = {}
    for sensor in sensors:
        py_cache[sensor] = inputs[sensor]

    for i, neuron in enumerate(neurons):
        py_incoming = np.array([py_cache[val] for val in wiring[neuron]])
        py_ws = weights[i]

        py_z = op((py_ws * py_incoming).sum())

        py_cache[neuron] = py_z

    py_results = []
    for i, out in enumerate(outputs):
        py_results.append(py_cache[out])

    return py_results


def build(sensors, neurons, outputs, subset):

    float_ptr = ir.PointerType(ir.FloatType())
    fnty = ir.FunctionType(ir.VoidType(), [float_ptr] * (len(neurons) + 2))
    mod = ir.Module(name=__file__)
    func = ir.Function(mod, fnty, name="forward")
    builder = ir.IRBuilder()

    inputs_ptr = func.args[0]
    weights_ptr = func.args[1:-1]
    results_ptr = func.args[-1]

    bb_entry = func.append_basic_block(name="entry")
    # sin = mod.declare_intrinsic("llvm.sin", [ir.FloatType()])
    builder.position_at_end(bb_entry)

    cache = {}
    for sensor in sensors:
        cache[sensor] = builder.load(builder.gep(inputs_ptr, [i32(sensor)]))

    for i, neuron in enumerate(neurons):
        incoming = [cache[val] for val in subset[neuron]]
        ws = weights_ptr[i]

        acc = f32(0)
        for j in range(len(subset[neuron])):
            w = builder.load(builder.gep(ws, [i32(j)]))
            x = incoming[j]
            acc = builder.fma(x, w, acc)
        z = relu(builder, acc)
        # z = builder.call(sin, [acc])

        cache[neuron] = z

    for i, out in enumerate(outputs):
        builder.store(cache[out], builder.gep(results_ptr, [i32(i)], name=f"out"))

    builder.ret_void()

    llvm_manager = LLVM()
    mod = llvm_manager.optimize_ir(mod)
    mod = llvm_manager.compile_ir(mod)
    # Look up the function pointer (a Python int)
    func_ptr = llvm_manager.get_fptr("forward")

    def ndptr(dtype, ndim, shape):
        return ndpointer(dtype=dtype, ndim=ndim, shape=shape)

    arr_ptrs = (
        [ndptr(dtype=np.float32, ndim=1, shape=len(sensors))]
        + [ndptr(dtype=np.float32, ndim=1, shape=len(subset[node])) for node in neurons]
        + [ndptr(dtype=np.float32, ndim=1, shape=len(outputs))]
    )
    cfunc = CFUNCTYPE(c_int, *arr_ptrs)(func_ptr)

    # TODO: if I don't keep this reference around I get segmentation fault
    # what's the proper way to handle it?
    cfunc.llvm_manager = llvm_manager

    return cfunc


def main():

    NNODES = 256
    NINPS = 32
    NOUTS = 16
    MINED = 4
    MAXED = 16

    generator = np.random.default_rng(SEED)
    wiring = make_dag(
        n_nodes=NNODES,
        n_inputs=NINPS,
        min_edges=MINED,
        max_edges=MAXED,
        seed=SEED,
    )
    outputs = list(wiring.keys())[-NOUTS:]

    all_weights = {
        node: generator.normal(size=(len(wiring[node]))).astype(np.float32) * 0.1
        for node in wiring
    }

    subset, sensors, neurons = extract_subset(outputs, wiring)

    cfunc = build(sensors, neurons, outputs, subset)

    inputs = generator.random(NINPS).astype(np.float32)
    weights = [all_weights[neuron] for neuron in neurons]
    results = np.zeros(shape=len(outputs), dtype=np.float32)

    tot_edges = sum([weight.size for weight in weights])
    print(f"Network size: {tot_edges}")

    CNT = 32

    st = monotonic()
    for _ in range(CNT):
        cfunc(inputs, *weights, results)
    et = monotonic()
    ll_elapsed = (et - st) / CNT
    print(f"llvm: {et - st:.6f}s ({tot_edges/ll_elapsed/1e6:.2f}M edges/second)")

    py_results = []
    st = monotonic()
    for _ in range(CNT):
        py_results = forward(inputs, sensors, neurons, wiring, weights, outputs)
    et = monotonic()
    py_elapsed = (et - st) / CNT
    print(f"  py: {et - st:.6f}s ({tot_edges/py_elapsed/1e6:.2f}M edges/second)")

    # print(" ".join([f"{val:>6.3f}" for val in results]))
    assert np.allclose(
        results, py_results
    ), f"MISMATCH: {results.sum()} != {np.sum(py_results)}"
    pprint("[yellow]MATCH: OK")

    speedup = (ll_elapsed - py_elapsed) / py_elapsed

    faster = "[green]faster" if speedup < 0 else "[red]slower"
    pprint(f"LLVM vs PYTHON: llvm is {speedup * 100:.1f}% {faster}")


if __name__ == "__main__":
    main()
