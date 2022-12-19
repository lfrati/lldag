import numpy as np
import numba


@numba.njit("f8(f8[:],f8[:])")
def dummy(inputs, weights):
    s = 0
    s += inputs[0] * weights[0]
    s += inputs[1] * weights[1]
    s += inputs[2] * weights[2]
    # v = max(0, s)
    v = np.sin(s)
    return v


xs = np.random.rand(3)
ws = np.random.rand(3)


dummy(xs, ws)

# for v in dir(dummy):
#     print(f"{v}: {getattr(dummy, v)}")
#     print("---------------")


# inspect_asm
# inspect_cfg
# inspect_disasm_cfg
# inspect_llvm
# inspect_types

# get_annotation_info
# get_call_template
# get_compile_result
# get_function_type
# get_metadata
# get_overload

## What do these track? Runtime performance?
# _cache
# _cache_hits
# _cache_misses

# dummy.get_annotation_info()
# dummy.inspect_llvm().keys()
# for v in dummy.inspect_cfg().values():
for v in dummy.inspect_llvm().values():
    # for v in dummy.inspect_asm().values():
    print(v)
