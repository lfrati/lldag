# lldag

Arbitrary sparse DAGs are a pain, you don't get any of the deeplearning gpu speedup and you pay the full price of slow python. Let's change that with LLVM.

```mermaid
graph TD
    Graph -.->|Python| Tape([Tape])
    Tape -.->|LLVMlite| Forward
    Inputs --> Forward[[Forward]]
    Weights --> Forward
    Forward --> Outputs
```
# graph --> forward-pass-able form.

Got some inspiration from:
- [micrograd](https://github.com/karpathy/micrograd)
- [tinygrad](https://github.com/geohot/tinygrad)

# forward-pass-able form --> LLVM
