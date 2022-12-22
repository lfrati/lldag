from graphviz import Digraph, Source


def trace(root):
    nodes, edges = set(), set()

    def _trace(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                _trace(child)

    _trace(root)
    return nodes, edges


from collections import defaultdict

COLORS = defaultdict(lambda: "white")
COLORS["Input"] = "#80ff8080"
# COLORS["Value"] = "#EB867F80"
COLORS["Weight"] = "#FFFF8080"
COLORS["Const"] = "#C27EED80"


def draw_dot(root, format="svg", rankdir="TB"):
    """
    format: png | svg | ...
    rankdir: TB (top to bottom graph) | LR (left to right)
    """
    assert rankdir in ["LR", "TB"]
    nodes, edges = trace(root)
    dot = Digraph(format=format, graph_attr={"rankdir": rankdir})

    for n in nodes:
        dot.node(
            name=str(id(n)),
            label="{ %s }" % (n.name),
            shape="record",
            style="filled",
            fillcolor=COLORS[n.__class__.__name__],
        )
        if n._op:
            dot.node(
                name=str(id(n)) + n._op,
                label=n._op,
            )
            dot.edge(str(id(n)) + n._op, str(id(n)))

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    s = Source(str(dot), filename="gout.gv", format="svg")
    s.view()
