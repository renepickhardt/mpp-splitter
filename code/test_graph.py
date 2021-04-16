from dstar import Algo, Graph
import gzip
import json

def small_graph():
    g = Graph()
    g.add_edge("S", "A", capacity=2)
    g.add_edge("S", "X", capacity=1)
    g.add_edge("A", "B", capacity=2)
    g.add_edge("X", "B", capacity=9)
    g.add_edge("X", "Y", capacity=7)
    g.add_edge("B", "D", capacity=4)
    g.add_edge("Y", "D", capacity=4)
    return g


def test_graph_build(benchmark):
    benchmark(small_graph)


def test_global_residual(benchmark):
    """Test how long computing all residuals takes"""
    g = small_graph()
    a = Algo(g, "S", "D", 2, 1)
    benchmark(a.compute_residual)


def test_dijkstra(benchmark):
    g = small_graph()
    a = Algo(g, "S", "D", 2, 1)
    a.compute_residual()
    flow = benchmark(a.incflow)
    expected = [(1, "S", "X"), (1, "X", "B"), (1, "B", "D")]
    f = [(f[0], f[1].id, f[2].id) for f in flow]
    assert f == expected

    a.applyflow(flow)
    a.compute_residual()
    flow = a.incflow()
    f = [(f[0], f[1].id, f[2].id) for f in flow]
    expected = [
        (1, "S", "A"),
        (1, "A", "B"),
        (1, "B", "X"),
        (1, "X", "Y"),
        (1, "Y", "D"),
    ]
    assert f == expected

def test_dijkstra_correct():
    g = small_graph()
    a = Algo(g, "S", "D", 2, 1)
    a.checked_compute_residual()
    flow = a.incflow()
    print(flow)
    expected = [(1, "S", "X"), (1, "X", "B"), (1, "B", "D")]
    f = [(f[0], f[1].id, f[2].id) for f in flow]
    assert f == expected

    a.applyflow(flow)
    a.checked_compute_residual()
    flow = a.incflow()
    print(flow)
    f = [(f[0], f[1].id, f[2].id) for f in flow]
    expected = [
        (1, "S", "A"),
        (1, "A", "B"),
        (1, "B", "X"),
        (1, "X", "Y"),
        (1, "Y", "D"),
    ]
    assert f == expected

def test_ln_topo():
    g = Graph()
    node_ids = []
    amt = 500
    rounds = 500
    with gzip.open("tests/topo-20210416.json.gz") as f:
        listchannels = json.load(f)['channels']

        for c in listchannels:
            node_ids.append(c['source'])
            if c['source'] > c['destination']:
                continue  # only add one direction, the other is going
                          # to be added by add_edge
            g.add_edge(c['source'], c['destination'], float(c['satoshis']/10000))

    # Pick two nodes
    src = '03864ef025fde8fb587d989186ce6a4a186895ee44a926bfc370e2c366597a3f8f'
    dst = '02ad6fb8d693dc1e4569bcedefadf5f72a931ae027dc0f0c544b34c1c6f3b9a02b'

    print(f"source={src} destination={dst}")
    a = Algo(g, src, dst, amt, amt/rounds)
    for i in range(rounds):
        print(f"Iteration {i}/{rounds}")
        a.checked_compute_residual()
        flow = a.incflow()

        res = 1
        for h in flow:
            res *= h[3]

        print(flow, res)
        a.applyflow(flow)

    print(listchannels[0])
