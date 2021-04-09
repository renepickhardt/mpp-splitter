from dstar import Algo, Graph


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
    """Test how long computing all residuals takes
    """
    g = small_graph()
    a = Algo(g, 'S', 'D', 2, 1)
    benchmark(a.compute_residual)


def test_dijkstra(benchmark):
    g = small_graph()
    a = Algo(g, 'S', 'D', 2, 1)
    a.compute_residual()
    flow = benchmark(a.incflow)
    expected = [(1, 'S', 'X'), (1, 'X', 'B'), (1, 'B', 'D')]
    flow = [(f[0], f[1].id, f[2].id) for f in flow]
    assert flow == expected
