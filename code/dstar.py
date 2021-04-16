import networkx as nx
from typing import List
import heapq
import math
log = math.log2

INFTY = 99999


class Node:
    def __init__(self, id):
        self.id = id
        # All the following arrays are indexed using the same
        # position. They are not objects to cluster more items in the
        # same cache-line. We usually update only one array at a time,
        # so loading all elements related to a peer is inefficient.
        self.peers = []
        self.residual = []  # log-probabilities
        self.invresidual = []  # Our position in the peer's list
        self.capacities = []  # Constant
        self.flow = []  # Actual flow we are sending over this edge

        # Information for the D* algorithms
        self.predecessor = None
        self.logprob = INFTY  # Probability of the partial flow through
                            # successor
        self.dist = INFTY  # Distance from the source of the
                         # payment. Used to bias towards the source
                         # when starting from the destination.

        self.idx = None

    def add_peer(self, peer: "Node", capacity: float) -> int:
        """Returns the position of the peer in the list.
        """
        pos = len(self.peers)
        self.peers.append(peer)
        self.residual.append(capacity)
        self.invresidual.append(-1)  # Dummy to be replaced once we
                                     # have added both sides
        self.capacities.append(capacity)
        self.flow.append(0)
        return pos

    def get_peer_index(self, peer):
        return self.peers.index(peer)

    def __lt__(self, other):
        return self.logprob < other.logprob

    def __eq__(self, other):
        return self.id == other.id

    def __repr__(self) -> str:
        return f"Node[id={self.id}]"


class Graph:
    def __init__(self):
        self.nodes = []
        self.node_index = {}

    def add_node(self, id):
        n = Node(id)
        idx = len(self.nodes)
        n.idx = idx
        self.nodes.append(n)
        self.node_index[id] = idx
        return n

    def get_node(self, id):
        idx = self.node_index.get(id, None)
        if idx is None:
            return None
        else:
            return self.nodes[idx]

    def add_edge(self, id1, id2, capacity):
        n1 = self.get_node(id1)
        if n1 is None:
            n1 = self.add_node(id1)
        n2 = self.get_node(id2)
        if n2 is None:
            n2 = self.add_node(id2)

        i1 = n1.add_peer(n2, capacity)
        i2 = n2.add_peer(n1, capacity)

        # Now need to cross-link the residuals
        n1.invresidual[-1] = i2
        n2.invresidual[-1] = i1

    def print(self):
        for n in self.nodes:
            for i, p in enumerate(n.peers):
                print(n, p, n.capacities[i], n.flow[i], n.residual[i])

    def to_nx(self) -> nx.DiGraph:
        g = nx.DiGraph()
        for n in self.nodes:
            for i, p in enumerate(n.peers):
                g.add_edge(n.id, p.id, flow=n.flow[i], residual=n.residual[i])
        return g

    def print_graph(self):
        print("digraph {")
        for n in self.nodes:
            print(f'\tn{n.idx} [label="{n.id}"]')
        for n in self.nodes:
            for i, p in enumerate(n.peers):
                print(f'\tn{n.idx} -> n{p.idx} [residual="{n.residual[i]}", capacity="{n.capacities[i]}", flow="{n.flow[i]}"]')
        print("}")


class Algo:
    def __init__(self, graph: Graph, source, dest, amt: int, stepsize: int = 1) -> None:
        self.graph = graph
        self.olist: List[Node] = []
        self.source = self.graph.get_node(source)
        self.dest = self.graph.get_node(dest)
        self.stepsize = stepsize

    def _compute_residual_edge(self, n, i, p):
        amt = self.stepsize

        # Classical reach-around...
        rev_id = n.invresidual[i]
        assert(n == n.peers[i].peers[rev_id])

        cap = n.capacities[i]
        invcap = n.peers[i].capacities[rev_id]

        flow = n.flow[i]
        invflow = n.peers[i].flow[rev_id]

        diff = min(flow, invflow)
        if diff > 0:
            n.flow[i] -= diff
            n.peers[i].flow[rev_id] -= diff
            invflow -= diff
            flow -= diff

        # No flow at all yet, just compute the base case
        elif flow == 0 and invflow == 0:
            lprob = -log((cap + 1 - amt)/(cap + 1))
            assert(n.residual[i] >= lprob)
            n.residual[i] = lprob

        elif flow > 0:
            # We could add the amount in this direction.
            if flow + amt > cap:
                lprob = INFTY
            else:
                lprob = -log((cap + 1 - flow - amt) / (cap - flow + 1))
            assert(n.residual[i] >= lprob)
            n.residual[i] = lprob

            # If we already have a flow in this direction we
            # could also remove it and add in the opposite
            # direction
            if amt < flow:
                lprob = -log((cap + 1 - flow + amt) / (cap + 1 - flow))
            else:
                lprob = INFTY
            assert(n.peers[i].residual[rev_id] >= lprob)
            n.peers[i].residual[rev_id] = lprob

        elif invflow > 0:
            return
            # We could add the amount in the opposite direction
            if invflow + amt > invcap:
                lprob = INFTY
            else:
                lprob = -log((invcap + 1 - invflow - amt) / (invcap - invflow + 1))
            assert(n.peers[i].residual[rev_id] >= lprob)
            n.peers[i].residual[rev_id] = lprob

            if amt < invflow:
                # 1/ P(X>=f | X >= f - amt) = (c+1-f)/(c+1-f+a)
                lprob = -log((invcap + 1 - invflow + amt)/(invcap + 1 - invflow))
            else:
                lprob = INFTY
            assert(n.residual[i] >= lprob)
            n.residual[i] = lprob

    def checked_compute_residual(self):
        """Compute residual graph, but check how much changed.
        """
        prev = [None] * len(self.graph.nodes)
        for j, n in enumerate(self.graph.nodes):
            prev[j] = [r for r in n.residual]

        res = self.compute_residual()

        diff = 0
        count = 0
        for j, n in enumerate(self.graph.nodes):
            for i, r in enumerate(n.residual):
                count += 1
                if prev[j][i] != r:
                    diff += 1

        print(f"Residual check: {diff}/{count} changed since last round")
        return res

    def compute_residual(self):
        """Compute the residual graph based on the estimated capacities and
        the stepsize. Not to be used in the loop since it iterates
        over all edges.

        """
        # Reset:
        prev = []
        for j, n in enumerate(self.graph.nodes):
            prev.append([])
            for i, p in enumerate(n.peers):
                n.residual[i] = INFTY
                prev[j].append(n.residual[i])

        amt = self.stepsize
        for j, n in enumerate(self.graph.nodes):
            for i, p in enumerate(n.peers):
                self._compute_residual_edge(n, i, p)



    def incflow(self):
        """Compute an incremental flow using stepsize and the residual graph.
        """
        dest = self.dest
        src = self.source
        dest.logprob = 0
        dest.predecessor = None
        olist = [dest]

        while len(olist) > 0:
            n = heapq.heappop(olist)
            for i, p in enumerate(n.peers):
                logprob = n.logprob + n.residual[i]
                #print("Considering", n, p, logprob, p.logprob)
                if p.logprob > logprob:
                    p.predecessor = n
                    p.logprob = logprob
                    heapq.heappush(olist, p)
                    #print("Taken")

        c = src
        flow = []
        while c.predecessor is not None:
            flow.append((self.stepsize, c, c.predecessor))
            c = c.predecessor

        return flow

    def applyflow(self, flow):
        """
        """
        for f in flow:
            a, b = f[1], f[2]
            i = a.get_peer_index(b)
            a.flow[i] += f[0]  # Flow amount added on this edge
            # TODO Update the residual a -> b and b -> a
        for n in self.graph.nodes:
            n.logprob = INFTY
