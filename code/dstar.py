import networkx as nx
from typing import List
import heapq
import math
log = math.log2

INFTY = None


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
        self.edge = None  # Best edge to use to get from predecessor
                          # to this node

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

    def print_graph(self, nodes=True, nonflowedges=True):
        print("digraph {")
        if nodes:
            for n in self.nodes:
                print(f'\tn{n.id} [label="{n.id}"]')
        for n in self.nodes:
            for i, p in enumerate(n.peers):
                if n.flow[i] == 0 and not nonflowedges:
                    continue
                print(f'\tn{n.id} -> n{p.id} [residual="{n.residual[i]}", capacity="{n.capacities[i]}", flow="{n.flow[i]}"]')
        print("}")


class Algo:
    def __init__(self, graph: Graph, source, dest, amt: int, stepsize: int = 1) -> None:
        self.graph = graph
        self.source = self.graph.get_node(source)
        self.dest = self.graph.get_node(dest)
        self.olist: List[Node] = [self.source]
        self.stepsize = stepsize

    def _compute_residual_edge(self, n, i, p):
        """Compute the residual flow for a single edge. The edge considered is
        the outgoing edge from node `n` to `p`, with index `i`
        (pointing into the `n.peers` list).

        """
        amt = self.stepsize

        # Classical reach-around...
        rev_id = n.invresidual[i]

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
            n.residual[i] = lprob

        elif flow > 0:
            # We could add the amount in this direction.
            if flow + amt > cap:
                lprob = INFTY
            else:
                lprob = -log((cap + 1 - flow - amt) / (cap - flow + 1))
            n.residual[i] = lprob

            # If we already have a flow in this direction we
            # could also remove it and add in the opposite
            # direction
            if amt <= flow:
                lprob = -log((cap + 1 - flow + amt) / (cap + 1 - flow))
            else:
                lprob = INFTY
            n.peers[i].residual[rev_id] = lprob

        elif invflow > 0:
            # We could add the amount in the opposite direction
            if invflow + amt > invcap:
                lprob = INFTY
            else:
                lprob = -log((invcap + 1 - invflow - amt) / (invcap - invflow + 1))
            n.peers[i].residual[rev_id] = lprob

            if amt <= invflow:
                # 1/ P(X>=f | X >= f - amt) = (c+1-f)/(c+1-f+a)
                lprob = -log((invcap + 1 - invflow + amt)/(invcap + 1 - invflow))
            else:
                lprob = INFTY
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
        self.source.logprob = 0
        self.source.predecessor = None
        considered = 0

        visited = [0] * len(self.graph.nodes)
        while len(self.olist) > 0:
            considered += 1
            n = heapq.heappop(self.olist)
            visited[n.idx] += 1

            if visited[n.idx] > 1000:
                h = [i for i, v in enumerate(visited) if v >= 1]
                print("High visit count:", h)
                for i in h:
                    print(self.graph.nodes[i].id)
                self.graph.print_graph()

                raise ValueError

            # If we made it into olist we must be reachable and we
            # must have a logprob.
            assert(n.logprob is not None)

            #print("incflow", n, n.logprob, len(olist))
            for i, p in enumerate(n.peers):

                # Only consider this peer if we can reach it over an
                # edge that is not infinitely expensive, i.e., the
                # edge should not be considered.
                if n.residual[i] == INFTY:
                    continue

                logprob = n.logprob + n.residual[i]
                if p.logprob is None or p.logprob > logprob:
                    p.predecessor = n
                    p.edge = i
                    p.logprob = logprob
                    heapq.heappush(self.olist, p)

        c = self.dest
        flow = []
        while c.predecessor is not None:
            assert(c.predecessor.peers[c.edge] == c)
            flow.append((self.stepsize, c.predecessor, c, c.edge))
            c = c.predecessor

        print(f"We considered {considered} nodes")
        return flow[::-1]

    def applyflow(self, flow):
        """
        """
        for f in flow:
            a, b = f[1], f[2]
            a.flow[b.edge] += f[0]  # Flow amount added on this edge

        # Recompute the residual for the edges we just changed with the flow:
        for f in flow:
            a = f[1]
            b = f[2]
            forward = (f[1], f[2].edge, a, b)

            #print("Updating residual on edge", forward)
            self._compute_residual_edge(a, f[2].edge, b)

        for f in flow:
            # Set f[2] to INFTY so we'll recompute it
            f[2].logprob = INFTY
            f[2].predecessor = None

        updated = {}
        for f in flow:
            for p in f[2].peers:
                if p.logprob == INFTY:
                    continue
                updated[p.id] = p

        for n in updated.values():
            heapq.heappush(self.olist, n)
