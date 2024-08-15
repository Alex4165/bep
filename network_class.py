from typing import List

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt


class Network:
    def __init__(self):
        self.adj_matrix = None
        self.graph = None
        self.size = 0

    def gen_random(self, size, p):
        """Generate an Erdos-Renyi random graph with connection probability p."""
        self.graph = nx.erdos_renyi_graph(size, p)
        self.adj_matrix = nx.adjacency_matrix(self.graph).toarray()
        self.size = self.graph.number_of_nodes()

    def gen_small_world(self, size, k, p):
        """Generate a Watts-Strogatz random graph.
        k is the max distance of nearest neighbors first connected to.
        p is the rewiring probability."""
        self.graph = nx.watts_strogatz_graph(size, k, p)
        self.adj_matrix = nx.adjacency_matrix(self.graph).toarray()
        self.size = self.graph.number_of_nodes()

    def gen_community(self, sizes: List[int], probabilities: List[List[float]] = None):
        """Generate a community graph.
        sizes is a list of the size of each community.
        probabilities is a list of lists of probabilities such that (r,s) gives the
        probability of community r attaching to s. If None, random values are taken."""
        if probabilities is None:
            probabilities = [[np.random.rand() for s in range(len(sizes))] for r in range(len(sizes))]
        self.graph = nx.stochastic_block_model(sizes=sizes, p=probabilities)
        self.adj_matrix = nx.adjacency_matrix(self.graph).toarray()
        self.size = self.graph.number_of_nodes()

    def gen_hub(self, size, minimum_deg):
        """Generate Barabasi-Albert graph which attaches new nodes preferentially to high degree nodes.
        Creates a hub structure with predefined minimum node degree."""
        self.graph = nx.barabasi_albert_graph(size, minimum_deg)
        self.adj_matrix = nx.adjacency_matrix(self.graph).toarray()
        self.size = self.graph.number_of_nodes()

    def plot_graph(self, title=" ", weighted=False, saving=False):
        """Plot graph"""
        fig, ax = plt.subplots()
        ax.spines[['top', 'right', 'left', 'bottom']].set_visible(False)
        pos = nx.spring_layout(self.graph, k=1 / self.size ** 0.1)
        if weighted:
            for u, v, d in self.graph.edges(data=True):
                d['weight'] = self.adj_matrix[u, v]

            edges, weights = zip(*nx.get_edge_attributes(self.graph, 'weight').items())

            nx.draw(self.graph, pos, node_color='b', edgelist=edges, edge_color=weights,
                    width=3, edge_cmap=plt.cm.Blues)
        else:
            nx.draw_networkx(self.graph, pos, with_labels=False, ax=ax)
        if not saving:
            plt.title(title)
        else:
            plt.savefig("data/"+title+".pdf")
        plt.show()

    def randomize_weights(self, factor=lambda x: 2*x-1, symmetric=True):
        """Reweight the network with factor a function of a uniform random variable between 0 and 1."""
        if self.adj_matrix is not None:
            random_adjustments = factor(np.random.random_sample((self.size, self.size)))
            if symmetric:
                random_adjustments = (random_adjustments + random_adjustments.T)/2
            self.adj_matrix = np.multiply(self.adj_matrix, random_adjustments)
        else:
            raise ValueError("No network to speak of!")

    def exp_weights(self, scale=1, symmetric=True):
        """Reweight the network with exponentially distributed weights."""
        if self.adj_matrix is not None:
            random_adjustments = np.random.exponential(scale, (self.size, self.size))
            if symmetric:
                random_adjustments = (random_adjustments + random_adjustments.T)/2
            self.adj_matrix = np.multiply(self.adj_matrix, random_adjustments)
        else:
            raise ValueError("No network to speak of!")