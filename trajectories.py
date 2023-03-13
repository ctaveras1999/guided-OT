import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import networkx as nx
from simplicial_paths import *

class Trajectory:
    def __init__(self, SC, nodes):
        self.nodes = nodes
        self.path_edges = self.to_edge_sequence(nodes)
        self.path_edge_vec_seq = self.to_edge_vec_seq(SC, nodes)
        self.edge_projections = self.project(SC)

    @staticmethod
    def to_edge_sequence(nodes):
        seq = [(nodes[i], nodes[i+1]) for i in range(len(nodes)-1)]
        return seq
    
    def to_edge_vec_seq(self, SC, nodes):
        path_edges = Trajectory.to_edge_sequence(nodes)
        num_total_edges, num_path_edges = SC.edge_vec.shape[0], len(path_edges) + 1
        path_edge_vec_seq = np.zeros((num_total_edges, num_path_edges))

        for i, edge in enumerate(path_edges):
            e1, e2 = edge
            parity = -2 * int(e1 > e2) + 1
            oriented_edge = (min(edge), max(edge))
            edge_idx = SC.edges.index(oriented_edge)
            path_edge_vec_seq[edge_idx, i+1] = parity

        path_edge_vec_seq = np.cumsum(path_edge_vec_seq, axis=1)
        self.path_edge_vec_seq = path_edge_vec_seq

        return path_edge_vec_seq

    def project(self, SC): # returns cumulative projection (not final projection)
        edge_projections = SC.harm_proj(self.path_edge_vec_seq)
        self.edge_projections = edge_projections
        return edge_projections


class Trajectory_Class:
    def __init__(self, SC):
        self.SC = SC
        self.G = deepcopy(SC.graph)
        self.paths = []

    # generates a path using a sequence of nodes in the graph
    def make_path(self, nodes_to_visit, visit_weight=1e-2): 
        num_nodes_to_visit = len(nodes_to_visit)

        path_nodes = [nodes_to_visit[0]]

        for i in range(num_nodes_to_visit-1):
            start, end = nodes_to_visit[i], nodes_to_visit[i+1]
            interp_path_nodes = nx.dijkstra_path(self.G, start, end)
            interp_path_len = len(interp_path_nodes) - 1

            for j in range(interp_path_len):
                v0, v1 = interp_path_nodes[j], interp_path_nodes[j+1]
                self.G[v0][v1]['weight'] += visit_weight

            path_nodes += interp_path_nodes[1:]
        
        path = Trajectory(self.SC, path_nodes)

        return path

    def generate_paths(self, path_coord_template, num_paths, num_nodes, visit_weight):
        candidate_node_set = Trajectory_Class.nodes_from_coords(self.SC, path_coord_template, num_nodes)
        num_template_coords = len(path_coord_template)

        paths = []

        for i in range(num_paths):
            node_idx = np.random.choice(np.arange(num_nodes), num_template_coords)
            path_template = [candidate_node_set[k, node_idx[k]] for k in range(num_template_coords)]
            path = self.make_path(path_template, visit_weight)
            paths.append(path)
        
        self.paths = paths

    @staticmethod
    def nodes_from_coords(SC, coords, num_nodes=1):
        node_set = []
        
        for i, coord in enumerate(coords):
            node_dists = np.linalg.norm(SC.nodes - coord, axis=1)
            node_order = np.argsort(node_dists)[:num_nodes]
            node_set.append(node_order)
        
        return np.array(node_set)

    def plot_projections(self, color, label, ax=None):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
        for i, path in enumerate(self.paths):
            projections = path.edge_projections
            ax.scatter(projections[:-1,0], projections[:-1,1], s=5, c=color, marker=',')
            if i == 0:
                ax.scatter(projections[-1,0], projections[-1,1], s=250, c=color, marker='*', label=label)
            else:
                ax.scatter(projections[-1,0], projections[-1,1], s=250, c=color, marker='*')

    def plot_paths(self, color, label, ax=None):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            self.SC.plot()
        for path in self.paths:
            for edge in path.path_edges:
                n1, n2 = edge
                line = np.vstack([self.SC.nodes[n1], self.SC.nodes[n2]])
                ax.plot(line[:,0], line[:,1], color=color, linewidth=3)

            start_and_end = np.array([path.nodes[0], path.nodes[-1]])

            ax.scatter(self.SC.nodes[start_and_end, 0], self.SC.nodes[start_and_end, 1], color='black', s=70, marker='o')


class Trajectory_Holder:
    def __init__(self, SC, class_coords, num_paths, class_names, class_color, num_nodes=5, visit_weight=1e-2):
        self.SC = SC
        self.path_classes = []
        self.num_classes = len(class_coords)
        self.class_names = class_names
        self.class_color = class_color
        self.class_coords = class_coords

        for coords in class_coords:
            path_class = Trajectory_Class(self.SC)
            path_class.generate_paths(coords, num_paths, num_nodes, visit_weight)
            self.path_classes.append(path_class)
    
    def plot_projections(self):
        proj_fig = plt.figure()
        proj_ax = proj_fig.add_subplot(1,1,1)
        for i, path_class in enumerate(self.path_classes):
            color = self.class_color[i]
            label = self.class_names[i]
            path_class.plot_projections(color, label, proj_ax)
        proj_ax.legend()

    def plot_paths(self, overlay=False):
        proj_fig = plt.figure()
        proj_ax = proj_fig.add_subplot(1,1,1)
        for i, path_class in enumerate(self.path_classes):
            if overlay:
                self.SC.plot()
            color = self.class_color[i]
            label = self.class_names[i]
            path_class.plot_paths(color, label, proj_ax)
            if not overlay:
                plt.show()
        if overlay:
            plt.show()