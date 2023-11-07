import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.style
from util.simplicial_paths import *

class Trajectory:
    def __init__(self, SC, nodes, extra=0):
        self.extra = extra
        self.nodes = nodes
        self.path_coords = [SC.nodes[node] for node in self.nodes]
        self.path_edges = self.to_edge_sequence(nodes)
        self.path_edge_vec_seq = self.to_edge_vec_seq(SC, nodes)
        self.path_length = self.get_path_length(SC)
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
        if self.extra > 0:
            edge_projections = SC.extra_proj(self.path_edge_vec_seq, self.extra)
        else:
            edge_projections = SC.harm_proj(self.path_edge_vec_seq)
        self.edge_projections = edge_projections
        return edge_projections

    def get_path_length(self, SC):
        edge_lens = np.array([SC.graph[u][v]['weight'] for (u,v) in self.path_edges])
        total_len = np.cumsum(edge_lens)
        return total_len

    def __repr__(self) -> str:
        res = "-".join([str(node) for node in self.nodes])
        return res
    
    def __sub__(self, other):
        x = self.path_edge_vec_seq[:,-1]
        y = other.path_edge_vec_seq[:,-1]
        z = x - y
        z_projs = self.edge_projections[-1] - other.edge_projections[-1]
        
"""
Make a class that allows me to plot chains. Trajectories should inherit from 
chains with the added orded structure.
"""
class Trajectory_Class:
    def __init__(self, SC, extra=False):
        self.extra = extra
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
        
        path = Trajectory(self.SC, path_nodes, self.extra)

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

    def add_path(self,path):
        self.paths.append(path)

    @staticmethod
    def nodes_from_coords(SC, coords, num_nodes=1):
        node_set = []
        
        for i, coord in enumerate(coords):
            node_dists = np.linalg.norm(SC.nodes - coord, axis=1)
            node_order = np.argsort(node_dists)[:num_nodes]
            node_set.append(node_order)
        
        return np.array(node_set)
    
    """
    TODO: 
    Allow plotting of projections that have number of holes other than 2
    """
    def plot_projections(self, color, label, proj_axes=(0,1), ax=None, arrow_width = 0.009, star_size = 300,  origin_size = 250, figsize=(6,6)):
        matplotlib.style.use('ggplot')
        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(1,1,1)
        for i, path in enumerate(self.paths):
            projections = path.edge_projections
            plot_projs = np.zeros((projections.shape[0], 2))
            num_holes = projections.shape[1]
            if num_holes == 1:
                plot_projs[:,0] = projections
            if num_holes == 2:
                plot_projs = projections
            if num_holes > 2:
                plot_projs[:,0] = projections[:,proj_axes[0]]
                plot_projs[:,1] = projections[:,proj_axes[1]]

            for j in range(plot_projs.shape[0]-1):
                curr_proj, next_proj = plot_projs[j], plot_projs[j+1]
                curr_x, curr_y = curr_proj
                next_x, next_y = next_proj
                dx, dy = next_x - curr_x, next_y - curr_y
                ax.arrow(curr_x, curr_y, dx, dy, color=color, width=arrow_width, length_includes_head=True)
            ax.scatter(plot_projs[-1,0], plot_projs[-1,1], s=star_size, c=color, marker='*')#, label=label)
        ax.scatter([0], [0], s=origin_size, c='black', marker='.')#, label=label)
            
    def plot_paths(self, color, label, ax=None, figsize=(6,6)):
        matplotlib.style.use('default')
        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(1,1,1)
            self.SC.plot()
        for path in self.paths:
            for edge in path.path_edges:
                n1, n2 = edge
                x1, y1 = self.SC.nodes[n1]
                x2, y2 = self.SC.nodes[n2]
                dx, dy = x2 - x1, y2 - y1
                ax.arrow(x1, y1, dx, dy, color=color, width=0.01, length_includes_head=True)

            start_and_end = np.array([path.nodes[0], path.nodes[-1]])

            ax.scatter(self.SC.nodes[start_and_end, 0], self.SC.nodes[start_and_end, 1], color='black', s=50, marker='*')


class Trajectory_Holder:
    def __init__(self, SC, def_class_coords=[], num_paths=0, def_class_names=[], def_class_colors=[], num_nodes=5, visit_weight=1e-2):
        self.SC = SC
        self.path_classes = []
        self.num_classes = len(def_class_coords)
        self.class_names = def_class_names[:]
        self.class_colors = def_class_colors[:]
        self.class_coords = def_class_coords[:]

        if isinstance(def_class_colors, list) and (def_class_colors != []):
            for coords in def_class_coords:
                if (coords == []) or (coords is None):
                    continue
                path_class = Trajectory_Class(self.SC)
                path_class.generate_paths(coords, num_paths, num_nodes, visit_weight)
                self.path_classes.append(path_class)
    
    def add_class(self, path_class, base_coords, class_name, class_color):
        self.class_colors.append(class_color)
        self.path_classes.append(path_class)
        self.class_coords.append(base_coords)
        self.class_names.append(class_name)
        self.num_classes += 1

    """
    TODO: 
    Allow plotting of projections that have number of holes other than 2
    """
    def plot_projections(self, proj_axes=(0,1), ax=None, arrow_width = 0.009, star_size = 300,  origin_size = 250, figsize=(6,6)):
        matplotlib.style.use('ggplot')
        if ax is None:
            proj_fig = plt.figure(figsize=figsize)
            proj_ax = proj_fig.add_subplot(1,1,1)
        else:
            proj_ax = ax
        for i, path_class in enumerate(self.path_classes):
            color, label = self.class_colors[i], self.class_names[i]
            path_class.plot_projections(color, label, proj_axes=proj_axes, ax=proj_ax, arrow_width=arrow_width, star_size=star_size, origin_size=origin_size)
        #proj_ax.legend()

    def plot_paths(self, overlay=False, figsize=(6,6), fname=None):
        matplotlib.style.use('default')
        if overlay:
            proj_fig = plt.figure(figsize=figsize)
            proj_ax = proj_fig.add_subplot(1,1,1)
            self.SC.plot()
        else:
            proj_ax = None

        for i, path_class in enumerate(self.path_classes):
            color = self.class_colors[i]
            label = self.class_names[i]
            path_class.plot_paths(color, label, proj_ax, figsize=figsize)
            if not overlay:
                if fname is None:
                    plt.show()
                else:
                    plt.savefig(fname, bbox_inches='tight', format='png', pad_inches=0) 
        if overlay:
            if fname is None:
                plt.show()
            else:
                plt.savefig(fname, bbox_inches='tight', format='png', pad_inches=0) 

def path_from_coords(SC, key_pts, color='red', show_plot=False):
    num_paths, num_nodes = 1, 1
    path_class = Trajectory_Class(SC)
    path_class.generate_paths(key_pts, num_paths, num_nodes, 0)
    path = path_class.paths[0]
    path_proj = path.edge_projections[-1]
    if show_plot:
        path_class.plot_paths(color, 'reference')
    start, end = path.nodes[0], path.nodes[-1]
    return path, path_class, path_proj, (start, end)
