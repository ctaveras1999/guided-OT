import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import networkx as nx

def generate_pts(mode, num_pts=10):
    if mode == 0:
        nx, ny = num_pts, num_pts
        x = np.linspace(-1, 1, nx)
        y = np.linspace(-1, 1, ny)
        xv, yv = np.meshgrid(x, y)
        xv, yv = xv.ravel(), yv.ravel()
        pts = np.array([(xv[i], yv[i]) for i in range(xv.shape[0])])
    else:
        pts = 2 * np.random.random((num_pts ** 2, 2)) - 1
    return pts

def simp_from_pts(pts):
    tri = Delaunay(pts).simplices
    faces = [x for x in tri if len(set(x)) == 3] # face has 3 distinct nodes
    faces = [tuple(sorted(x)) for x in faces]    # sort node order 

    edge_idx_dict, edge_dict = {}, {}
    num_edge = 0

    for face in faces:
        n1, n2, n3 = face
        face_edges = [(n1, n2), (n1, n3), (n2, n3)]

        for edge in face_edges:
            if edge not in edge_dict:
                edge_dict[edge] = num_edge
                edge_idx_dict[num_edge] = edge
                num_edge += 1

    edges = [edge_idx_dict[i] for i in range(num_edge)]

    edges.sort(key=lambda x:x[0])
    faces.sort(key=lambda x:x[0])

    return edges, faces
    
class SimplicialComplex:
    def __init__(self, pts, weighted_edges=False):
        self.nodes = pts
        self.edges, self.faces = simp_from_pts(pts)
        num_nodes, num_edges, num_faces = len(pts), len(self.edges), len(self.faces)
        self.num_nodes, self.num_edges, self.num_faces = num_nodes, num_edges, num_faces
        self.node_vec = np.zeros(num_nodes, dtype=int)
        self.edge_vec = np.zeros(num_edges, dtype=int)
        self.face_vec = np.zeros(num_faces, dtype=int)

        self.is_weighted = weighted_edges

        self.B0_main, self.B1_main = self.make_incidence_matrices()

    def make_incidence_matrices(self):
        B0 = np.zeros((self.num_nodes, self.num_edges))
        B1 = np.zeros((self.num_edges, self.num_faces))

        for i, edge in enumerate(self.edges):
            v1, v2 = edge
            B0[v1, i] = -1
            B0[v2, i] = 1
        
        for i, face in enumerate(self.faces):
            v1, v2, v3 = face
            e1, e2, e3 = (v1, v2), (v1, v3), (v2, v3)
            e1_idx, e2_idx, e3_idx = self.edges.index(e1), self.edges.index(e2), self.edges.index(e3)
            B1[e1_idx, i] = 1
            B1[e2_idx, i] = -1
            B1[e3_idx, i] = 1

        return B0, B1

    def add_simplices(self, simp_set):
        """
        simplices: dictionary of simplices to be added.
        dictionary keys can be 0, 1, and/or 2
        e.g. simplicies[0] = [1, 3, 4] 
        (set containing 1st, 3rd and 4th node in mesh) 

        Want to add simplices in such a way that simplicial complex structure is 
        maintained. Therefore, if we add a face, we must ensure to add its 
        edges and nodes if they aren't already in the complex.
        """
        simp_dict = {0:set(), 1:set(), 2:set()}
        
        for i in range(3):
            if i in simp_set:
                simp_dict[i] = simp_dict[i].union(simp_set[i])

        # Add faces
        for face in simp_dict[2]:
            n1, n2, n3 = self.faces[face]
            self.face_vec[face] = 1

            e1, e2, e3 = (n1, n2), (n1, n3), (n2, n3)

            e1_idx, e2_idx, e3_idx = self.edges.index(e1), self.edges.index(e2), self.edges.index(e3)

            nodes_needed, edges_needed = set([n1, n2, n3]), set([e1_idx, e2_idx, e3_idx])

            simp_dict[0] = simp_dict[0].union(nodes_needed)
            simp_dict[1] = simp_dict[1].union(edges_needed)

        # Add edges
        for edge in simp_dict[1]:
            self.edge_vec[edge] = 1
            n1, n2 = self.edges[edge]

            nodes_needed = set([n1, n2])
            simp_dict[0] = simp_dict[0].union(nodes_needed)
            
        # Add nodes
        for node in simp_dict[0]:
            self.node_vec[node] = 1

        self.make_graph(self.is_weighted)
        self.compute_proj_matrix()

    def remove_simplices(self, simp_set):
        simp_dict = {0:set(), 1:set(), 2:set()}
        
        for i in range(3):
            if i in simp_set:
                simp_dict[i] = simp_dict[i].union(simp_set[i])

        for node in simp_dict[0]:
            self.node_vec[node] = 0

            for i, edge in enumerate(self.edges):
                if node in edge:
                    simp_dict[1] = simp_dict[1].union(set([i]))
                
            for i, face in enumerate(self.faces):
                if node in face:
                    simp_dict[2] = simp_dict[2].union(set([i]))

        for edge in simp_dict[1]:
            self.edge_vec[edge] = 0
            n1, n2 = self.edges[edge]
            for i, face in enumerate(self.faces):
                if (n1 in face) and (n2 in face):
                    simp_dict[2] = simp_dict[2].union(set([i]))
    
        for face in simp_dict[2]:
            self.face_vec[face] = 0
        
        self.make_graph(self.is_weighted)
        self.compute_proj_matrix()
        
    def get_incidence_matrices(self):
        B0 = self.B0_main[self.node_vec.astype(bool),:]
        B0 = B0[:,self.edge_vec.astype(bool)]
        B1 = self.B1_main[self.edge_vec.astype(bool),:]
        B1 = B1[:,self.face_vec.astype(bool)]
        self.B0, self.B1 = B0, B1
        return B0, B1

    def make_graph(self, weighted=True):
        G = nx.Graph()

        for i, edge in enumerate(self.edges):
            if self.edge_vec[i]:
                u, v = edge
                weight = np.linalg.norm(self.nodes[u] - self.nodes[v]) if weighted else 1
                G.add_edge(u, v, weight=weight)
        
        self.graph = G

    def num_simplices(self, k):
        if k == 0:
            res = np.sum(self.node_vec)
        elif k == 1:
            res = np.sum(self.edge_vec)
        else:
            res = np.sum(self.face_vec)
        return int(res)

    def star(self, node_vec, edge_vec):        
        star_edges = edge_vec + np.abs(self.B0_main).T @ node_vec
        star_edges = (star_edges > 0).astype(int)
        star_faces = np.abs(self.B1_main).T @ star_edges
        star_faces = (star_faces > 0).astype(int)
        return star_edges, star_faces
        
    def closure(self, node_vec, edge_vec, face_vec):
        edges_needed = edge_vec + np.abs(self.B1_main) @ face_vec
        edges_needed = np.array([int(edges_needed[i] > 0) for i in range(edge_vec.shape[0])])
        nodes_needed = node_vec + np.abs(self.B0_main) @ edges_needed 
        nodes_needed = np.array([int(nodes_needed[i] > 0) for i in range(node_vec.shape[0])])
        return nodes_needed, edges_needed

    def link(self, node_vec, edge_vec, face_vec):
        # closure (cl) then star (st)
        nodes_cs, edges_cs = self.closure(node_vec, edge_vec, face_vec)
        edges_cs, faces_cs = self.star(nodes_cs, edges_cs)
        # star (st) then closure (cl)
        edges_sc, faces_sc = self.star(node_vec, edge_vec)
        nodes_sc, edges_sc = self.closure(node_vec, edges_sc, faces_sc)

        # Link = cl(st(x)) \ st(cl(x)) 
        link_nodes = np.maximum(nodes_sc - nodes_cs, 0)
        link_nodes = (link_nodes > 0).astype(int)

        link_edges = np.maximum(edges_sc - edges_cs, 0)
        link_edges = (link_edges > 0).astype(int)

        return link_nodes, link_edges

    def euler_characterstic(self):
        num_nodes = np.sum(self.node_vec)
        num_edges = np.sum(self.edge_vec)
        num_faces = np.sum(self.face_vec)
        return num_nodes - num_edges + num_faces

    def plot(self):
        for i, face in enumerate(self.faces):
            if self.face_vec[i] == 0:
                continue
            n1, n2, n3 = face
            tri = np.vstack([self.nodes[n1], self.nodes[n2], self.nodes[n3]])
            ti = plt.Polygon(tri, color='black', alpha=0.1)
            plt.gca().add_patch(ti)

        for i, edge in enumerate(self.edges):
            if self.edge_vec[i] == 0:
                continue
            n1, n2 = edge
            line = np.vstack([self.nodes[n1], self.nodes[n2]])

            plt.plot(line[:,0], line[:,1], color='black', linewidth=1, alpha=0.6)
            plt.axis('off')

        active_nodes = self.node_vec.astype(bool)

        plt.scatter(self.nodes[active_nodes,0], self.nodes[active_nodes,1], color='black', s=7)

    def plot_trajectories(self, trajectories, color):

        self.plot()
        
        #colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        for i, trajectory in enumerate(trajectories):
            #c = colors[i%len(colors)]
            for edge in trajectory:
                n1, n2 = edge
                line = np.vstack([self.nodes[n1], self.nodes[n2]])

                plt.plot(line[:,0], line[:,1], color=color, linewidth=3)

            start_end_nodes = np.array([trajectory[0][0], trajectory[-1][-1]])

            plt.scatter(self.nodes[start_end_nodes, 0], self.nodes[start_end_nodes, 1], color='black', s=70, marker='o')#, color='k', marker='x')

    def make_holes(self, hole_locs, r=0.25):
        hole_nodes = set()
        
        for i, hole in enumerate(hole_locs):
            hole_idx = np.linalg.norm(self.nodes - hole, axis=1) < r
            hole_idx = set(np.where(hole_idx == True)[0])
            hole_nodes = hole_nodes.union(hole_idx)
        
        nodes_to_remove = {0:set(hole_nodes)}
        self.remove_simplices(nodes_to_remove)

    def compute_proj_matrix(self, thresh=1e-12):
        B0, B1 = self.get_incidence_matrices()
        L = B0.T @ B0 + B1 @ B1.T

        eig_w, eig_v = np.linalg.eigh(L)
        ev_list = [(eig_w[i], eig_v[i]) for i in range(len(eig_w))]
        ev_list.sort(key=lambda var:var[0], reverse=False)
        eig_w, eig_v = np.array([tup[0] for tup in ev_list]), np.array([tup[1] for tup in ev_list])

        num_holes = np.sum(np.abs(eig_w) < thresh)
        H = eig_v[:,:num_holes]

        self.H, self.num_holes = H, num_holes

    def harm_proj(self, x):
        x_hat = x[self.edge_vec.astype(bool)] # consider only edges in edge set
        y = (self.H.T @ x_hat).T
        return y

def find_hole_edges(SC, plot=False):
    node_zeros = 1 - SC.node_vec
    null_edge = np.zeros_like(SC.edge_vec)
    null_face =  np.zeros_like(SC.face_vec)
    link_nodes, link_edges = SC.link(node_zeros, null_edge, null_face)
    if plot:
        S2 = SimplicialComplex(SC.nodes)
        edge_idx = np.where(link_edges == 1)[0]
        S2.add_simplices({1:set(edge_idx)})

        #print(int(np.sum(link_nodes)), 'LINK NODES')
        #print(int(np.sum(link_edges)), 'LINK EDGES')

        plt.figure()
        plt.subplot(2, 1, 1)
        plt.xlim([-1,1])
        plt.ylim([-1,1])
        SC.plot()
        plt.subplot(2, 1, 2)
        plt.xlim([-1,1])
        plt.ylim([-1,1])
        S2.plot()

    link_nodes = link_nodes[SC.node_vec == 1]
    link_edges = link_edges[SC.edge_vec == 1]

    return link_nodes, link_edges

