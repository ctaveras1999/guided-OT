import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import networkx as nx
from simplicial_paths import *

def edges_from_shortest_path(shortest_path):
    return [(shortest_path[i], shortest_path[i+1]) for i in range(len(shortest_path)-1)]
            

def get_near_pts(SC, point, num_pts):
    if num_pts == 1:
        return [point]

    edges = [s for (i,s) in enumerate(SC.edges) if ((point in s) and (SC.edge_vec[i]))]
    neighbors = [(j[0] if j[0] != point else j[1]) for j in edges]
    point_set = [point] + neighbors

    rand_pts = list(np.random.choice(point_set, num_pts))

    pts = rand_pts

    return pts

# Adapted from 
# https://git.rwth-aachen.de/netsci/trajectory-outlier-detection-flow-embeddings/-/blob/main/util/trajectory.py 
def generate_paths(SC, start, end, num_trajectories):
    # Use 0-hop and 1-hop neighborhood of start and end points to generate 
    # num_trajectories start and end points.

    # Copy graph
    G = deepcopy(SC.graph)

    # Multiply adjacency matrix by one-hot start point and end point vectors.
    start_points = get_near_pts(SC, start, num_trajectories)
    end_points = get_near_pts(SC, end, num_trajectories)

    paths = []

    for i in range(num_trajectories):
        node_path = nx.shortest_path(G, start_points[i], end_points[i])
        path = edges_from_shortest_path(node_path)
        for (v0, v1) in path:
            G[v0][v1]['weight'] += 1e-4 

        paths.append(path)

    return paths

def path_interp(SC, path1, path2):
    new_path, start, end = [], path1[-1][-1], path2[0][0]

    interp_path = generate_paths(SC, start, end, 1)[0]
    new_path += path1
    new_path += interp_path
    new_path += path2

    return [new_path]


# return nodes closest to coordinate
def coord_to_node(SC, coords, num_nodes=1):
    node_set = []
    
    for i, coord in enumerate(coords):
        node_dists = np.linalg.norm(SC.nodes - coord, axis=1)
        node_order = np.argsort(node_dists)[:num_nodes]
        node_set.append(node_order)
    
    return np.array(node_set)

# coords is a sequence of nodes that we want our trajectories to pass near.  
class Trajectory:
    def __init__(self, SC, coords, num_trajectories, num_nodes):
        self.SC = SC
        self.G = deepcopy(SC.graph)

        if coords is None:
            return

        self.SC = SC
        self.G = deepcopy(SC.graph)

        node_set = coord_to_node(SC, coords, num_nodes)
        num_coords = len(coords)


        self.all_trajectory_nodes, self.all_trajectories = [], []

        for i in range(num_trajectories):
            idxs = np.random.choice(np.arange(num_nodes), num_coords)
            skeleton = [node_set[i, idxs[i]] for i in range(num_coords)]
            trajectory_nodes = [skeleton[0]]
            trajectory = []
    
            for j in range(len(skeleton)-1):
                path = nx.dijkstra_path(self.G, skeleton[j], skeleton[j+1])
                edges_in_path = edges_from_shortest_path(path)
                trajectory_nodes += path[1:]
                trajectory += edges_in_path

                for (v0, v1) in edges_in_path:
                    self.G[v0][v1]['weight'] += 1e-2

            self.all_trajectory_nodes.append(trajectory_nodes)
            self.all_trajectories.append(trajectory)
            
    def set(self, edges):
        self.all_trajectories = [edges]
        self.all_trajectory_nodes = [[edges[0]] + [[e[1] for e in edges]]]


    def trajectory_to_edgevec(self, trajectory):
        x_vec = np.zeros_like(self.SC.edge_vec)
        edge_seq_vec = np.zeros((len(trajectory), len(self.SC.edge_vec)))

        for i, edge in enumerate(trajectory):
            e1, e2 = edge
            parity = -2 * int(e1 > e2) + 1
            actual_edge = ( min([e1, e2]), max([e1, e2]) )
            edge_idx = self.SC.edges.index(actual_edge)
            x_vec[edge_idx] += parity
            edge_seq_vec[i,edge_idx] = parity

        return x_vec, edge_seq_vec

    def project_trajectories(self):
        proj_res_path = []
        proj_res_val = []
        for _, trajectory in enumerate(self.all_trajectories):
            x_vec, edge_vec_seq = self.trajectory_to_edgevec(trajectory)
            harm_proj_path = self.SC.harm_proj(x_vec) # projection of overall path
            intermediate_proj_size = (len(trajectory)+1, self.SC.num_holes) # +1 to include origin in path
            harm_proj_intermediate = np.zeros(intermediate_proj_size) # edge-by-edge projection of path
            for i in range(edge_vec_seq.shape[0]):
                x_i = edge_vec_seq[i,:]
                x_i_proj = self.SC.harm_proj(x_i)
                harm_proj_intermediate[i+1,:] = x_i_proj
            harm_proj_intermediate = np.cumsum(harm_proj_intermediate, axis=0)

            proj_res_path.append(harm_proj_intermediate)
            proj_res_val.append(harm_proj_path)

        return proj_res_path, np.array(proj_res_val)

    def plot_trajectory_proj(self, class_name, color):
        proj_res_path, proj_res_val = self.project_trajectories()
        all_projections = None

        for _, trajectory_proj in enumerate(proj_res_path):
            projections = trajectory_proj #np.cumsum(trajectory_proj, axis=0)
            if all_projections is None:
                all_projections = projections
            else:
                all_projections = np.vstack([all_projections, projections])


        all_projections = np.array(all_projections)
        plt.scatter(all_projections[:,0], all_projections[:,1], c=color, marker=',', s=5)
        plt.scatter(proj_res_val[:,0], proj_res_val[:,1], c=color, label=class_name, marker='*', s=250)

    """
    def trajectory_to_edgevec(self, trajectory):
        x_vec = np.zeros_like(self.SC.edge_vec)
        for _, edge in enumerate(trajectory):
            e1, e2 = edge
            parity = -2 * int(e1 > e2) + 1
            actual_edge = ( min([e1, e2]), max([e1, e2]) )
            edge_idx = self.SC.edges.index(actual_edge)
            x_vec[edge_idx] += parity

        return x_vec

    def project_edge(self, edge):
        e1, e2 = edge
        parity = -2 * int(e1 > e2) + 1
        actual_edge =  ( min([e1, e2]), max([e1, e2]) )
        edge_idx = self.SC.edges.index(actual_edge)
        x_vec = np.zeros_like(self.SC.edge_vec)
        x_vec[edge_idx] = parity
        return x_vec, edge_idx


    def project_trajectories(self):
        proj_res_path = []
        proj_res_val = []
        for _, trajectory in enumerate(self.all_trajectories):
            traj_len = len(trajectory)
            #print(traj_len, self.SC.num_holes)
            trajectory_projs = np.zeros((traj_len, self.SC.num_holes))
            #print(trajectory_projs.shape)
            for i, edge in enumerate(trajectory):
                x_vec, edge_idx = self.project_edge(edge)
                edge_proj = self.SC.harm_proj(x_vec)[edge_idx,:]
                #print(edge_proj.shape, trajectory_projs.shape)
                trajectory_projs[i] = edge_proj

            proj_sum = np.sum(trajectory_projs, axis=0)

            proj_res_path.append(trajectory_projs)
            proj_res_val.append(proj_sum)

        return proj_res_path, np.array(proj_res_val)"""

"=============================================================================="

if __name__ == "__main__":
    n_side, point_gen_mode = 20, 0
    hole_locs = [(0.5, 0.5), (-0.5, -0.5)]
    r = 0.1

    #np.random.seed(1)
    pts = generate_pts(point_gen_mode, n_side)
    SC = SimplicialComplex(pts)
    all_faces = np.arange(SC.face_vec.shape[0])
    faces_to_add = {2:set(all_faces)}
    SC.add_simplices(faces_to_add)
    SC.make_holes(hole_locs, r)

    num_paths, num_nodes = 2, 6

    trajectories = []

    #coords1 = [(-0.75,-0.75), (0,-0.75), (0,0), (0,0.75), (0.75,0.75)]
    #trajectories.append(Trajectory(SC, coords1, num_paths, num_nodes))

    #coords2 = [(-0.75,-0.75), (-0.75,0.5), (0.75,0.75)]
    #trajectories.append(Trajectory(SC, coords2, num_paths, num_nodes))

    #coords3 = [(-0.75,-0.75), (0.5,-0.75), (0.75,0.75)]
    #trajectories.append(Trajectory(SC, coords3, num_paths, num_nodes))


    coords3 = [(-0.75,-0.75), (0.75,-0.75), (0.75,0.75), (-0.75,0.75), (-0.75,-0.75)]
    trajectories.append(Trajectory(SC, coords3, num_paths, num_nodes))

    proj_res_path = []
    proj_res_val = []
    for i, trajectory in enumerate(trajectories):
        for path in trajectory.path_edges:
            x_vec = np.zeros_like(SC.edge_vec)
            parity_sum = 0
            for i, edge in enumerate(path):
                e1, e2 = edge
                parity = -2 * int(e1 > e2) + 1
                actual_edge = ( min([e1, e2]), max([e1, e2]) )
                edge_idx = SC.edges.index(actual_edge)
                x_vec[edge_idx] += parity
            harm_proj_path = SC.harm_proj(x_vec)
            harm_proj_val = np.sum(harm_proj_path, axis=0)
            proj_res_path.append(harm_proj_path)
            proj_res_val.append(harm_proj_val)

        plt.figure()
        SC.plot_trajectories(trajectory.path_edges)

    for val in proj_res_val:
        print(val)    
    plt.show()



    """all_faces = np.arange(S.face_vec.shape[0])
    faces_to_add = {2:set(all_faces)}
    S.add_simplices(faces_to_add)
    S.make_holes(hole_locs, r)
    
    nodes_to_cross = coord_to_node(S, [(-1,-1), (0,0), (1,1)], 3)
    print(nodes_to_cross)

    start, end, num_trajectories = 2 * (n_side + 1), (n_side-1) ** 2 - 3, 2
    paths = generate_paths(S, start, end, num_trajectories)

    back_path = [(b, a) for (a,b) in paths[1][::-1]]
    new_path = path_interp(S, paths[0], back_path)

    for path in new_path:
        x_vec = np.zeros_like(S.edge_vec)
        parity_sum = 0
        for i, edge in enumerate(path):
            e1, e2 = edge
            parity = -2 * int(e1 > e2) + 1
            actual_edge = ( min([e1, e2]), max([e1, e2]) )
            edge_idx = S.edges.index(actual_edge)
            x_vec[edge_idx] += parity
        print(np.sum(S.harm_proj(x_vec), axis=0))

    S.plot_trajectories(paths)
    plt.show()"""


