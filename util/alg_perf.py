from util.trajectories import *
from util.simplicial_paths import * 
from itertools import combinations


def dijkstra_tree(SC, best_path, other_paths, ref_path_class, proj_axes=(0,1), colors=['green', 'red', 'blue'], plot_others=False, figsize=(5,5), plot_paths=True, plot_proj=True, plot_save=None):
    shortest_class = Trajectory_Class(SC)
    shortest_class.add_path(best_path)
    other_class = Trajectory_Class(SC)

    for path in other_paths:
        other_class.add_path(path)

    holder = Trajectory_Holder(SC)

    if plot_others:
        holder.add_class(other_class, None, 'others', colors[0])
        
    holder.add_class(ref_path_class, None, 'reference', colors[1])
    holder.add_class(shortest_class, None, 'result', colors[2])

    if plot_paths:
        holder.plot_paths(True, figsize=figsize, fname=plot_save)

    if plot_proj:
        holder.plot_projections(proj_axes=proj_axes, figsize=figsize)

    return holder

def get_node_weights(path):
    return path.path_length

def get_path_proj_diff(path, ref_proj, norm_type=1):
    return np.round(np.linalg.norm(path.edge_projections - ref_proj, norm_type, axis=1), 10)

def compute_costs(path, ref_proj, alpha, norm_type):
    edge_weights = np.hstack([np.array([0]), get_node_weights(path)])
    proj_diffs = get_path_proj_diff(path, ref_proj, norm_type)
    return edge_weights, proj_diffs, edge_weights + alpha * proj_diffs

def compare_path_perf(path1, path2, ref_proj, alpha, norm_type=1):
    num_nodes_path1, num_nodes_path2 = len(path1.nodes), len(path2.nodes)
    len_diff = np.abs(num_nodes_path1 - num_nodes_path2)
    shorter_path = np.argmin([ num_nodes_path1, num_nodes_path2 ])
    path1_costs = compute_costs(path1, ref_proj, alpha, norm_type)
    path2_costs = compute_costs(path2, ref_proj, alpha, norm_type)
    
    if len_diff == 0:
        return path1_costs, path2_costs
    elif shorter_path == 0:
        edge_cost = np.ones(num_nodes_path2) * path1_costs[0][-1]
        edge_cost[:num_nodes_path1] = path1_costs[0]

        proj_cost = np.ones(num_nodes_path2) * path1_costs[1][-1]
        proj_cost[:num_nodes_path1] = path1_costs[1]

        total_cost = np.ones(num_nodes_path2) * path1_costs[2][-1]
        total_cost[:num_nodes_path1] = path1_costs[2]

        path1_costs = edge_cost, proj_cost, total_cost
    else:
        edge_cost = np.ones(num_nodes_path1) * path2_costs[0][-1]
        edge_cost[:num_nodes_path2] = path2_costs[0]

        proj_cost = np.ones(num_nodes_path1) * path2_costs[1][-1]
        proj_cost[:num_nodes_path2] = path2_costs[1]

        total_cost = np.ones(num_nodes_path1) * path2_costs[2][-1]
        total_cost[:num_nodes_path2] = path2_costs[2]

        path2_costs = edge_cost, proj_cost, total_cost

    return path1_costs, path2_costs

def plot_proj_diff(path1, path2):
    path1_costs, path2_costs = compare_path_perf(path1, path2, path2.edge_projections[-1], 0)
    path1_costs, path2_costs = path1_costs[1], path2_costs[1]
    colors = ["red", "blue"]
    x_axis = np.arange(np.max([len(path1.nodes), len(path2.nodes)]))
    plt.plot(x_axis, path1_costs, color=colors[0], marker="1")
    plt.plot(x_axis, path2_costs, color=colors[1], marker="2")
    plt.xlabel('Path node')
    plt.ylabel('Projection difference')
    plt.legend(["reference path", "resulting path"])


def plot_proj_diffs(path1, path2, figsize=(5,5)):
    path_proj_diff = path1.edge_projections - path2.edge_projections[-1]
    path_proj_diff = np.abs(path_proj_diff)
    num_coords = path_proj_diff.shape[1]
    plt.figure(figsize=figsize)
    total_diff = np.sum(path_proj_diff, axis=1)
    for i in range(num_coords):
        plt.plot(path_proj_diff[:,i], label='coord ' + str(i))
    plt.plot(total_diff, label='total diff')
    plt.legend()
    plt.show()

    return path_proj_diff

