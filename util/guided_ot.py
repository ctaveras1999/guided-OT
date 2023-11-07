import numpy as np
from util.trajectories import *
from util.pqdict import pqdict

def plot_ref_vs_guided(SC, ref_path, guided_path, figsize=(12,12)):
    labels = ['ref', 'guided']
    colors = ['red', 'blue']
    path_coords = [SC.nodes[v] for v in guided_path.nodes]
    ref_coords = [SC.nodes[v] for v in ref_path.nodes]
    class_coords = [ref_coords, path_coords]
    holder = Trajectory_Holder(SC, class_coords, 1, labels, colors, 1, 0)
    holder.plot_paths(True, figsize)

def augmented_dijkstra(SC, start, end, ref_proj, alpha, verbose=False, num_extra=0, other=False, eps=1e-6):
    
    def backtrace(prev, start, end):
        node = end
        path = []
        while node != start:
            path.append(node)
            node = prev[node]
        path.append(node) 
        path.reverse()
        return path
    
    def get_next_proj(u, v, curr_proj):
        edge = Trajectory(SC, [u,v], num_extra)
        proj = edge.edge_projections[1] # idx 0 is zero vec, idx 1 is edge proj
        next_proj = curr_proj + proj
        return next_proj

    def proj_cost(proj):
        #print(proj.shape, ref_proj.shape)
        proj_diff = proj - ref_proj
        return np.round(np.linalg.norm(proj_diff, 1), 8)

    def cost(u, v, proj, edge_weight):
        edge_cost = SC.graph[u][v]['weight']
        next_edge_weight = edge_weight + edge_cost
        next_proj = get_next_proj(u, v, proj)
        next_proj_diff = proj_cost(next_proj)
        potential_cost = next_edge_weight + alpha * next_proj_diff
        return  potential_cost, next_proj, next_edge_weight
    
    dim_proj = (SC.H.shape[0] + num_extra) if (num_extra > 0) else SC.H.shape[0]

    prev = {}
    dist = {v:np.inf for v in range(SC.node_vec.shape[0])}
    proj = {v:np.inf * np.ones(dim_proj) for v in range(SC.node_vec.shape[0])}
    edge_weight = {v:np.inf for v in range(SC.node_vec.shape[0])}
    visited = set()

    pq = pqdict()

    for i  in np.where(SC.node_vec == 1)[0]:
        pq[i] = float("inf") if (i != start) else 0

    dist[start], edge_weight[start], proj[start] = 0, 0, np.zeros(dim_proj)
    
    visit_order = []
    node_children = {node:0 for node in np.where(SC.node_vec == 1)[0]}
    
    while len(pq) > 0: # u = node with lowest cost
        u = pq.pop()
        if (u == end) and (other is False):
            break
        visit_order.append(u)
        curr_proj = proj[u]
        curr_edge_weight = edge_weight[u]
        visited.add(u)
        if verbose:
            print('. . . . . . . . . . . '*3)
            print(f'visiting {u} with cost {dist[u]}')
            print(f'node position = {SC.nodes[u]}')

        num_children = 0

        for v in SC.graph.neighbors(u): # v = neighbor of lowest cost node u
            if v in visited: # node must not have already been visited
                continue

            new_cost, new_proj, new_edge_weight = cost(u, v, curr_proj, curr_edge_weight)
            old_cost = dist[v]

            if verbose:
                print(f'curr neighbor = {v}, curr cost = {old_cost}, new cost : {new_cost}')
                print(f'curr proj diff = {proj_cost(proj[v])}, new proj diff = {proj_cost(new_proj)}')
                print(f'neighbor position = {SC.nodes[v]}')
                print(f"old edge weights: {edge_weight[v]} & new edge weights: {new_edge_weight}")

            #if new_edge_weight < edge_weight[v]:
            if ((new_cost == old_cost) and u < prev[v]) or (new_cost < old_cost):
                if v in prev:
                    node_children[prev[v]] -= 1
                num_children += 1
                dist[v] = new_cost
                prev[v] = u
                edge_weight[v] = new_edge_weight
                proj[v] = new_proj 
                pq[v] = new_cost
            
        node_children[u] = num_children
            
    path = Trajectory(SC, backtrace(prev, start, end), num_extra)

    if verbose:
        print("=== Dijkstra's Algo Output ===")
        print("Distances")
        print(dist)
        #print("Visited")
        #print(visited)
        #print("Previous")
        #print(prev)
        print("Path")
        print(path)
    
    if other:
        heads = [node for node in node_children if not node_children[node]]
        other_paths = [Trajectory(SC, backtrace(prev, start, head), num_extra) for head in heads if head != end]
    else: 
        other_paths = []


    return path, dist[end], prev, visit_order, other_paths

# Augmented dijkstra where we ignore the edge weights and use projection divergence
def augmented_dijkstra_v2(SC, start, end, ref_proj, verbose=False, extra=False, eps=1e-10):
    
    def backtrace(prev, start, end):
        node = end
        path = []
        while node != start:
            path.append(node)
            node = prev[node]
        path.append(node) 
        path.reverse()
        return path
    
    def get_next_proj(u, v, curr_proj):
        edge = Trajectory(SC, [u,v], extra)
        proj = edge.edge_projections[1] # idx 0 is zero vec, idx 1 is edge proj
        #print(f"Proj: {proj}")
        next_proj = curr_proj + proj
        return next_proj

    def proj_cost(proj):
        proj_diff = proj - ref_proj
        return np.round(np.linalg.norm(proj_diff, 1), 10)
    
    dim_proj = SC.H_extra.shape[0] if extra else SC.H.shape[0]

    prev = {}
    dist = {v:np.inf for v in range(SC.node_vec.shape[0])}
    proj = {v:np.inf * np.ones(dim_proj) for v in range(SC.node_vec.shape[0])}
    edge_weight = {v:np.inf for v in range(SC.node_vec.shape[0])}
    visited = set()

    pq = pqdict()

    for i  in np.where(SC.node_vec == 1)[0]:
        pq[i] = float("inf") if (i != start) else 0

    dist[start], edge_weight[start], proj[start] = 0, 0, np.zeros(dim_proj)
    
    visit_order = []
    node_children = {node:0 for node in np.where(SC.node_vec == 1)[0]}
    
    while len(pq) > 0: # u = node with lowest cost
        u = pq.pop()
        visit_order.append(u)
        curr_proj = proj[u]
        curr_edge_weight = edge_weight[u]
        visited.add(u)
        if verbose:
            print('. . . . . . . . . . . '*3)
            print(f'visiting {u} with cost {dist[u]}')
            print(f'node position = {SC.nodes[u]}')

        num_children = 0

        for v in SC.graph.neighbors(u): # v = neighbor of lowest cost node u
            if v in visited: # node must not have already been visited
                continue
            new_cost = curr_edge_weight + SC.graph[u][v]['weight']
            old_cost = edge_weight[v]
            new_proj = get_next_proj(u, v, curr_proj)

            if verbose:
                print(f'curr neighbor = {v}, curr cost = {old_cost}, new cost : {new_cost}')
                print(f'curr proj diff = {proj_cost(proj[v])}, new proj diff = {proj_cost(new_proj)}')
                print(f'neighbor position = {SC.nodes[v]}')
                print(f"old edge weights: {edge_weight[v]} & new edge weights: {new_cost}")


            cond1 = edge_weight[v] == np.inf
            cond2 = (np.linalg.norm(proj[v] - new_proj, ord=1) < eps) and (new_cost < old_cost)
            cond3 = (proj_cost(new_proj) < proj_cost(proj[v]))

            if cond1 or cond2 or cond3:
                if v in prev:
                    node_children[prev[v]] -= 1
                num_children += 1
                dist[v] = new_cost
                prev[v] = u
                edge_weight[v] = new_cost
                proj[v] = new_proj 
                pq[v] = new_cost
            
        node_children[u] = num_children
            
    path = Trajectory(SC, backtrace(prev, start, end), extra)

    heads = [node for node in node_children if not node_children[node]]

    other_paths = [Trajectory(SC, backtrace(prev, start, head), extra) for head in heads if head != end]

    if verbose:
        print("=== Dijkstra's Algo Output ===")
        print("Distances")
        print(dist)
        #print("Visited")
        #print(visited)
        #print("Previous")
        #print(prev)
        print("Path")
        print(path)
    

    return path, dist[end], prev, visit_order, other_paths

def augmented_dijkstra_v3(SC, start, end, ref_proj, alpha, verbose=False, num_extra=0):
    
    def backtrace(prev, start, end):
        node = end
        path = []
        while node != start:
            path.append(node)
            node = prev[node]
        path.append(node) 
        path.reverse()
        return path
    
    def get_next_proj(u, v, curr_proj):
        edge = Trajectory(SC, [u,v], num_extra)
        proj = edge.edge_projections[1] # idx 0 is zero vec, idx 1 is edge proj
        next_proj = curr_proj + proj
        return next_proj

    def proj_cost(proj):
        proj_diff = proj - ref_proj
        return np.round(np.linalg.norm(proj_diff, 1), 8)

    def cost(u, v, proj, edge_weight):
        edge_cost = SC.graph[u][v]['weight']
        next_edge_weight = edge_weight + edge_cost
        next_proj = get_next_proj(u, v, proj)
        next_proj_diff = proj_cost(next_proj)
        potential_cost = next_edge_weight + alpha * next_proj_diff
        return  potential_cost, next_proj, next_edge_weight
    
    dim_proj = (SC.H.shape[0] + num_extra) if (num_extra > 0) else SC.H.shape[0]


    prev = {}
    dist = {v:np.inf for v in range(SC.node_vec.shape[0])}
    proj = {v:np.inf * np.ones(dim_proj) for v in range(SC.node_vec.shape[0])}
    edge_weight = {v:np.inf for v in range(SC.node_vec.shape[0])}
    visited = set()

    pq = pqdict()

    for i  in np.where(SC.node_vec == 1)[0]:
        pq[i] = float("inf") if (i != start) else 0

    dist[start], edge_weight[start], proj[start] = 0, 0, np.zeros(dim_proj)
    
    visit_order = []
    node_children = {node:0 for node in np.where(SC.node_vec == 1)[0]}
    
    while len(pq) > 0: # u = node with lowest cost
        u = pq.pop()
        visit_order.append(u)
        curr_proj = proj[u]
        curr_edge_weight = edge_weight[u]
        visited.add(u)
        if verbose:
            print('. . . . . . . . . . . '*3)
            print(f'visiting {u} with cost {dist[u]}')
            print(f'node position = {SC.nodes[u]}')

        num_children = 0

        for v in SC.graph.neighbors(u): # v = neighbor of lowest cost node u
            if v in visited: # node must not have already been visited
                continue

            new_cost, new_proj, new_edge_weight = cost(u, v, curr_proj, curr_edge_weight)
            old_cost = dist[v]

            if verbose:
                print(f'curr neighbor = {v}, curr cost = {old_cost}, new cost : {new_cost}')
                print(f'curr proj diff = {proj_cost(proj[v])}, new proj diff = {proj_cost(new_proj)}')
                print(f'neighbor position = {SC.nodes[v]}')
                print(f"old edge weights: {edge_weight[v]} & new edge weights: {new_edge_weight}")

            #if new_edge_weight < edge_weight[v]:
            if new_cost < old_cost:
                if v in prev:
                    node_children[prev[v]] -= 1
                num_children += 1
                dist[v] = new_cost
                prev[v] = u
                edge_weight[v] = new_edge_weight
                proj[v] = new_proj 
                pq[v] = new_cost
            
        node_children[u] = num_children
            
    path = Trajectory(SC, backtrace(prev, start, end), num_extra)

    heads = [node for node in node_children if not node_children[node]]

    other_paths = [Trajectory(SC, backtrace(prev, start, head), num_extra) for head in heads if head != end]

    if verbose:
        print("=== Dijkstra's Algo Output ===")
        print("Distances")
        print(dist)
        #print("Visited")
        #print(visited)
        #print("Previous")
        #print(prev)
        print("Path")
        print(path)
    

    return path, dist[end], prev, visit_order, other_paths

def accumulated_proj_diff_dijkstra(SC, start, end, ref_path, alpha,  num_extra=0, other=False):
    def backtrace(prev, start, end):
        path, node = [], end
        while node != start:
            path.append(node)
            node = prev[node]
        path.append(node) 
        path.reverse()
        return path
    
    def get_next_proj(u, v, curr_proj):
        edge = Trajectory(SC, [u,v], num_extra)
        proj = edge.edge_projections[1] # idx 0 is zero vec, idx 1 is edge proj
        next_proj = curr_proj + proj
        return next_proj

    def proj_cost(proj, ref_proj):
        proj_diff = np.linalg.norm(proj - ref_proj, np.inf)
        return np.round(proj_diff, 16)

    def cost(u, v, proj, edge_weight, ref_proj):
        edge_cost = SC.graph[u][v]['weight']
        next_edge_weight = edge_weight + edge_cost
        next_proj = get_next_proj(u, v, proj)
        next_proj_diff = proj_cost(next_proj, ref_proj)
        potential_cost = next_edge_weight + alpha * next_proj_diff
        return potential_cost, next_proj, next_edge_weight
    
    def get_ref_proj(idx):
        if 0 <= idx < len(ref_path.path_edges):
            proj = ref_path.edge_projections[idx]
        elif idx < 0:
            proj = np.inf * np.ones(dim_proj)
        else:
            proj = ref_path.edge_projections[-1]
        return proj

    def init():
        prev = {}
        edge_sum = {v:np.inf for v in range(SC.node_vec.shape[0])}
        dist2src = {v:-1 for v in range(SC.node_vec.shape[0])}
        cost_dict = {v:np.inf for v in range(SC.node_vec.shape[0])}
        node_proj = {v:np.inf * np.ones(dim_proj) for v in range(SC.node_vec.shape[0])}

        visited = set()
        pq = pqdict()

        for i  in np.where(SC.node_vec == 1)[0]:
            pq[i] = float("inf") if (i != start) else 0

        dist2src[start], edge_sum[start] = 0, 0, 
        node_proj[start], cost_dict[start] = np.zeros(dim_proj), 0    


        return prev, edge_sum, dist2src, cost_dict, node_proj, visited, pq

    dim_proj = (SC.H.shape[0] + num_extra) if (num_extra > 0) else SC.H.shape[0]
    prev, edge_sum, dist2src, cost_dict, node_proj, visited, pq = init()

    node_children = {node:0 for node in np.where(SC.node_vec == 1)[0]}


    while len(pq) > 0:
        u = pq.pop()
        proj_u = node_proj[u]
        weight_u = edge_sum[u]
        n_edges_u = dist2src[u]
        visited.add(u)
        n_edges_v = n_edges_u + 1        

        num_children = 0

        for v in SC.graph.neighbors(u): # v = neighbor of lowest cost node u
            if v in visited: # node must not have already been visited
                continue

            ref_proj = get_ref_proj(n_edges_v)
            new_cost, new_proj_v, new_edge_sum_v = cost(u, v, proj_u, weight_u, ref_proj)
            old_cost = cost_dict[v]

            if new_cost < old_cost:
                if v in prev:
                    node_children[prev[v]] -= 1
                num_children += 1
                
                proj_diff = np.linalg.norm(new_proj_v - ref_proj, 1)
                edge_weight = new_edge_sum_v - edge_sum[u]
                cost_dict[v] = new_cost
                prev[v] = u
                edge_sum[v] = new_edge_sum_v
                node_proj[v] = new_proj_v
                dist2src[v] = n_edges_v
                pq[v] = new_cost
            
        node_children[u] = num_children

    path = Trajectory(SC, backtrace(prev, start, end), num_extra)

    heads = [node for node in node_children if not node_children[node]]

    other_paths = [Trajectory(SC, backtrace(prev, start, head), num_extra) for head in heads if head != end]

    if not other:
        return path
    else:
        return path, other_paths

def rollout(SC, ref_path, alpha):
    start, end = ref_path.nodes[0], ref_path.nodes[-1]
    vk = start
    path = [start]

    ref_class = Trajectory_Class(SC)
    ref_class.add_path(ref_path)
    best_costs = []
    best_paths = []

    while vk != end:
        print("vk =", vk, ", neighbors =", list(SC.graph.neighbors(vk)))
        p_ref_nodes = ref_path.nodes if (vk == start) else path[1:][::-1] + ref_path.nodes
        ref_proj = ref_path.edge_projections[-1]

        best_cost, best_node, best_path = np.inf, None, None
        for u in SC.graph.neighbors(vk):
            p_ref_nodes_u = [u] + p_ref_nodes
            p_ref_u = Trajectory(SC, p_ref_nodes_u, 0)
            ref_proj_u = p_ref_u.edge_projections[-1]
            partial_path_u,_,_,_,_ = augmented_dijkstra(SC, u, end, ref_proj_u, alpha, False, 0)
            path_u = Trajectory(SC, path + partial_path_u.nodes, 0)
            path_u_proj = path_u.edge_projections[-1]
            path_u_len = path_u.path_length[-1]
            cost_u = path_u_len + alpha * np.linalg.norm(path_u_proj - ref_proj, ord=1)
            print("node: ", u, "cost: ", cost_u)
            if cost_u < best_cost:
                best_cost = cost_u
                best_node = u
                best_path = path_u
                
        print("best node =", best_node)
        path += [best_node]
        best_costs += [best_cost]
        best_paths += [best_path]
        #dijkstra_tree(SC, best_path, [best_path], ref_class)

        vk = best_node

    return Trajectory(SC, path, 0), best_costs, best_paths

def find_intermediate_path(SC, partial, ref_path, end, alpha):
    #ref_prefix = [curr] + partial[1:][::-1] if (len(partial) > 1) else [curr]
    ref_prefix = partial[1:][::-1]
    p_ref_nodes = ref_prefix + ref_path.nodes 
    p_ref_w_prefix = Trajectory(SC, p_ref_nodes, 0)
    ref_proj_w_prefix = p_ref_w_prefix.edge_projections[-1] 
    curr = partial[-1]
    remainder,_,_,_,_ = augmented_dijkstra(SC, curr, end, ref_proj_w_prefix, alpha, False, 0)
    path_u = Trajectory(SC, partial + remainder.nodes[1:])
    return path_u

def get_path_cost(path, ref_path, alpha):
    path_len = path.path_length[-1]
    path_proj = path.edge_projections[-1]
    ref_proj = ref_path.edge_projections[-1]
    cost = path_len + alpha * np.linalg.norm(path_proj - ref_proj, ord=1)
    return cost

def rollout_rec(SC, partial, curr_node, ref_path, alpha, start, end, depth, max_depth):
    if curr_node == end:
        path = Trajectory(SC, partial, 0)
        cost = get_path_cost(path, ref_path, alpha)
        return path, cost

    elif depth == max_depth:
        path = find_intermediate_path(SC, partial, ref_path, end, alpha)
        cost = get_path_cost(path, ref_path, alpha)
        return path, cost

    path_options = []
    costs = []
    
    for v in SC.graph.neighbors(curr_node):
        path_v, cost_path_v = rollout_rec(SC, partial + [v], v, ref_path, alpha, start, end, depth+1, max_depth)
        path_options += [path_v]
        costs += [cost_path_v]
        print("cost:", cost_path_v, "curr next:", (curr_node, v),  "path:", path_v)

    best_cost, best_idx = np.min(np.array(costs)), np.argmin(np.array(costs))
    best_path = path_options[best_idx]

    return best_path, best_cost


def k_rollout(SC, ref_path, alpha, max_depth):
    start, end = ref_path.nodes[0], ref_path.nodes[-1]
    vk = start
    path = [start]

    ref_class = Trajectory_Class(SC)
    ref_class.add_path(ref_path)
    best_costs, best_paths = [], []

    k=0
    # print('vk =', vk)
    while vk != end:
        path_vk, cost_vk = rollout_rec(SC, path, vk, ref_path, alpha, start, end, 0, max_depth)
        vk = path_vk.nodes[k+1]
        path += [vk]
        best_paths += [path_vk]
        best_costs += [cost_vk]
        k+=1 
        #print("-"*200)
        #print('vk =', vk)

    return Trajectory(SC, path, 0), best_costs, best_paths


def make_analytical_func(r, N):
    numerator = 1j * 2 * np.pi * np.arange(N-1)
    zis = r * np.exp(numerator/(N-1))
    return  zis

def evaluate_f(zeros, z):
    return np.prod(z - zeros)

def H_signature(zeros, poles, z):
    N = poles.shape[0]
    F = np.zeros(N)
    for i in range(N):
        denoms_i = [poles[:i]] + poles[i+1:]
        F[i] = evaluate_f(zeros, z) / evaluate_f(denoms_i, z)
    return F    

def get_A(zeros, poles):
    N = len(poles)
    A_arr = []
    for l in range(N):
        z_l = poles[l]
        f0_l = evaluate_f(zeros, z_l)
        A_l = f0_l
        for j, z_j in enumerate(poles):
            if j == l:
                continue 
            A_l = A_l / (z_l - z_j)
        A_arr.append(A_l)
    return np.array(A_arr)

def make_proj_dict(SC, zeros, poles, num_digits=15):
    proj_dict = {}
    A_arr = get_A(zeros, poles)

    for i, edge in enumerate(SC.edges):
        if SC.edge_vec[i] == 0:
            continue
        v1, v2 = edge
        z1 = SC.nodes[v1][0] + 1j * SC.nodes[v1][1]
        z2 = SC.nodes[v2][0] + 1j * SC.nodes[v2][1]

        z2_diff = z2 - poles
        z1_diff = z1 - poles
        
        mag_diff = np.log(np.abs(z2_diff)) - np.log(np.abs(z1_diff))
        angle_diff = np.angle(z2_diff) - np.angle(z1_diff)
        
        for i, angle in enumerate(angle_diff):
            while angle < -np.pi:
                angle += 2 * np.pi
            while angle > np.pi:
                angle -= 2 * np.pi
            angle_diff[i] = angle

        #print(A_arr * (mag_diff + 1j * angle_diff))
        complex_sum = A_arr * (mag_diff + 1j * angle_diff)
        complex_sum = np.round(complex_sum, num_digits)
        proj_dict[(v1, v2)] = cmplx2vec(complex_sum)
        proj_dict[(v2, v1)] = -cmplx2vec(complex_sum)
    
    return proj_dict

def get_path_proj(path, proj_dict):
    tot = np.array([proj_dict[path[i], path[i+1]] for i in range(len(path)-1)])
    #print(tot)
    tot_sum = np.sum(tot, axis=0)
    return tot_sum

def r2complex(coords):
    return np.array(coords)[:,0] + 1j * np.array(coords)[:,1]

def cmplx2vec(x):
    if isinstance(x, np.complex128):
        x_arr = np.zeros(2)
        x_arr[0], x_arr[1] = np.real(x), np.imag(x)
    else: 
        x_arr = np.zeros(2 * len(x))
        for i, xi in enumerate(x):
            x_arr[2*i], x_arr[2*i+1] = np.real(xi), np.imag(xi)
    return x_arr

def bhattacharya(SC, ref_path, start, end, proj_dict, num_digits=6, eps=1e-4, others=False):
    ref_proj = get_path_proj(ref_path.nodes, proj_dict)
    pq = pqdict()
    zero_proj = tuple(np.zeros(ref_proj.shape[0]))
    pq[(start, zero_proj)] = 0
    prev = {(start, zero_proj):None}
    dist = {(start, zero_proj):0}
    visited = set()

    while True:
        u, proj_u = pq.pop()
        dist_u = dist[(u, proj_u)]
        visited.add((u, proj_u))
        if (u == end) and (np.linalg.norm(ref_proj - proj_u) < eps):
            break

        for v in SC.graph.neighbors(u):
            proj_v = np.array(proj_u) + proj_dict[(u, v)]
            proj_v = tuple(np.round(proj_v, num_digits+1))
            if (v, proj_v) in visited:
                continue
            dist_v = dist_u + SC.graph[u][v]["weight"]
            v_list = [x for x in dist if (x[0] == v) and (np.linalg.norm(np.array(x[1]) - np.array(proj_v)) < eps)]
            if len(v_list) == 0:
                prev[(v, proj_v)] = (u, proj_u)
                dist[(v, proj_v)] = dist_v
                pq[(v, proj_v)] = dist_v

    def backtrace(prev, start, end, ref_proj, others=others):
        node = None
        other_nodes = []
        for x in prev:
            if (x[0] == end) and (np.linalg.norm(x[1] - ref_proj) < eps):
                node = x
                break
            elif x[0] == end:
                other_nodes.append(x)
        path = []
        while node[0] != start:
            path.append(node[0])
            node = prev[node]
        path.append(node[0]) 
        path.reverse()
        path = Trajectory(SC, path)

        if not others:
            return path

        other_paths = []
        for node in other_nodes:
            other_path = []
            while node[0] != start:
                other_path.append(node[0])
                node = prev[node]
            other_path.append(node[0])
            other_path.reverse()
            other_paths.append(Trajectory(SC, other_path))

        return path, other_paths

    if not others:
        path = backtrace(prev, start, end, ref_proj, others)
        return path
    else:
        path, other_paths = backtrace(prev, start, end, ref_proj, others)
        return path, other_paths
