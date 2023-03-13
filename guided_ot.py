import numpy as np
from trajectories import *
from queue import PriorityQueue

def augmented_dijkstra(SC, start, end, ref_proj, alpha, verbose=False):
    
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
        edge = Trajectory(SC, [u,v])
        proj = edge.edge_projections[1] # check dimensions of this (potential bug)
        next_proj = curr_proj + proj
        return next_proj

    def proj_cost(proj):
        proj_diff = proj - ref_proj
        return np.linalg.norm(proj_diff) ** 2

    def cost(u, v, curr_proj):
        edge_cost = SC.graph[u][v]['weight']
        next_proj = get_next_proj(u, v, curr_proj)
        #curr_proj_diff = 0 if (u == start) else proj_cost(curr_proj)
        curr_proj_diff = 0
        next_proj_diff = proj_cost(next_proj)
        #print(next_proj_diff - curr_proj_diff)
        return edge_cost + alpha * (next_proj_diff - curr_proj_diff), next_proj
                                                                                   
    prev = {}
    dist = {v:np.inf for v in range(SC.node_vec.shape[0])}
    proj = {v:np.inf * np.ones(SC.num_holes) for v in range(SC.node_vec.shape[0])}
    visited = set()

    pq = PriorityQueue()
    dist[start], proj[start] = 0, np.zeros(SC.num_holes)

    pq.put( ( dist[start], start ) )

    while pq.qsize() != 0:
        curr_cost, curr_node = pq.get()
        curr_proj = proj[curr_node]
        visited.add(curr_node)
        if verbose:
            print('. . . . . . . . . . . ')
            print(f'visiting {curr_node}')
        for neighbor in SC.graph.neighbors(curr_node):
        
            neighbor_cost, neighbor_proj = cost(curr_node, neighbor, curr_proj)
            path_cost = curr_cost + neighbor_cost

            if verbose:
                print(f'path cost : {path_cost}')
                print(f'neighbor {neighbor} : {dist[neighbor]}')
                
            if path_cost < dist[neighbor]:         
                if neighbor not in visited:
                    visited.add(neighbor)
                else:      
                    if pq.empty():
                        break         
                    _ = pq.get((dist[neighbor], neighbor))

                dist[neighbor] = path_cost
                prev[neighbor] = curr_node
                proj[neighbor] = neighbor_proj    

                pq.put( (dist[neighbor], neighbor))

    if verbose:
        print("=== Dijkstra's Algo Output ===")
        print("Distances")
        print(dist)
        print("Visited")
        print(visited)
        print("Previous")
        print(prev)
    
    path = Trajectory(SC, backtrace(prev, start, end))

    return path, dist[end]
        