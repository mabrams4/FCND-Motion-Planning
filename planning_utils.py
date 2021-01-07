from enum import Enum
from queue import PriorityQueue
import numpy as np
from shapely.geometry import Polygon, Point, LineString
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm 

NUM_SAMPLES = 200
TARGET_ALTITUDE = 5
SAFETY_DISTANCE = 5
K = 10

def create_graph(data):
    print('creating graph')
    xmin = np.floor(np.min(data[:, 0] - data[:, 3]))
    xmax = np.ceil(np.max(data[:, 0] + data[:, 3]))

    ymin = np.floor(np.min(data[:, 1] - data[:, 4]))
    ymax = np.ceil(np.max(data[:, 1] + data[:, 4]))   

    zmin = 0
    zmax = 2 * TARGET_ALTITUDE
    samples = get_samples(data, xmin, xmax, ymin, ymax, zmin, zmax)

    obstacle_dict = {}
    d_north_max = 0
    d_east_max = 0
    max_area = 0
    for i in range (data.shape[0]):
        north, east, alt, d_north, d_east, d_alt = data[i, :]
        corner1 = (north - d_north, east - d_east)
        corner2 = (north + d_north, east - d_east)
        corner3 = (north + d_north, east + d_east)
        corner4 = (north - d_north, east + d_east)

        corners = [corner1, corner2, corner3, corner4]
        height = alt + d_alt

        p = Polygon(corners)
        obstacle_dict[(north, east, alt)] = (p, height) 

        area = p.area
        if area > max_area:
            d_north_max = d_north
            d_east_max = d_east
            max_area = area

    max_radius = np.sqrt(d_north_max**2 + d_east_max**2)
    obstacle_tree = KDTree(list(obstacle_dict.keys()))

    valid_nodes = []
    for s in samples:
        obstacles = find_closest_obstacles(s, obstacle_tree, max_radius, obstacle_dict)
        if not collides_with(s, obstacles):
            valid_nodes.append(s)

    node_tree = KDTree(valid_nodes)
    graph = nx.Graph()
    print(f'attempting to connect {len(valid_nodes)} nodes to each other...')
    for n1 in tqdm(valid_nodes):
        neighbors = node_tree.query([n1], K, return_distance=False)[0]
        for index in neighbors:
            n2 = valid_nodes[index]
            if n2 == n1:
                continue
            distance = np.linalg.norm(np.array(n1) - np.array(n2))
            search_radius = 0.5*distance + max_radius
            mid_point = find_mid_point(n1,n2)
            obstacles = find_closest_obstacles(mid_point, obstacle_tree, search_radius, obstacle_dict)

            if can_connect(n1, n2, obstacles):
                graph.add_edge(n1, n2, weight=distance)

    print(f'number of edges {graph.number_of_edges()}')
    return graph, xmin, ymin, xmax, ymax, zmax, obstacle_tree, obstacle_dict, max_radius, valid_nodes

def find_closest_obstacles(node, obstacle_tree, radius, obstacle_dict):
    obstacles = []
    potential_obstacles = obstacle_tree.query_radius([node], radius, return_distance=False)[0]
    for ob in potential_obstacles:
        key_list = list(obstacle_dict.keys())
        obstacle_center = tuple(key_list[ob])
        obstacles.append(obstacle_dict[obstacle_center])
    
    return obstacles

def collides_with(node, obstacles):
    for (p, height) in obstacles:
        if p.contains(Point(node)) and height >= node[2]:
            return True
    return False   

def can_connect(n1, n2, obstacles):
    line = LineString([n1,n2])
    for (p, height) in obstacles:
        if p.crosses(line) and height >= min(n1[2],n2[2]):
            return False
    return True

def find_mid_point(n1,n2):
    x = (n1[0] + n2[0]) / 2
    y = (n1[1] + n2[1]) / 2
    z = (n1[2] + n2[2]) / 2

    return [x,y,z]

def get_samples(data, xmin, xmax, ymin, ymax, zmin, zmax):
    xvals = np.random.randint(xmin, xmax, NUM_SAMPLES)
    yvals = np.random.randint(ymin, ymax, NUM_SAMPLES)
    zvals = np.random.randint(zmin, zmax, NUM_SAMPLES)
    samples = list(zip(xvals, yvals, zvals))

    return samples

def closest_node(graph, point):
    dist = 10000
    closest = None
    for node in graph.nodes:
        d = np.linalg.norm(np.array(node) - np.array(point))
        if d < dist:
            dist = d
            closest = node        
    return closest

def prune_path(path, obstacle_tree, obstacle_dict, max_radius):
    print('pruning path')
    pruned_path = [p for p in path]
    i = 0
    while i < len(pruned_path) - 2:
        n1 = pruned_path[i]
        n2 = pruned_path[i+1]
        n3 = pruned_path[i+2]

        distance = np.linalg.norm(np.array(n1) - np.array(n3))
        search_radius = 0.5*distance + max_radius
        mid_point = find_mid_point(n1,n3)
        obstacles = find_closest_obstacles(mid_point, obstacle_tree, search_radius, obstacle_dict)
        if can_connect(n1,n3, obstacles):
            pruned_path.remove(pruned_path[i+1])
        else:
            i += 1
    print(f'pruned_path from {len(path)} to {len(pruned_path)}')        
    return pruned_path

def a_star_graph(graph, h, start, goal):
    print('running a_star')
    path = []
    queue = PriorityQueue()
    queue.put((0, start))
    visited = set(start)

    branch = {}
    found = False
    
    if start == goal:
        print(f'START {start} = GOAL {goal}')
    while not queue.empty():
        print(f'queue length: {queue.qsize()}')
        item = queue.get()
        current_cost, current_node = item
        if current_node == goal:        
            print('Found a path.')
            found = True
            break
        else:
            for next_node in graph[current_node]:
                cost =  graph.edges[current_node, next_node]['weight']
                new_cost = cost + current_cost + h(next_node, goal)
                
                if next_node not in visited:                
                    visited.add(next_node)  
                    queue.put((new_cost, next_node))             
                    branch[next_node] = (new_cost, current_node)
                    
             
    if found:
        # retrace steps
        n = goal
        path_cost = branch[n][0]
        path.append(goal)
        while branch[n][1] != start:
            path.append(branch[n][1])
            n = branch[n][1]
        path.append(branch[n][1])
    else:
        print('**********************')
        print('Failed to find a path!')
        print('**********************') 
        return None, None
    return path[::-1], path_cost

def visualize_path(data, graph, pruned_path, start, goal):
    grid, _, _ = create_grid(data) 
    fig = plt.figure()

    plt.imshow(grid, cmap='Greys', origin='lower')

    nmin = np.min(data[:, 0])
    emin = np.min(data[:, 1])

    plt.scatter(start[0] - nmin, start[1] - emin, c='pink')
    plt.scatter(goal[0] - nmin, goal[1] - emin, c='yellow')

    # draw nodes
    for n1 in graph.nodes:
        plt.scatter(n1[1] - emin, n1[0] - nmin, c='red')
        
    # draw edges
    for (n1, n2) in graph.edges:
        plt.plot([n1[1] - emin, n2[1] - emin], [n1[0] - nmin, n2[0] - nmin], 'black')
        
    # TODO: add code to visualize the path
    path_pairs = zip(pruned_path[:-1], pruned_path[1:])
    for (n1, n2) in path_pairs:
        plt.plot([n1[1] - emin, n2[1] - emin], [n1[0] - nmin, n2[0] - nmin], 'blue')


    plt.xlabel('NORTH')
    plt.ylabel('EAST')

    plt.show()

def visualize_graph(data, start, goal, graph, valid_nodes):
    grid, _, _ = create_grid(data)     
    fig = plt.figure()

    plt.imshow(grid, cmap='Greys', origin='lower')

    nmin = np.min(data[:, 0])
    emin = np.min(data[:, 1])

    # draw edges
    for (n1, n2) in graph.edges:
        plt.plot([n1[1] - emin, n2[1] - emin], [n1[0] - nmin, n2[0] - nmin], 'black' , alpha=0.5)

    # draw all nodes
    for n1 in valid_nodes:
        plt.scatter(n1[1] - emin, n1[0] - nmin, c='blue')

    # draw connected nodes
    for n1 in graph.nodes:
        plt.scatter(n1[1] - emin, n1[0] - nmin, c='red')

    # draw start and goal
    plt.scatter(start[0], start[1], c='pink')
    plt.scatter(goal[0] - nmin, goal[1] - emin, c='yellow')

    plt.xlabel('NORTH')
    plt.ylabel('EAST')

    plt.show()




def create_grid(data):
    """
    Returns a grid representation of a 2D configuration space
    based on given obstacle data, drone altitude and safety distance
    arguments.
    """

    # minimum and maximum north coordinates
    north_min = np.floor(np.min(data[:, 0] - data[:, 3]))
    north_max = np.ceil(np.max(data[:, 0] + data[:, 3]))

    # minimum and maximum east coordinates
    east_min = np.floor(np.min(data[:, 1] - data[:, 4]))
    east_max = np.ceil(np.max(data[:, 1] + data[:, 4]))

    # given the minimum and maximum coordinates we can
    # calculate the size of the grid.
    north_size = int(np.ceil(north_max - north_min))
    east_size = int(np.ceil(east_max - east_min))

    # Initialize an empty grid
    grid = np.zeros((north_size, east_size))

    # Populate the grid with obstacles
    for i in range(data.shape[0]):
        north, east, alt, d_north, d_east, d_alt = data[i, :]
        if alt + d_alt + SAFETY_DISTANCE > TARGET_ALTITUDE:
            obstacle = [
                int(np.clip(north - d_north - SAFETY_DISTANCE - north_min, 0, north_size-1)),
                int(np.clip(north + d_north + SAFETY_DISTANCE - north_min, 0, north_size-1)),
                int(np.clip(east - d_east - SAFETY_DISTANCE - east_min, 0, east_size-1)),
                int(np.clip(east + d_east + SAFETY_DISTANCE - east_min, 0, east_size-1)),
            ]
            grid[obstacle[0]:obstacle[1]+1, obstacle[2]:obstacle[3]+1] = 1

    return grid, int(north_min), int(east_min)


# Assume all actions cost the same.
class Action(Enum):
    """
    An action is represented by a 3 element tuple.

    The first 2 values are the delta of the action relative
    to the current grid position. The third and final value
    is the cost of performing the action.
    """

    WEST = (0, -1, 1)
    EAST = (0, 1, 1)
    NORTH = (-1, 0, 1)
    SOUTH = (1, 0, 1)

    @property
    def cost(self):
        return self.value[2]

    @property
    def delta(self):
        return (self.value[0], self.value[1])


def valid_actions(grid, current_node):
    """
    Returns a list of valid actions given a grid and current node.
    """
    valid_actions = list(Action)
    n, m = grid.shape[0] - 1, grid.shape[1] - 1
    x, y = current_node

    # check if the node is off the grid or
    # it's an obstacle

    if x - 1 < 0 or grid[x - 1, y] == 1:
        valid_actions.remove(Action.NORTH)
    if x + 1 > n or grid[x + 1, y] == 1:
        valid_actions.remove(Action.SOUTH)
    if y - 1 < 0 or grid[x, y - 1] == 1:
        valid_actions.remove(Action.WEST)
    if y + 1 > m or grid[x, y + 1] == 1:
        valid_actions.remove(Action.EAST)

    return valid_actions

def a_star(grid, h, start, goal):

    path = []
    path_cost = 0
    queue = PriorityQueue()
    queue.put((0, start))
    visited = set(start)

    branch = {}
    found = False
    
    while not queue.empty():
        item = queue.get()
        current_node = item[1]
        if current_node == start:
            current_cost = 0.0
        else:              
            current_cost = branch[current_node][0]
            
        if current_node == goal:        
            print('Found a path.')
            found = True
            break
        else:
            for action in valid_actions(grid, current_node):
                # get the tuple representation
                da = action.delta
                next_node = (current_node[0] + da[0], current_node[1] + da[1])
                branch_cost = current_cost + action.cost
                queue_cost = branch_cost + h(next_node, goal)
                
                if next_node not in visited:                
                    visited.add(next_node)               
                    branch[next_node] = (branch_cost, current_node, action)
                    queue.put((queue_cost, next_node))
             
    if found:
        # retrace steps
        n = goal
        path_cost = branch[n][0]
        path.append(goal)
        while branch[n][1] != start:
            path.append(branch[n][1])
            n = branch[n][1]
        path.append(branch[n][1])
    else:
        print('**********************')
        print('Failed to find a path!')
        print('**********************') 
    return path[::-1], path_cost



def heuristic(position, goal_position):
    return np.linalg.norm(np.array(position) - np.array(goal_position))

