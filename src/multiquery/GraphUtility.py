from math import dist

def addWeightedEdge(graph, start, goal):
    graph.add_edge(start, goal, weight=dist(graph.nodes[start]['pos'], graph.nodes[goal]['pos']))

def pathLength(graph, path):
    return sum(
        graph[u][v].get("weight", 1.0)
        for u, v in zip(path[:-1], path[1:])
    )