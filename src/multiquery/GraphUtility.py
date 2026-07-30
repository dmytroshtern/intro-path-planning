from math import dist

def addWeightedEdge(graph, start, goal, color = "black"):
    """Add an edge with Euclidean weight between two nodes that already have positions."""
    graph.add_edge(start, goal, color = color, weight=dist(graph.nodes[start]['pos'], graph.nodes[goal]['pos']))

def pathLength(graph, path):
    """Return the total weighted length of ``path`` in ``graph``."""
    return sum(
        graph[u][v].get("weight", 1.0)
        for u, v in zip(path[:-1], path[1:])
    )
