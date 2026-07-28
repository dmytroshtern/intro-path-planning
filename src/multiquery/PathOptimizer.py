from multiquery import VisibilityStatsHandler
from IPPerfMonitor import IPPerfMonitor
from notebooks.IPEnvironment import CollisionChecker
import networkx as nx
from typing import Any
import numpy as np
from math import dist
import multiquery.GraphUtility as gu


class PathOptimizer:

    def __init__(self, collisionChecker: CollisionChecker):
        self._collisionChecker = collisionChecker

    def euclidean(self, a, b):
        return np.linalg.norm(np.array(a) - np.array(b))

    @IPPerfMonitor
    def shortcut_path(self, originalGraph : nx.Graph, originalPath):
        """
        Try to shorten a PRM path.

        G              : networkx graph
        path           : list of node names
        """
        graph = originalGraph.copy()
        path = originalPath.copy()
        optimizedPairs = []
        posList = nx.get_node_attributes(graph, 'pos')

        initialPathLength = gu.pathLength(graph, path)
        oldPathLength = initialPathLength
        improved = True

        while improved:
            improved = False

            candidates = self.findPossibleShorcuts(graph, optimizedPairs, path, posList)

            for gain, i in candidates:
                if gain <= 0:
                    continue

                a = path[i]
                b = path[i+2]

                # Direct shortcut
                if not self._collisionChecker.lineInCollision(posList[a], posList[b]):
                    path = path[:i + 1] + path[i+2:]
                    gu.addWeightedEdge(graph, a, b)
                    improved = True
                    break

                # Intermediate Point search


                optimizedPairs.append((a, b))


        return path, graph

    def findPossibleShorcuts(self, graph: nx.Graph, optimizedPairs: list[Any], path, posList: dict[_Node, Any]):
        candidates = []
        for i in range(len(path) - 2):
            # Don't skip a goal
            if path[i + 1].startswith("G"):
                continue

            if (path[i], path[i + 2]) in optimizedPairs:
                continue

            old_cost = gu.pathLength(graph, path[i:i+3])

            direct_cost = self.euclidean(
                posList[path[i]],
                posList[path[i + 2]]
            )

            gain = old_cost - direct_cost

            candidates.append(
                (gain, i)
            )

        # Try biggest improvement first
        candidates.sort(reverse=True)
        return candidates
