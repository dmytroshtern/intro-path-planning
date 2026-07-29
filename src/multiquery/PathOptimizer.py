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
        self.nodeLabelPrefix = "del"
        self.nodeCounter = 0
        self.requiredImprovement = 0.01

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

        improved = True

        while improved:
            improved = False

            candidates = self._rankPotentialShortcuts(graph, optimizedPairs, path, posList)

            for gain, i in candidates:

                # Direct shortcut
                startNode = path[i]
                endNode = path[i + 2]
                if not self._collisionChecker.lineInCollision(posList[startNode], posList[endNode]):
                    del path[i+1]
                    gu.addWeightedEdge(graph, startNode, endNode)
                    improved = True
                    break

                # Intermediate Point search
                if self._delTreeAlgorithm(graph, posList, path, i):
                    # Update needed since we added nodes
                    posList = nx.get_node_attributes(graph, 'pos')
                    improved = True
                    break



                optimizedPairs.append((startNode, endNode))


        return path, graph

    def _rankPotentialShortcuts(self, graph: nx.Graph, optimizedPairs: list[Any], path, posList):
        candidates = []
        currentPathLength = gu.pathLength(graph, path)
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
            if gain/currentPathLength > self.requiredImprovement:
                candidates.append(
                    (gain, i)
                )

        # Try biggest improvement first
        candidates.sort(reverse=True)
        return candidates

    def _delTreeAlgorithm(self, graph, posList, path, startIndex):
        success = False
        depth = 1
        startNode = path[startIndex]
        midNode = path[startIndex + 1]
        endNode = path[startIndex + 2]

        startPos = posList[startNode]
        endPos = posList[endNode]

        currentPathLength = gu.pathLength(graph, path)
        originalCost = gu.pathLength(graph, path[startIndex:startIndex + 3])

        while True:
            mid1Pos = self._findIntermediateCoordinates(posList, startNode, midNode, depth)
            mid2Pos = self._findIntermediateCoordinates(posList, midNode, endNode, depth)
            expectedCost = self.euclidean(startPos, mid1Pos) + self.euclidean(mid1Pos, mid2Pos) + self.euclidean(mid2Pos, endPos)
            gain = originalCost - expectedCost
            if gain/currentPathLength > self.requiredImprovement:
                if not self._collisionChecker.lineInCollision(mid1Pos, mid2Pos):
                    mid1Label = self._generateNodeLabel()
                    mid2Label = self._generateNodeLabel()
                    graph.add_node(mid1Label, pos=mid1Pos, color='lightgreen')
                    graph.add_node(mid2Label, pos=mid2Pos, color='lightgreen')
                    gu.addWeightedEdge(graph, startNode, mid1Label)
                    gu.addWeightedEdge(graph, mid1Label, mid2Label)
                    gu.addWeightedEdge(graph, mid2Label, endNode)
                    del path[startIndex+1]
                    path.insert(startIndex+1, mid1Label)
                    path.insert(startIndex+2, mid2Label)
                    success = True
                    break
            else:
                break
            depth += 1
        return success


    def _findIntermediateCoordinates(self, nodePositions, startNode, endNode, depth):
        return (np.asarray(nodePositions[startNode]) + np.asarray(nodePositions[endNode])) / pow(2, depth)

    def _generateNodeLabel(self):
        label = f"{self.nodeLabelPrefix}{self.nodeCounter}"
        self.nodeCounter += 1
        return label
