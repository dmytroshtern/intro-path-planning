from timeit import repeat

from multiquery import VisibilityStatsHandler
from IPPerfMonitor import IPPerfMonitor
from notebooks.IPEnvironment import CollisionChecker
import networkx as nx
import random
from typing import Any
import numpy as np
from math import dist
import multiquery.GraphUtility as gu

class RoadmapOptimizer:

    def __init__(self, collisionChecker: CollisionChecker):
        self._collisionChecker = collisionChecker
        self.nodeLabelPrefix = "opt"
        self.nodeCounter = 0

    @IPPerfMonitor
    def optimizeRoadmap(self, originalGraph: nx.Graph):
        graph = originalGraph.copy()
        self._probabilisticShortcuts(graph, 200)
        return graph

    def _probabilisticShortcuts(self, graph: nx.Graph, tries: int):
        nodeLabels = list(graph.nodes())
        posList = nx.get_node_attributes(graph, 'pos')
        nonGuardNode = [node for node in nodeLabels if not graph.nodes[node]['nodeType'] == 'Guard']
        if len(nonGuardNode) == 0 or len(nodeLabels) < 3:
            return

        for i in range(tries):
            startNode = random.choice(nodeLabels)
            #By construction guards can not see each other - don't build such pairs, also exclude self loops and already existing edges
            possible_end_nodes = [
                n for n in (nonGuardNode if graph.nodes[startNode]['nodeType'] == 'Guard' else nodeLabels)
                if n != startNode and not graph.has_edge(startNode, n)
            ]

            if possible_end_nodes:
                endNode = random.choice(possible_end_nodes)
            else:
                continue

            if not self._collisionChecker.lineInCollision(posList[startNode], posList[endNode]):
                gu.addWeightedEdge(graph, startNode, endNode, color = "red")