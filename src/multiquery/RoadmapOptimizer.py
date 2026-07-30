from os.path import samefile
from timeit import repeat

from IPPRMBase import PRMBase
from multiquery import VisibilityStatsHandler
from IPPerfMonitor import IPPerfMonitor
from notebooks.IPEnvironment import CollisionChecker
import networkx as nx
import random
from typing import Any
import numpy as np
from math import dist
import multiquery.GraphUtility as gu

class RoadmapOptimizer(PRMBase):

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

    def bridgeSamples(self, maxSamples):
        samples = 0
        validMidpoints = []
        while samples < maxSamples:
            obs1, samples = self._getRandomOccupiedPosition(samples, maxSamples)
            obs2, samples = self._getRandomOccupiedPosition(samples, maxSamples)
            if obs1 is None or obs2 is None:
                break
            mid = self._findMidpoint(obs1, obs2)
            if not self._collisionChecker.pointInCollision(mid):
                validMidpoints.append(mid)
        return validMidpoints

    def _getRandomOccupiedPosition(self,samples, maxSamples):
        while samples < maxSamples:
            samples += 1
            pos = np.asarray(self._getRandomPosition())
            if self._collisionChecker.pointInCollision(pos):
                return pos, samples
        return None, samples

    def _findMidpoint(self, startPos, goalPos):
        goalToStart = -goalPos + startPos
        return goalPos + goalToStart/2