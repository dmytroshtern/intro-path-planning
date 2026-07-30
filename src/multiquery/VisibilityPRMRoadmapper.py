# coding: utf-8

"""
This code is part of the course "Introduction to robot path planning" (Author: Bjoern Hein).

License is based on Creative Commons: Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) (pls. check: http://creativecommons.org/licenses/by-nc/4.0/)
"""

from IPPRMBase import PRMBase
import networkx as nx
from . import GraphUtility as gu
from scipy.spatial import cKDTree
from IPPerfMonitor import IPPerfMonitor
from math import dist, inf

from .VisibilityStatsHandler import VisibilityStatsHandler


class VisibilityPRMRoadmapper(PRMBase):
    """Class implements an simplified version of a visibility PRM"""

    def __init__(self, _collChecker):
        super(VisibilityPRMRoadmapper, self).__init__(_collChecker)
        self.nodeNumber = 0

    def _isVisible(self, pos, goalPos, maxVisibility = inf):
        return dist(pos, goalPos) < maxVisibility and not self._collisionChecker.lineInCollision(pos, goalPos)

    @IPPerfMonitor
    def learnRoadmap(self, config, visibilityStatsHandler : VisibilityStatsHandler, maxVisibility = inf):
        return self.refineRoadmap(config, visibilityStatsHandler, nx.Graph(), maxVisibility)


    def refineRoadmap(self, config, visibilityStatsHandler : VisibilityStatsHandler,  graph, maxVisibility = inf):
        ntry = config.get("ntry", 40)
        currTry = 0
        while currTry < ntry:
            guardAdded = self.integrateNode(graph, visibilityStatsHandler, maxVisibility)
            if guardAdded:
                currTry = 0
            else:
                currTry += 1
            self.nodeNumber += 1

        return graph

    def integrateNode(self, graph: nx.Graph, visibilityStatsHandler, maxVisibility = inf):
        g_vis = None
        guardAdded = False
        # select a random  free position
        q_pos = self._getRandomFreePosition()
        visibilityStatsHandler.addNodeAtPos(self.nodeNumber, q_pos)
        merged = False
        for comp in nx.connected_components(graph):  # Impliciteley represents G_vis
            found = False
            #merged = False
            for g in comp:  # connected components consists of guards and connection: only test nodes of type 'Guards'
                if graph.nodes()[g]['nodeType'] == 'Guard':
                    visibilityStatsHandler.addVisTest(self.nodeNumber, g)
                    if self._isVisible(q_pos, graph.nodes()[g]['pos'], maxVisibility):
                        found = True
                        if g_vis is None:
                            g_vis = g
                        else:
                            graph.add_node(f"{self.nodeNumber}", pos=q_pos, color='lightblue', nodeType='Connection')
                            gu.addWeightedEdge(graph, f"{self.nodeNumber}", f"{g}")
                            gu.addWeightedEdge(graph, f"{self.nodeNumber}", f"{g_vis}")
                            merged = True
                    # break, if node was visible,because visibility from one node of the guard is sufficient...
                    if found == True: break;
            # break, if connection was found. Reason: computed connected components (comp) are not correct any more,
            # they've changed because of merging
            if merged == True:  # how  does it change the behaviour? What has to be done to keep the original behaviour?
                break

        if (merged == False) and (g_vis is None):
            graph.add_node(f"{self.nodeNumber}", pos=q_pos, color='red', nodeType='Guard')
            guardAdded = True

        return guardAdded


    def _addNodeToRoadmap(self, graph, posList, kdTree, node_pos, label, multipleConnections=False):
        graph.add_node(label, nodeType = "Guard", pos=node_pos)
        connectionCandidates = kdTree.query(node_pos, k=15)
        result = False
        for connectionCandidate in connectionCandidates[1]:
            if connectionCandidate == kdTree.n:
                break

            if self._isVisible(node_pos, (graph.nodes[list(posList.keys())[connectionCandidate]]['pos'])):
                gu.addWeightedEdge(graph, label, list(posList.keys())[connectionCandidate])
                result = True
                if not multipleConnections:
                    break
        return result


    def addStartAndGoalsToRoadmap(self, graph, startList, goalList, optimize):
        posList = nx.get_node_attributes(graph, 'pos')
        kdTree = cKDTree(list(posList.values()))
        self._addNodeToRoadmap(graph, posList, kdTree, startList[0], "S", optimize)
        for index, goal in enumerate(goalList):
            if optimize:
                posList = nx.get_node_attributes(graph, 'pos')
                kdTree = cKDTree(list(posList.values()))
            self._addNodeToRoadmap(graph, posList, kdTree, goal, f"G{index}", optimize)
        return graph