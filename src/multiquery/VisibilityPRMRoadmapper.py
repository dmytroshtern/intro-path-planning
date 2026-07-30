# coding: utf-8

"""
This code is part of the course "Introduction to robot path planning" (Author: Bjoern Hein).

License is based on Creative Commons: Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) (pls. check: http://creativecommons.org/licenses/by-nc/4.0/)
"""
from typing import Any

from IPPRMBase import PRMBase
import networkx as nx
import multiquery.GraphUtility as gu
from scipy.spatial import cKDTree
from IPPerfMonitor import IPPerfMonitor
import numpy as np
from IPVisibilityPRM import VisPRM
from math import dist, inf

from multiquery import VisibilityStatsHandler


class VisibilityPRMRoadmapper(PRMBase):
    """Class implements an simplified version of a visibility PRM"""

    def __init__(self, _collChecker):
        super(VisibilityPRMRoadmapper, self).__init__(_collChecker)
                
    def _isVisible(self, pos, goalPos, maxVisibility = inf):
        return dist(pos, goalPos) < maxVisibility and not self._collisionChecker.lineInCollision(pos, goalPos)

    @IPPerfMonitor
    def learnRoadmap(self, config, visibilityStatsHandler : VisibilityStatsHandler, maxVisibility = inf):
        return self.refineRoadmap(config, visibilityStatsHandler, nx.Graph(), 0, maxVisibility)


    def refineRoadmap(self, config, visibilityStatsHandler : VisibilityStatsHandler,  graph, startNumber:int, maxVisibility = inf):
        ntry = config.get("ntry", 40)
        nodeNumber = startNumber
        currTry = 0
        while currTry < ntry:
            guardAdded = self.integrateNode(graph, nodeNumber, visibilityStatsHandler, maxVisibility)
            if guardAdded:
                currTry = 0
            else:
                currTry += 1
            nodeNumber += 1


        return graph, nodeNumber

    def integrateNode(self, graph: nx.Graph, nodeLabel: int, visibilityStatsHandler, maxVisibility = inf):
        g_vis = None
        guardAdded = False
        # select a random  free position
        q_pos = self._getRandomFreePosition()
        visibilityStatsHandler.addNodeAtPos(nodeLabel, q_pos)
        merged = False
        for comp in nx.connected_components(graph):  # Impliciteley represents G_vis
            found = False
            #merged = False
            for g in comp:  # connected components consists of guards and connection: only test nodes of type 'Guards'
                if graph.nodes()[g]['nodeType'] == 'Guard':
                    visibilityStatsHandler.addVisTest(nodeLabel, g)
                    if self._isVisible(q_pos, graph.nodes()[g]['pos'], maxVisibility):
                        found = True
                        if g_vis == None:
                            g_vis = g
                        else:
                            graph.add_node(nodeLabel, pos=q_pos, color='lightblue', nodeType='Connection')
                            gu.addWeightedEdge(graph, nodeLabel, g)
                            gu.addWeightedEdge(graph, nodeLabel, g_vis)
                            merged = True
                    # break, if node was visible,because visibility from one node of the guard is sufficient...
                    if found == True: break;
            # break, if connection was found. Reason: computed connected components (comp) are not correct any more,
            # they've changed because of merging
            if merged == True:  # how  does it change the behaviour? What has to be done to keep the original behaviour?
                break

        if (merged == False) and (g_vis is None):
            graph.add_node(nodeLabel, pos=q_pos, color='red', nodeType='Guard')
            guardAdded = True

        return guardAdded