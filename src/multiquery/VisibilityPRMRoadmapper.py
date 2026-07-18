# coding: utf-8

"""
This code is part of the course "Introduction to robot path planning" (Author: Bjoern Hein).

License is based on Creative Commons: Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) (pls. check: http://creativecommons.org/licenses/by-nc/4.0/)
"""

from IPPRMBase import PRMBase
import networkx as nx
from scipy.spatial import cKDTree
from IPPerfMonitor import IPPerfMonitor
import numpy as np
from IPVisibilityPRM import VisPRM
from math import dist

class VisibilityStatsHandler():
    
    def __init__(self):
        self.graph = nx.Graph()
        
    def addNodeAtPos(self,nodeNumber,pos):
        self.graph.add_node(nodeNumber, pos=pos, color='yellow')
        return
    
    def addVisTest(self,fr,to):
        self.graph.add_edge(fr, to)
        return
        
class VisibilityPRMRoadmapper(PRMBase):
    """Class implements an simplified version of a visibility PRM"""

    def __init__(self, _collChecker, _statsHandler = None):
        super(VisibilityPRMRoadmapper, self).__init__(_collChecker)
        self.graph = nx.Graph()
        self.statsHandler = VisibilityStatsHandler() # not yet fully customizable (s. parameters of constructors)
                
    def _isVisible(self, pos, guardPos):
        return not self._collisionChecker.lineInCollision(pos, guardPos)
        
    def _addWeightedEdge(self, start, goal):
        self.graph.add_edge(start, goal, weight=dist(self.graph.nodes[start]['pos'], self.graph.nodes[goal]['pos']))

    @IPPerfMonitor
    def _learnRoadmap(self, ntry):

        nodeNumber = 0
        currTry = 0
        while currTry < ntry:
            #print currTry
            # select a random  free position
            q_pos = self._getRandomFreePosition()
            if self.statsHandler:
                self.statsHandler.addNodeAtPos(nodeNumber, q_pos)
           
            g_vis = None
        
            # every connected component represents one guard
            merged = False
            for comp in nx.connected_components(self.graph): # Impliciteley represents G_vis
                found = False
                merged = False
                for g in comp: # connected components consists of guards and connection: only test nodes of type 'Guards'
                    if self.graph.nodes()[g]['nodeType'] == 'Guard':
                        if self.statsHandler:
                            self.statsHandler.addVisTest(nodeNumber, g)
                        if self._isVisible(q_pos, self.graph.nodes()[g]['pos']):
                            found = True
                            if g_vis == None:
                                g_vis = g
                            else:
                                self.graph.add_node(nodeNumber, pos = q_pos, color='lightblue', nodeType = 'Connection')
                                self._addWeightedEdge(nodeNumber, g)
                                self._addWeightedEdge(nodeNumber, g_vis)
                                merged = True
                        # break, if node was visible,because visibility from one node of the guard is sufficient...
                        if found == True: break;
                # break, if connection was found. Reason: computed connected components (comp) are not correct any more, 
                # they've changed because of merging
                if merged == True: # how  does it change the behaviour? What has to be done to keep the original behaviour?
                    break;                    

            if (merged==False) and (g_vis == None):
                self.graph.add_node(nodeNumber, pos = q_pos, color='red', nodeType = 'Guard')
                #print "ADDED Guard ", nodeNumber
                currTry = 0
            else:
                currTry += 1

            nodeNumber += 1
        
    
    def _addNodeToRoadmap(self, posList, kdTree, node_pos, label, multipleConnections = False):
        '''
        optimizations
        1. allow connection between start/goal nodes -> add to KD-tree
        '''
        
        connectionCandidates = kdTree.query(node_pos,k=5)
        result = False
        for connectionCandidate in connectionCandidates[1]:
            try :
                if self._isVisible(node_pos, (self.graph.nodes[list(posList.keys())[connectionCandidate]]['pos'])):
                    self.graph.add_node(label, pos=node_pos, color='lightgreen')
                    self._addWeightedEdge(label, list(posList.keys())[connectionCandidate])
                    result = True
                    if not multipleConnections:
                        break
            except Exception as e:
                    raise ValueError(f"{connectionCandidates[1]}, {connectionCandidate}, {list(posList.keys())}")
        return result
        
        
    @IPPerfMonitor
    def createNewRoadmapGraph(self, startList, goalList, config):
        checkedStartList, checkedGoalList = self._checkStartGoal(startList,goalList)
        self.graph.clear()
        self._learnRoadmap(config["ntry"])
        
        posList = nx.get_node_attributes(self.graph,'pos')
        kdTree = cKDTree(list(posList.values()))
        if not self._addNodeToRoadmap(posList, kdTree, checkedStartList[0], "start", config["mConnections"]):
            return None
        for index, goal in enumerate(checkedGoalList):
            if config["directConnections"]:
                posList = nx.get_node_attributes(self.graph,'pos')
                kdTree = cKDTree(list(posList.values()))
            if not self._addNodeToRoadmap(posList, kdTree, goal, f"goal_{index}", config["mConnections"]):
                return None
        return self.graph