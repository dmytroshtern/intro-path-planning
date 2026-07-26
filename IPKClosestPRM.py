# coding: utf-8

"""
This code is part of the course "Introduction to robot path planning" (Author: Bjoern Hein).

License is based on Creative Commons: Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) (pls. check: http://creativecommons.org/licenses/by-nc/4.0/)
"""

import heapq

import IPPRMBase
from IPPerfMonitor import IPPerfMonitor
import networkx as nx
import numpy as np

from scipy.spatial import KDTree
from scipy.spatial.distance import euclidean


class KClosestPRM(IPPRMBase.PRMBase):

    def __init__(self, _collChecker):
        super(KClosestPRM, self).__init__(_collChecker)
        self.graph = nx.Graph()

    @IPPerfMonitor
    def _inSameConnectedComponent(self, node1, node2):
        """Check whether two nodes are part of the same connected component."""
        for connectedComponent in nx.connected_components(self.graph):
            if (node1 in connectedComponent) and (node2 in connectedComponent):
                return True
        return False

    @IPPerfMonitor
    def _nearestNeighboursK(self, pos, k, excludeNode=None):
        """Find the k nearest graph nodes by brute force."""
        heap = list()
        for index, node in enumerate(self.graph.nodes(data=True)):
            if node[0] == excludeNode:
                continue
            heapq.heappush(heap, (euclidean(node[1]['pos'], pos), index, node))

        result = list()
        while len(heap) > 0 and len(result) < k:
            result.append(heapq.heappop(heap)[2])
        return result

    @IPPerfMonitor
    def _nearestNeighboursKSorted(self, pos, k, excludeNode=None):
        """Find the k nearest graph nodes by sorting all current nodes."""
        nodeList = [
            node for node in self.graph.nodes(data=True)
            if node[0] != excludeNode
        ]
        sortKey = lambda data: euclidean(pos, data[1]["pos"])
        return sorted(nodeList, key=sortKey)[:k]

    @IPPerfMonitor
    def _learnRoadmapNearestNeighbourKDTree(self, k, numNodes, collisionCheckingSteps=40):
        """Generate a roadmap and connect every node to its k nearest neighbours using a KDTree."""
        for nodeID in range(numNodes):
            self.graph.add_node(nodeID, pos=self._getRandomFreePosition())

        nodeList = list(self.graph.nodes())
        posList = [self.graph.nodes[node]["pos"] for node in nodeList]
        kdTree = KDTree(posList)

        for node in nodeList:
            queryK = min(k + 1, len(nodeList))
            distances, indices = kdTree.query(self.graph.nodes[node]['pos'], queryK)
            for dataIndex in np.atleast_1d(indices):
                neighbourNode = nodeList[int(dataIndex)]
                if neighbourNode == node:
                    continue
                if self._inSameConnectedComponent(node, neighbourNode):
                    continue

                nodePos = self.graph.nodes[node]['pos']
                neighbourPos = self.graph.nodes[neighbourNode]['pos']
                if not self._collisionChecker.lineInCollision(
                    nodePos, neighbourPos, steps=collisionCheckingSteps
                ):
                    self.graph.add_edge(
                        node,
                        neighbourNode,
                        weight=euclidean(nodePos, neighbourPos),
                    )

    @IPPerfMonitor
    def _learnRoadmapNearestNeighbour(self, k, numNodes, collisionCheckingSteps=40):
        """Generate a roadmap incrementally and connect every new node to up to k nearest neighbours."""
        nodeID = 1
        while nodeID <= numNodes:
            newNodePos = self._getRandomFreePosition()
            self.graph.add_node(nodeID, pos=newNodePos)

            result = self._nearestNeighboursKSorted(newNodePos, k, excludeNode=nodeID)

            for data in result:
                neighbourNode = data[0]
                neighbourPos = data[1]['pos']
                if self._inSameConnectedComponent(nodeID, neighbourNode):
                    continue

                if not self._collisionChecker.lineInCollision(
                    newNodePos, neighbourPos, steps=collisionCheckingSteps
                ):
                    self.graph.add_edge(
                        nodeID,
                        neighbourNode,
                        weight=euclidean(newNodePos, neighbourPos),
                    )

            nodeID += 1

    @IPPerfMonitor
    def planPath(self, startList, goalList, config):
        """
        Args:
            startList (list): list of possible start configurations
            goalList (list): list of possible goal configurations
            config (dict): planner configuration

        Example:
            config["k"] = 5
            config["numNodes"] = 300
            config["collisionCheckingSteps"] = 40
            config["useKDTree"] = True
        """
        self.graph.clear()
        collisionCheckingSteps = config.get("collisionCheckingSteps", 40)

        checkedStartList, checkedGoalList = self._checkStartGoal(startList, goalList)

        if config.get("useKDTree", True):
            self._learnRoadmapNearestNeighbourKDTree(
                config["k"],
                config["numNodes"],
                collisionCheckingSteps=collisionCheckingSteps,
            )
        else:
            self._learnRoadmapNearestNeighbour(
                config["k"],
                config["numNodes"],
                collisionCheckingSteps=collisionCheckingSteps,
            )

        result = self._nearestNeighboursK(checkedStartList[0], config["k"])
        for node in result:
            neighbourPos = node[1]['pos']
            if not self._collisionChecker.lineInCollision(
                checkedStartList[0], neighbourPos, steps=collisionCheckingSteps
            ):
                self.graph.add_node("start", pos=checkedStartList[0], color='lightgreen')
                self.graph.add_edge(
                    "start",
                    node[0],
                    weight=euclidean(checkedStartList[0], neighbourPos),
                )
                break

        result = self._nearestNeighboursK(checkedGoalList[0], config["k"])
        for node in result:
            neighbourPos = node[1]['pos']
            if not self._collisionChecker.lineInCollision(
                checkedGoalList[0], neighbourPos, steps=collisionCheckingSteps
            ):
                self.graph.add_node("goal", pos=checkedGoalList[0], color='lightgreen')
                self.graph.add_edge(
                    "goal",
                    node[0],
                    weight=euclidean(checkedGoalList[0], neighbourPos),
                )
                break

        try:
            path = nx.shortest_path(self.graph, "start", "goal", weight="weight")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []
        return path
