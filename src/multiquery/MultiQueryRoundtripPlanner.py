import networkx as nx
import traceback
import numpy as np
from typing import Type, Any, List
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree


from IPPRMBase import PRMBase
from IPPerfMonitor import IPPerfMonitor
from IPEnvironment import CollisionChecker

import multiquery.GraphUtility as gu
from multiquery.PathOptimizer import PathOptimizer
from multiquery.VisibilityPRMRoadmapper import VisibilityPRMRoadmapper
from multiquery.VisibilityStatsHandler import VisibilityStatsHandler
from notebooks.IPPlanerBase import PlanerBase
from src.roundtrip_algorithm.result import (
    roundtrip_failure,
    roundtrip_success,
)

class MultiQueryRoundtripPlanner(PlanerBase):

    def __init__(self, roadmapPlanner: VisibilityPRMRoadmapper, collisionChecker: CollisionChecker):
        #assert hasattr(roadmapPlannerClass, "createNewRoadmapGraph"), "roadmapPlannerClass must have a method called 'createNewRoadmapGraph'"
        super().__init__(collisionChecker)
        self.baseRoadmap = nx.Graph() # base visibility PRM roadmap to be reused
        self.graph = nx.Graph() # graph to store all paths between start and goal nodes
        self._roadmapPlanner = roadmapPlanner
        self.statsHandler = VisibilityStatsHandler()
        self._collisionChecker = collisionChecker
        self._pathOptimizer = PathOptimizer(collisionChecker)

    @IPPerfMonitor
    def planPath(self, startList: List[List[Any]], goalList: List[List[Any]], config) -> dict[str, Any]:
        """
        Plans a roundtrip path that visits the first start and goal nodes.
        Args:
            startList (array): start position in planning space. E.g. [[1,2]]
            goalList (array) : goal position in planning space. E.g. [[3,4]]
            config (dict): dictionary with the needed information about the configuration options

        Returns:
            List[List[Any]]: A list representing the roundtrip path visiting all goals and returning to the start.
        """
        solutions = {}
        checkedStartList, checkedGoalList = self._checkStartGoal(startList, goalList)
        if not len(checkedGoalList) is len(goalList):
            return roundtrip_failure(
                reason=f"GoalList has collisions {checkedGoalList} and {goalList}",
                pairwise_results=None,
                failed_pairs=None,
                metadata={}
            )

        self.baseRoadmap = self._roadmapPlanner.learnRoadmap(config, self.statsHandler)
        # Conversion of node names to string for consistency
        self.baseRoadmap = nx.relabel_nodes(self.baseRoadmap, str, copy=False)

        roadmapWithSTartAndGoals = self.addStartGoalToRoadmap(self.baseRoadmap.copy(), checkedStartList, checkedGoalList, config)



        goalNodes = [node for node in roadmapWithSTartAndGoals.nodes if node.startswith("G")]
        
        try:            
            tsgSolution = list(nx.algorithms.approximation.traveling_salesman_problem(
                roadmapWithSTartAndGoals,
                nodes = ["S"] + goalNodes,  # Include start node in the TSP solver
                cycle = True
            ))
            tsgCost = gu.pathLength(roadmapWithSTartAndGoals, tsgSolution)
            solutions["TSG"] = (tsgSolution, tsgCost, roadmapWithSTartAndGoals)


        except Exception as e:
            self.graph = roadmapWithSTartAndGoals
            return roundtrip_failure(
                reason=f"TSG solver failed: {e}",
                pairwise_results=None,
                failed_pairs=None,
                metadata={
                    "stage": "TSG solver"
                }
            )

        if True:
            shortcutSolution, sortcutGraph = self._pathOptimizer.shortcut_path(roadmapWithSTartAndGoals, tsgSolution)
            shortcutCost = gu.pathLength(sortcutGraph, shortcutSolution)
            solutions["Shortcut"] = (shortcutSolution, shortcutCost, sortcutGraph)

        solution = solutions["Shortcut"]
        usedSolution = solution[0]
        usedCost = solution[1]
        self.graph = solution[2]
        visitOrder = self.trimPathToKnownNodes(usedSolution)
        result = roundtrip_success(
            visit_order=visitOrder,
            used_pairs=self.makeUsedPairs(visitOrder),
            final_path_configs=usedSolution,
            tour_cost=usedCost,
            pairwise_results={},
            failed_pairs=None,
            metadata={}
        )
        return result

    def trimPathToKnownNodes(self, path) -> list[str]:
        return [node for node in path if node.startswith("G") or node.startswith("S")]

    def makeUsedPairs(self, visitOrder: list[str]) -> list[tuple[str, str]]:
        return list(zip(visitOrder, visitOrder[1:]))



    def _addNodeToRoadmap(self, graph, posList, kdTree, node_pos, label, multipleConnections=False):
        '''
        optimizations
        1. allow connection between start/goal nodes -> add to KD-tree
        '''
        graph.add_node(label, pos=node_pos, color='lightgreen')
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

    @IPPerfMonitor
    def addStartGoalToRoadmap(self, graph, startList, goalList, config):
        posList = nx.get_node_attributes(graph, 'pos')
        kdTree = cKDTree(list(posList.values()))
        self._addNodeToRoadmap(graph, posList, kdTree, startList[0], "S", config["mConnections"])
        for index, goal in enumerate(goalList):
            if config["directConnections"]:
                posList = nx.get_node_attributes(graph, 'pos')
                kdTree = cKDTree(list(posList.values()))
            self._addNodeToRoadmap(graph, posList, kdTree, goal, f"G{index}", config["mConnections"])
        return graph



    def _isVisible(self, pos, guardPos):
        return not self._collisionChecker.lineInCollision(pos, guardPos)
