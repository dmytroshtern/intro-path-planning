from cProfile import label

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
from multiquery.RoadmapOptimizer import RoadmapOptimizer
from multiquery.VisibilityPRMRoadmapper import VisibilityPRMRoadmapper
from multiquery.VisibilityStatsHandler import VisibilityStatsHandler
from notebooks.IPPlanerBase import PlanerBase
from src.roundtrip_algorithm.result import (
    roundtrip_failure,
    roundtrip_success,
)

class MultiQueryRoundtripPlanner(PlanerBase):

    def __init__(self, collisionChecker: CollisionChecker):
        #assert hasattr(roadmapPlannerClass, "createNewRoadmapGraph"), "roadmapPlannerClass must have a method called 'createNewRoadmapGraph'"
        super().__init__(collisionChecker) # base visibility PRM roadmap to be reused
        self.graph = nx.Graph() # graph to store all paths between start and goal nodes
        self._roadmapPlanner = VisibilityPRMRoadmapper(collisionChecker)
        self.statsHandler = VisibilityStatsHandler()
        self._collisionChecker = collisionChecker
        self._pathOptimizer = PathOptimizer(collisionChecker)
        self._roadmapOptimizer = RoadmapOptimizer(collisionChecker)

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
        checkedStartList, checkedGoalList = self._checkStartGoal(startList, goalList)
        if len(checkedStartList) == 0 or not checkedStartList[0] == startList[0]:
            return roundtrip_failure(
                reason=f"StartList has collisions {checkedStartList} and {startList}",
                pairwise_results=None,
                failed_pairs=None,
                metadata={}
            )
        if not len(checkedGoalList) is len(goalList):
            return roundtrip_failure(
                reason=f"GoalList has collisions {checkedGoalList} and {goalList}",
                pairwise_results=None,
                failed_pairs=None,
                metadata={}
            )

        baseRoadmap, bs = self._roadmapPlanner.learnRoadmap(config, self.statsHandler)
        #baseRoadmap, bs = self._roadmapPlanner.refineRoadmap(config, self.statsHandler, baseRoadmap, bs, 3)
        #baseRoadmap, bs = self._roadmapPlanner.refineRoadmap(config, self.statsHandler, baseRoadmap, bs)
        # Conversion of node names to string for consistency
        baseRoadmap = nx.relabel_nodes(baseRoadmap, str, copy=False)

        if config.get("optimizeRoadmap", False):
            usedRoadmapWithGoals = self.addStartAndGoalsToRoadmap(baseRoadmap.copy(), checkedStartList,
                                                                  checkedGoalList, True)
            usedRoadmapWithGoals = self._roadmapOptimizer.optimizeRoadmap(usedRoadmapWithGoals)
        else:
            usedRoadmapWithGoals = self.addStartAndGoalsToRoadmap(baseRoadmap.copy(), checkedStartList,
                                                                  checkedGoalList, False)

        try:
            tsgSolution = self._findTSGSolution(usedRoadmapWithGoals)

        except Exception as e:
            self.graph = usedRoadmapWithGoals
            return roundtrip_failure(
                reason=f"TSG solver failed: {e}",
                pairwise_results=None,
                failed_pairs=None,
                metadata={
                    "stage": "TSG solver"
                }
            )
        if config.get("optimizePath", False):
            shortcutSolution, shortcutGraph = self._pathOptimizer.shortcut_path(usedRoadmapWithGoals, tsgSolution)
            usedSolution = shortcutSolution
            usedCost = gu.pathLength(shortcutGraph, shortcutSolution)
            self.graph = shortcutGraph
        else:
            usedSolution = tsgSolution
            usedCost = gu.pathLength(usedRoadmapWithGoals, tsgSolution)
            self.graph = usedRoadmapWithGoals

        visitOrder = self.trimPathToKnownNodes(usedSolution)
        return roundtrip_success(
            visit_order=visitOrder,
            used_pairs=self._makeUsedPairs(visitOrder),
            final_path_configs=usedSolution,
            tour_cost=usedCost,
            pairwise_results={},
            failed_pairs=None,
            metadata={}
        )

    @IPPerfMonitor
    def planPathBenchmarking(self, startList: List[List[Any]], goalList: List[List[Any]], config) -> dict[str, Any]:
        """
        Plans a roundtrip path that visits the first start and goal nodes.
        Args:
            startList (array): start position in planning space. E.g. [[1,2]]
            goalList (array) : goal position in planning space. E.g. [[3,4]]
            config (dict): dictionary with the needed information about the configuration options

        Returns:
            List[List[Any]]: A list representing the roundtrip path visiting all goals and returning to the start.
        """
        checkedStartList, checkedGoalList = self._checkStartGoal(startList, goalList)
        if len(checkedStartList) == 0 or not checkedStartList[0] == startList[0]:
            return {}
        if not len(checkedGoalList) is len(goalList):
            return {}

        baseRoadmap, bs = self._roadmapPlanner.learnRoadmap(config, self.statsHandler)
        # baseRoadmap, bs = self._roadmapPlanner.refineRoadmap(config, self.statsHandler, baseRoadmap, bs, 3)
        # baseRoadmap, bs = self._roadmapPlanner.refineRoadmap(config, self.statsHandler, baseRoadmap, bs)
        # Conversion of node names to string for consistency
        baseRoadmap = nx.relabel_nodes(baseRoadmap, str, copy=False)

        solutions = {}

        basicRoadmapWithStartAndGoals = self.addStartAndGoalsToRoadmap(baseRoadmap.copy(), checkedStartList,
                                                                       checkedGoalList, False)
        try:
            # Create basic roadmap TSG solution
            basicRoadmapTSGSolution = self._findTSGSolution(basicRoadmapWithStartAndGoals)
            self._addSolution(solutions, basicRoadmapTSGSolution, basicRoadmapWithStartAndGoals,
                              "basicRoadmapTSGSolution")

            # Create path optimized solution
            basicRoadmapShortcutSolution, basicRoadmapShortcutGraph = self._pathOptimizer.shortcut_path(
                basicRoadmapWithStartAndGoals,
                basicRoadmapTSGSolution)
            self._addSolution(solutions, basicRoadmapShortcutSolution, basicRoadmapShortcutGraph,
                              "basicRoadmapShortcutSolution")
        except Exception as e:
            self._addSolution(solutions, [], basicRoadmapWithStartAndGoals,
                              "basicRoadmapTSGSolution")
            self._addSolution(solutions, [], basicRoadmapWithStartAndGoals,
                              "basicRoadmapShortcutSolution")

        optimizedRoadmapWithStartAndGoals = self.addStartAndGoalsToRoadmap(baseRoadmap.copy(), checkedStartList,
                                                                           checkedGoalList, True)
        optimizedRoadmapWithStartAndGoals = self._roadmapOptimizer.optimizeRoadmap(
            optimizedRoadmapWithStartAndGoals)
        try:
            # Create optimized roadmap TSG solution
            optimizedRoadmapTSGSolution = self._findTSGSolution(optimizedRoadmapWithStartAndGoals)
            self._addSolution(solutions, optimizedRoadmapTSGSolution, optimizedRoadmapWithStartAndGoals,
                              "optimizedRoadmapTSGSolution")
        except Exception as e:
            # No solution for optimized Roadmap -> No solution at all expected
            self.graph = optimizedRoadmapWithStartAndGoals
            self._addSolution(solutions, [], optimizedRoadmapWithStartAndGoals,
                              "optimizedRoadmapTSGSolution")
            self._addSolution(solutions, [], optimizedRoadmapWithStartAndGoals,
                              "optimizedRoadmapShortcutSolution")
            return solutions

        # Create path optimized solution
        optimizedRoadmapShortcutSolution, optimizedRoadmapShortcutGraph = self._pathOptimizer.shortcut_path(
            optimizedRoadmapWithStartAndGoals,
            optimizedRoadmapTSGSolution)
        self._addSolution(solutions, optimizedRoadmapShortcutSolution, optimizedRoadmapShortcutGraph,
                          "optimizedRoadmapShortcutSolution")
        return solutions

    @IPPerfMonitor
    def _findTSGSolution(self, usedRoadmapWithGoals):
        goalNodes = [node for node in usedRoadmapWithGoals.nodes if node.startswith("G")]
        return list(nx.algorithms.approximation.traveling_salesman_problem(
            usedRoadmapWithGoals,
            nodes=["S"] + goalNodes,  # Include start node in the TSP solver
            cycle=True
        ))

    def _addSolution(self, solutions, path, graph, name):
        cost = gu.pathLength(graph, path)
        solutions[name] = {
            "path" : path,
            "cost" : cost,
            "graph" : graph,
        }


    def trimPathToKnownNodes(self, path) -> list[str]:
        return [node for node in path if node.startswith("G") or node.startswith("S")]

    def _makeUsedPairs(self, visitOrder: list[str]) -> list[tuple[str, str]]:
        return list(zip(visitOrder, visitOrder[1:]))



    def _addNodeToRoadmap(self, graph, posList, kdTree, node_pos, label, multipleConnections=False):
        graph.add_node(label, nodeType = label, pos=node_pos)
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



    def _isVisible(self, pos, guardPos):
        return not self._collisionChecker.lineInCollision(pos, guardPos)
