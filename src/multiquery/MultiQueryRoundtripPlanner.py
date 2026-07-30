from cProfile import label

import networkx as nx
from typing import Type, Any, List

from IPPerfMonitor import IPPerfMonitor
from IPEnvironment import CollisionChecker

from . import GraphUtility as gu
from .PathOptimizer import PathOptimizer
from .RoadmapOptimizer import RoadmapOptimizer
from .VisibilityPRMRoadmapper import VisibilityPRMRoadmapper
from .VisibilityStatsHandler import VisibilityStatsHandler
from notebooks.IPPlanerBase import PlanerBase
from src.roundtrip_algorithm.result import (
    roundtrip_failure,
    roundtrip_success,
)

class MultiQueryRoundtripPlanner(PlanerBase):
    """Plan roundtrip tours by reusing a visibility roadmap across multiple goals."""

    def __init__(self, collisionChecker: CollisionChecker):
        """Initialize the reusable roadmap, optimizers, and result graph."""
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
        Plan a roundtrip path through the provided start and goal poses.

        Args:
            startList: Start poses in planning space, typically a one-element list like ``[[x, y]]``.
            goalList: Goal poses in planning space.
            config: Configuration dictionary controlling roadmap and path optimization.

        Returns:
            A roundtrip result dictionary produced by ``roundtrip_success`` or ``roundtrip_failure``.
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

        #Overwrite Config to match general benchmarking scheme
        method = config.get("ordering_method", None)
        if not method == None:
            isOptimized = method == "exact"
            config["optimizeRoadmap"] = isOptimized
            config["optimizePath"] = isOptimized

        baseRoadmap = self._roadmapPlanner.learnRoadmap(config, self.statsHandler)

        if config.get("optimizeRoadmap", False):
            usedRoadmapWithGoals = self.generateOptimizedRoadmap(baseRoadmap, checkedGoalList, checkedStartList, config)
        else:
            usedRoadmapWithGoals = self._roadmapPlanner.addStartAndGoalsToRoadmap(baseRoadmap.copy(), checkedStartList,
                                                                  checkedGoalList, False)

        try:
            tspSolution = self._findTSPSolution(usedRoadmapWithGoals)

        except Exception as e:
            self.graph = usedRoadmapWithGoals
            return roundtrip_failure(
                reason=f"TSP solver failed: {e}",
                pairwise_results=None,
                failed_pairs=None,
                metadata={
                    "stage": "TSP solver"
                }
            )
        if config.get("optimizePath", False):
            shortcutSolution, shortcutGraph = self._pathOptimizer.shortcut_path(usedRoadmapWithGoals, tspSolution)
            usedSolution = shortcutSolution
            usedCost = gu.pathLength(shortcutGraph, shortcutSolution)
            self.graph = shortcutGraph
        else:
            usedSolution = tspSolution
            usedCost = gu.pathLength(usedRoadmapWithGoals, tspSolution)
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
        Run the planner in several configurations and return every intermediate solution.

        Args:
            startList: Start poses in planning space.
            goalList: Goal poses in planning space.
            config: Configuration dictionary controlling roadmap and path optimization.

        Returns:
            A dictionary mapping solution names to their path, graph, and cost.
        """
        checkedStartList, checkedGoalList = self._checkStartGoal(startList, goalList)
        if len(checkedStartList) == 0 or not checkedStartList[0] == startList[0]:
            return {}
        if not len(checkedGoalList) is len(goalList):
            return {}

        baseRoadmap = self._roadmapPlanner.learnRoadmap(config, self.statsHandler)
        # Conversion of node names to string for consistency
        baseRoadmap = nx.relabel_nodes(baseRoadmap, str, copy=False)

        solutions = {}

        basicRoadmapWithStartAndGoals = self._roadmapPlanner.addStartAndGoalsToRoadmap(baseRoadmap.copy(), checkedStartList,
                                                                       checkedGoalList, False)
        try:
            # Create basic roadmap TSP solution
            basicRoadmapTSPSolution = self._findTSPSolution(basicRoadmapWithStartAndGoals)
            self._addSolution(solutions, basicRoadmapTSPSolution, basicRoadmapWithStartAndGoals,
                              "basicRoadmapTSPSolution")

            # Create path optimized solution
            basicRoadmapShortcutSolution, basicRoadmapShortcutGraph = self._pathOptimizer.shortcut_path(
                basicRoadmapWithStartAndGoals,
                basicRoadmapTSPSolution)
            self._addSolution(solutions, basicRoadmapShortcutSolution, basicRoadmapShortcutGraph,
                              "basicRoadmapShortcutSolution")
        except Exception as e:
            #No TSP solution found
            self._addSolution(solutions, [], basicRoadmapWithStartAndGoals,
                              "basicRoadmapTSPSolution")
            self._addSolution(solutions, [], basicRoadmapWithStartAndGoals,
                              "basicRoadmapShortcutSolution")


        optimizedRoadmapWithStartAndGoals = self.generateOptimizedRoadmap(baseRoadmap, checkedGoalList,
                                                                          checkedStartList, config)

        try:
            # Create optimized roadmap TSP solution
            optimizedRoadmapTSPSolution = self._findTSPSolution(optimizedRoadmapWithStartAndGoals)
            self._addSolution(solutions, optimizedRoadmapTSPSolution, optimizedRoadmapWithStartAndGoals,
                              "optimizedRoadmapTSPSolution")
        except Exception as e:
            # No solution for optimized Roadmap -> No solution at all expected
            self.graph = optimizedRoadmapWithStartAndGoals
            self._addSolution(solutions, [], optimizedRoadmapWithStartAndGoals,
                              "optimizedRoadmapTSPSolution")
            self._addSolution(solutions, [], optimizedRoadmapWithStartAndGoals,
                              "optimizedRoadmapShortcutSolution")
            return solutions

        # Create path optimized solution
        optimizedRoadmapShortcutSolution, optimizedRoadmapShortcutGraph = self._pathOptimizer.shortcut_path(
            optimizedRoadmapWithStartAndGoals,
            optimizedRoadmapTSPSolution)
        self._addSolution(solutions, optimizedRoadmapShortcutSolution, optimizedRoadmapShortcutGraph,
                          "optimizedRoadmapShortcutSolution")
        return solutions

    def generateOptimizedRoadmap(self, baseRoadmap, checkedGoalList: list[Any], checkedStartList: list[Any],
                                 config: dict) -> Any:
        """Attach start and goal nodes, add random shortcuts, and refine the roadmap if needed."""
        optimizedRoadmapWithStartAndGoals = self._roadmapPlanner.addStartAndGoalsToRoadmap(baseRoadmap.copy(),
                                                                                           checkedStartList,
                                                                                           checkedGoalList, True)
        optimizedRoadmapWithStartAndGoals = self._roadmapOptimizer.optimizeRandomShortcuts(
            optimizedRoadmapWithStartAndGoals)

        component = nx.node_connected_component(optimizedRoadmapWithStartAndGoals, "S")
        all_connected = all(node in component for node in
                            [node for node in optimizedRoadmapWithStartAndGoals.nodes if node.startswith("G")])
        if not all_connected:
            optimizedRoadmapWithStartAndGoals = self._roadmapOptimizer.closeGaps(optimizedRoadmapWithStartAndGoals)
            optimizedRoadmapWithStartAndGoals = self._roadmapPlanner.refineRoadmap(config, self.statsHandler,
                                                                                   optimizedRoadmapWithStartAndGoals)
        return optimizedRoadmapWithStartAndGoals

    @IPPerfMonitor
    def _findTSPSolution(self, usedRoadmapWithGoals):
        """Solve a TSP over the start node and all goal nodes in the roadmap."""
        goalNodes = [node for node in usedRoadmapWithGoals.nodes if node.startswith("G")]
        return list(nx.algorithms.approximation.traveling_salesman_problem(
            usedRoadmapWithGoals,
            nodes=["S"] + goalNodes,  # Include start node in the TSP solver
            cycle=True
        ))

    def _addSolution(self, solutions, path, graph, name):
        """Store a named solution together with its cost and associated graph."""
        cost = gu.pathLength(graph, path)
        solutions[name] = {
            "path" : path,
            "cost" : cost,
            "graph" : graph,
        }


    def trimPathToKnownNodes(self, path) -> list[str]:
        """Keep only the start and goal labels from a full roadmap path."""
        return [node for node in path if node.startswith("G") or node.startswith("S")]

    def _makeUsedPairs(self, visitOrder: list[str]) -> list[tuple[str, str]]:
        """Convert a visit order into consecutive node pairs."""
        return list(zip(visitOrder, visitOrder[1:]))



    def _isVisible(self, pos, guardPos):
        """Return ``True`` when the segment between two poses is collision free."""
        return not self._collisionChecker.lineInCollision(pos, guardPos)
