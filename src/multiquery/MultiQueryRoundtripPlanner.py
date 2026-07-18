import networkx as nx
import numpy as np
from typing import Type, Any, List
import matplotlib.pyplot as plt

from IPPRMBase import PRMBase
from IPPerfMonitor import IPPerfMonitor
from IPEnvironment import CollisionChecker

class MultiQueryRoundtripPlanner:

    def __init__(self, roadmapPlanner: Type[Any], collisionChecker: CollisionChecker):
        #assert hasattr(roadmapPlannerClass, "createNewRoadmapGraph"), "roadmapPlannerClass must have a method called 'createNewRoadmapGraph'"
        self.graph = nx.Graph() # graph to store all paths between start and goal nodes
        self._roadmapPlanner = roadmapPlanner
        self.statsHandler = roadmapPlanner.statsHandler
        self._collisionChecker = collisionChecker


    @IPPerfMonitor
    def planPath(self, startList: List[List[Any]], goalList: List[List[Any]], config) -> List[Any]:
        """
        Plans a roundtrip path that visits the first start and goal nodes.
        Args:
            startList (array): start position in planning space. E.g. [[1,2]]
            goalList (array) : goal position in planning space. E.g. [[3,4]]
            config (dict): dictionary with the needed information about the configuration options

        Returns:
            List[List[Any]]: A list representing the roundtrip path visiting all goals and returning to the start.
        """

        roadmapWithStartAndGoals = self._roadmapPlanner.createNewRoadmapGraph(startList, goalList, config)
        
        #fig, axes = plt.subplots()
        #nx.draw(roadmapWithStartAndGoals, ax=axes, with_labels=True)

        # Conversion of node names to string for consistency
        roadmapWithStartAndGoals = nx.relabel_nodes(
            roadmapWithStartAndGoals,
            {node: str(node) for node in roadmapWithStartAndGoals.nodes},
            copy=False
        )

        # remove the self-loop edge if it exists
        #if self.plannerInstance.graph.has_edge("start", "start"):
        #    self.plannerInstance.graph.remove_edge("start", "start")

        goalNodes = [node for node in roadmapWithStartAndGoals.nodes if node.startswith("goal")]
        
        tsg_solution = np.asarray(nx.algorithms.approximation.traveling_salesman_problem(
            roadmapWithStartAndGoals,
            nodes=["start"] + goalNodes,  # Include start node in the TSP solver
            cycle=True
        ))

        if len(tsg_solution) < 2:
            return []
        
        self.graph = roadmapWithStartAndGoals

        return list(tsg_solution)
