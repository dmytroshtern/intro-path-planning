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

        self.graph = self._roadmapPlanner.createNewRoadmapGraph(startList, goalList, config)
        
        
        # Conversion of node names to string for consistency
        self.graph = nx.relabel_nodes(self.graph, str, copy=False)

        goalNodes = [node for node in self.graph.nodes if node.startswith("goal")]
        
        try:            
            tsg_solution = np.asarray(nx.algorithms.approximation.traveling_salesman_problem(
                self.graph,
                nodes=["start"] + goalNodes,  # Include start node in the TSP solver
                cycle=True
            ))
        except Exception as e:
            return []

        return list(tsg_solution)
