from IPPRMBase import PRMBase
import networkx as nx
from IPPerfMonitor import IPPerfMonitor
import NumPy as np

class MultiQueryRoundtripPlanner:

    def __init__(self, roadmapPlannerClass: Type[Any], collision_checker: CollisionChecker):
        assert hasattr(path_planner, "createNewRoadmapGraph"), "path_planner must have a method called 'planPath'"
        self.graph = nx.Graph() # graph to store all paths between start and goal nodes
        self._roadmapPlanner = roadmapPlannerClass(collision_checker)


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

        roadmap = self._roadmapPlanner(startList, goalList, config)


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
        
        self.graph = self.plannerInstance.graph

        return list(tsg_solution)
