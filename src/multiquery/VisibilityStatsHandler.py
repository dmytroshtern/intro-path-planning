import networkx as nx

class VisibilityStatsHandler():

    def __init__(self):
        self.graph = nx.Graph()

    def addNodeAtPos(self, nodeNumber, pos):
        self.graph.add_node(nodeNumber, pos=pos, color='yellow')
        return

    def addVisTest(self, fr, to):
        self.graph.add_edge(fr, to)
        return