from util import *


class CompositionGraph(nx.DiGraph):
    def __init__(self, problem, n, k):
        super().__init__()
        self._problem, self._n, self._k = problem, n , k
        self._initial_state = None
        self._machines , self._frontier = [] , []
        self._completed = False
        self._javaEnv = None
    def start_composition(self, mtsa_version_path = 'mtsa.jar'):
        assert(self._initial_state is None)
        print("Warning: underlying Java code runs unused feature computations and buffers")
        if not jpype.isJVMStarted(): jpype.startJVM(classpath=[mtsa_version_path])
        from MTSTools.ac.ic.doc.mtstools.model.operations.DCS.nonblocking import DirectedControllerSynthesisNonBlocking, FeatureBasedExplorationHeuristic, DCSForPython
        from ltsa.dispatcher import TransitionSystemDispatcher

        c = FeatureBasedExplorationHeuristic.compileFSP(f"{FSP_PATH}/{self._problem}/{self._problem}-{self._n}-{self._k}.fsp")
        ltss_init = c.getFirst()
        self._javaEnv = DCSForPython(None, None, 10000, ltss_init);
        self._javaEnv.startSynthesis(f"{FSP_PATH}/{self._problem}/{self._problem}-{self._n}-{self._k}.fsp")
        assert(self._javaEnv is not None)
        self._initial_state = self._javaEnv.dcs.initial
        self.add_node(self._initial_state)


    def expand(self, idx):
        assert (idx<len(self.getFrontier()) and idx>=0), "invalid index"
        self._javaEnv.expandAction(idx)
        new_state_action = self.getLastExpanded()
        controllability, label = self.getLastExpanded().action.isControllable(), self.getLastExpanded().action.toString()
        self.add_node(self.last_expansion_child_state())
        self.add_edge(new_state_action.state, self.last_expansion_child_state(), controllability=controllability, label=label)

    def last_expansion_child_state(self):
        return self._javaEnv.heuristic.lastExpandedTo
    def last_expansion_source_state(self):
        return self._javaEnv.heuristic.lastExpandedFrom
    def getFrontier(self): return self._javaEnv.heuristic.explorationFrontier
    def getLastExpanded(self): return self._javaEnv.heuristic.lastExpandedStateAction

    def _check_no_repeated_states(self):
        raise NotImplementedError


if __name__ == "__main__":
    d = CompositionGraph("AT", 2, 2)

    d.start_composition()
    d.expand(0)
    d.expand(0)
    d.expand(0)
    pos = nx.spring_layout(d)
    nx.draw(d, with_labels=True, arrows=True)

    edge_controllability = nx.get_edge_attributes(d, 'controllability')
    nx.draw_networkx_edge_labels(d, pos, edge_labels=edge_controllability)

    plt.savefig("graph.png")

