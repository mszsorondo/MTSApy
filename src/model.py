import networkx as nx

from util import *


class CompositionGraph(nx.DiGraph):
    def __init__(self, problem, n, k):
        super().__init__()
        self._problem, self._n, self._k = problem, n , k
        self._initial_state = None
        self._state_machines  = [] # : list[nx.DiGraph]
        self._frontier = []
        self._started, self._completed = False, False
        self._javaEnv = None
        self._alphabet = []
        self._no_indices_alphabet = []
    def start_composition(self, mtsa_version_path = 'mtsa.jar'):
        assert(self._initial_state is None)
        print("Warning: underlying Java code runs unused feature computations and buffers")
        if not jpype.isJVMStarted(): jpype.startJVM(classpath=[mtsa_version_path])
        from MTSTools.ac.ic.doc.mtstools.model.operations.DCS.nonblocking import DirectedControllerSynthesisNonBlocking, FeatureBasedExplorationHeuristic, DCSForPython
        from ltsa.dispatcher import TransitionSystemDispatcher
        self._started = True
        c = FeatureBasedExplorationHeuristic.compileFSP(f"{FSP_PATH}/{self._problem}/{self._problem}-{self._n}-{self._k}.fsp")
        ltss_init = c.getFirst()
        self._state_machines = [m.name for m in ltss_init.machines] #TODO: turn it into a dictionary that goes from the state machine name into its respective digraph
        self._javaEnv = DCSForPython(None, None, 10000, ltss_init)
        self._javaEnv.startSynthesis(f"{FSP_PATH}/{self._problem}/{self._problem}-{self._n}-{self._k}.fsp")
        assert(self._javaEnv is not None)
        self._initial_state = self._javaEnv.dcs.initial
        self.add_node(self._initial_state)
        self._alphabet = [e for e in self._javaEnv.dcs.alphabet.actions]
        self._alphabet.sort()



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

    def explored(self, transition):
        """
        Whether a transition from s or s ′ has
            already been explored.

        """
        raise NotImplementedError
    def last_expanded(self, transition):
        """Whether s is the last expanded state in h
            (outgoing or incoming)."""
        raise NotImplementedError

    def finished(self):
        return self._javaEnv.isFinished()
class CompositionAnalyzer:
    """class used to get Composition information, usable as hand-crafted features"""

    def __init__(self, composition : CompositionGraph):
        self.composition = composition
        assert (self.composition._started)

        self._no_indices_alphabet = list(set([self.remove_indices(str(e)) for e in composition._alphabet]))
        self._no_indices_alphabet.sort()
        breakpoint()
        print(self._no_indices_alphabet)

    def event_label_feature(self, transition):
        """
        Determines the label of ℓ in A E p .
        """
        raise NotImplementedError
    def state_label_feature(self, transition):
        """
        Determines the labels of the explored
            transitions that arrive at s.
        """
        raise NotImplementedError
    def controllable(self, transition):
        raise NotImplementedError

    def marked_state(self, transition):
        """Whether s and s ′ ∈ M E p ."""
        raise NotImplementedError

    def current_phase(self):
        raise NotImplementedError

    def child_node_state(self, transition):
        """Whether
        s ′ is winning, losing, none,
        or not yet
        explored."""
        raise NotImplementedError



    def remove_indices(self, transition_label : str):
        res = ""

        for c in transition_label:
            if not c.isdigit(): res += c

        return res



if __name__ == "__main__":
    d = CompositionGraph("AT", 2, 2)

    d.start_composition()
    da = CompositionAnalyzer(d)

    d.expand(0)
    d.expand(0)
    d.expand(0)
    pos = nx.spring_layout(d)
    nx.draw(d, with_labels=True, arrows=True)

    edge_controllability = nx.get_edge_attributes(d, 'controllability')
    nx.draw_networkx_edge_labels(d, pos, edge_labels=edge_controllability)

    plt.savefig("graph.png")

