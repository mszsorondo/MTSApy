import copy
import random
import warnings
import networkx as nx
from util import remove_indices as util_remove_indices
from util import *



class CompositionGraph(nx.MultiDiGraph):
    def __init__(self, problem, n, k):
        super().__init__()
        self._problem, self._n, self._k = problem, n , k
        self._initial_state = None
        self._state_machines  = [] # TODO: list[nx.DiGraph]
        self._frontier = []
        self._started, self._completed = False, False
        self._javaEnv = None
        self._alphabet = []
        self._no_indices_alphabet = []
        self._number_of_goals = 0
        self._expansion_order = []
        self._fast_no_indices_alphabet_dict = dict()
        print("Warning: underlying Java code runs unused feature computations and buffers")

    def __str__(self):
        return f"Composition for {self._problem,self._n, self._k}. {len(self.nodes)} nodes found and {len(self.edges)} edges expanded. \n"\
              + f"{self.getFrontierSize()} edges on frontier."

    def to_pure_nx(self, cls = nx.MultiDiGraph):
        D = cls()
        D.nodes, D.edges = self.nodes, self.edges
        return D
    @staticmethod
    def copy_with_nodes_as_ints(G, drop_edge_attrs = ["action_with_features"]):
        mapping = bidict({n:i for n,i in zip(G.nodes, range(len(G.nodes)))})

        D = nx.MultiDiGraph()
        for n,d in G.nodes(data=True):
            D.add_node(mapping[n], **d)
        for s,t,d in G.edges(data=True):
            [d.pop(attr) for attr in drop_edge_attrs]
            D.add_edge(mapping[s],mapping[t], **d)

        return D
    def load(self, path = f"/home/marco/Desktop/Learning-Synthesis/experiments/plants/full_AT_2_2.pkl"):
        raise NotImplementedError
        assert self._javaEnv is None, "You can't load a new graph in the middle of a composition. Make a new Composition object for that."
        with open(path, 'rb') as f:
            G_train = pickle.load(f)
            raise NotImplementedError

    def full_composition(self):
        """
        Composes the full plant.
        """
        assert self._javaEnv is not None and len(self.edges())==0, "You already started an expansion"
        self._javaEnv.set_initial_as_none()
        while(self.getFrontierSize()>0):

            self._javaEnv.set_initial_as_none()
            self.expand(0)
            nonfront = self.getNonFrontier()
            lastexp = self.getLastExpanded()
            assert lastexp.state == nonfront[len(nonfront)-1].state
            assert lastexp.action == nonfront[len(nonfront) - 1].action
            self._javaEnv.set_compostate_as_none(lastexp.state)

        return self


    def reset_from_copy(self):
        return self.__class__(self._problem, self._n, self._k).start_composition()
    def start_composition(self, mtsa_version_path = 'mtsa.jar', no_tau=True):
        #TODO more elegant tau removal
        assert(self._initial_state is None)
        if not jpype.isJVMStarted(): jpype.startJVM(classpath=[mtsa_version_path])
        from MTSTools.ac.ic.doc.mtstools.model.operations.DCS.nonblocking import DirectedControllerSynthesisNonBlocking, FeatureBasedExplorationHeuristic, DCSForPython
        from ltsa.dispatcher import TransitionSystemDispatcher
        self._started = True
        c = FeatureBasedExplorationHeuristic.compileFSP(f"{FSP_PATH}/{self._problem}/{self._problem}-{self._n}-{self._k}.fsp")
        ltss_init = c.getFirst()
        self._state_machines = [m.name for m in ltss_init.machines] #TODO: turn it into a dictionary that goes from the state machine name into its respective digraph
        #TODO stop using deleteme_featureset.txt for setting MTSA features below
        self._javaEnv = DCSForPython("./src/deleteme_featureset.txt", f"{LABELS_PATH}/{self._problem}.txt", 10000, ltss_init)
        self._javaEnv.startSynthesis(f"{FSP_PATH}/{self._problem}/{self._problem}-{self._n}-{self._k}.fsp")
        self._javaEnv.heuristic.debugging = False
        assert(self._javaEnv is not None)
        self._initial_state = self._javaEnv.dcs.initial
        self.add_node(self._initial_state)
        self._alphabet = [e for e in self._javaEnv.dcs.alphabet.actions]
        self._alphabet.sort()
        self._no_indices_alphabet = list(set([util_remove_indices(str(e)) for e in self._alphabet]))
        self._no_indices_alphabet.sort()
        if no_tau: self._no_indices_alphabet.remove("tau")
        for i in range(len(self._no_indices_alphabet)): self._fast_no_indices_alphabet_dict[
            self._no_indices_alphabet[i]] = i
        self._fast_no_indices_alphabet_dict = bidict(self._fast_no_indices_alphabet_dict)
        return self






    def expand(self, idx):
        assert(not self._javaEnv.isFinished()), "Invalid expansion, composition is already solved"
        assert (idx<self.getFrontierSize() and idx>=0), "Invalid index"

        if self.getFrontierSize()>0:
            self._javaEnv.expandAction(idx) #TODO check this is the same index as in the python frontier list
        else: return
        new_state_action = self.getLastExpanded()
        controllability, label = self.getLastExpanded().action.isControllable(), self.getLastExpanded().action.toString()
        self.add_node(self.last_expansion_child_state())
        self.add_edge(new_state_action.state, self.last_expansion_child_state(), controllability=controllability, label=label, action_with_features = new_state_action)
        self._expansion_order.append(self.getLastExpanded())
    def last_expansion_child_state(self):
        return self._javaEnv.heuristic.lastExpandedTo
    def last_expansion_source_state(self):
        return self._javaEnv.heuristic.lastExpandedFrom
    def getFrontier(self): return self._javaEnv.heuristic.explorationFrontier

    def getNonFrontier(self): return self._javaEnv.heuristic.allActionsWFNoFrontier
    def getFrontierSize(self): return self._javaEnv.frontierSize()
    def getLastExpanded(self): return self._javaEnv.heuristic.lastExpandedStateAction

    def _check_no_repeated_states(self):
        raise NotImplementedError

    def info(self):
        return {"n":self._n,"k":self._k,"problem":self._problem}
    def explored(self, transition):
        """
        TODO
        Whether a transition from s or s ′ has
            already been explored.

        """
        raise NotImplementedError
    def last_expanded(self, transition):
        """
        TODO
        Whether s is the last expanded state in h
            (outgoing or incoming)."""
        raise NotImplementedError

    def finished(self):
        return self._javaEnv.isFinished()
    def get_jvm(self):
        return self._javaEnv

    """if __name__ == "__main__":
        d = CompositionGraph("AT", 3, 3)
    
        d.start_composition()
        da = FeatureExtractor(d)
        k = 0
        i = 100
        while(i and not d._javaEnv.isFinished()):
            d.expand(0)
            frontier_features = [(da.compute_features(trans)) for trans in d.getFrontier()]
            assert(d._expansion_order[-1] in [e[2]["action_with_features"] for e in d.edges(data=True)])
    
            #k+=sum([sum(da.isLastExpanded(trans[2]["action_with_features"])) for trans in d.edges(data=True)])
            #i-=1
    
        print(k)
    """



