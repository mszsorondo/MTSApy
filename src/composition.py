import warnings

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
        self._number_of_goals = 0
        self._expansion_order = []
        print("Warning: underlying Java code runs unused feature computations and buffers")


    def reset_from_copy(self):
        return self.__class__(self._problem, self._n, self._k).start_composition()
    def start_composition(self, mtsa_version_path = 'mtsa.jar'):
        assert(self._initial_state is None)
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
        return self



    def expand(self, idx):
        assert(not self._javaEnv.isFinished()), "Invalid expansion, composition is already solved"
        assert (idx<len(self.getFrontier()) and idx>=0), "Invalid index"
        self._javaEnv.expandAction(idx) #TODO check this is the same index as in the python frontier list
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
    def getLastExpanded(self): return self._javaEnv.heuristic.lastExpandedStateAction

    def _check_no_repeated_states(self):
        raise NotImplementedError


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
class CompositionAnalyzer:
    """class used to get Composition information, usable as hand-crafted features
        TODO this class will be replaced by object-oriented Feature class
    """

    def __init__(self, composition : CompositionGraph):
        self.composition = composition
        assert (self.composition._started)

        self._no_indices_alphabet = list(set([self.remove_indices(str(e)) for e in composition._alphabet]))
        self._no_indices_alphabet.sort()
        self._fast_no_indices_alphabet_dict = dict()
        for i in range(len(self._no_indices_alphabet)): self._fast_no_indices_alphabet_dict[self._no_indices_alphabet[i]]=i
        self._fast_no_indices_alphabet_dict = bidict(self._fast_no_indices_alphabet_dict)
        self._feature_methods = [self.event_label_feature,self.state_label_feature,self.controllable
                                ,self.marked_state, self.current_phase,self.child_node_state,
                                 self.uncontrollable_neighborhood, self.explored_state_child, self.isLastExpanded]


    def test_features_on_transition(self, transition):
        res = []
        for compute_feature in self._feature_methods: res.extend(compute_feature(transition))
        return [float(e) for e in res]
    def event_label_feature(self, transition):
        """
        Determines the label of ℓ in A E p .
        """
        feature_vec_slice = [0 for _ in self._no_indices_alphabet]
        self._set_transition_type_bit(feature_vec_slice, transition.action)
        #print(no_idx_label, feature_vec_slice)
        return feature_vec_slice

    def _set_transition_type_bit(self, feature_vec_slice, transition):
        no_idx_label = self.remove_indices(transition.toString())
        feature_vec_slice_pos = self._fast_no_indices_alphabet_dict[no_idx_label]
        feature_vec_slice[feature_vec_slice_pos] = 1

    def state_label_feature(self, transition):
        """
        Determines the labels of the explored
            transitions that arrive at s.
        """
        feature_vec_slice = [0 for _ in self._no_indices_alphabet]
        arriving_to_s = transition.state.getParents()
        for trans in arriving_to_s: self._set_transition_type_bit(feature_vec_slice,trans.getFirst())
        return feature_vec_slice
    def controllable(self, transition):
        return [float(transition.action.isControllable())]
    def marked_state(self, transition):
        """Whether s and s ′ ∈ M E p ."""
        return [float(transition.childMarked)]

    def current_phase(self, transition):
        return [float(self.composition._javaEnv.dcs.heuristic.goals_found > 0),
                float(self.composition._javaEnv.dcs.heuristic.marked_states_found > 0),
                float(self.composition._javaEnv.dcs.heuristic.closed_potentially_winning_loops > 0)]



    def child_node_state(self, transition):
        """Whether
        s ′ is winning, losing, none,
        or not yet
        explored."""
        res = [0, 0, 0]
        if(transition.child is not None):
            res = [float(transition.child.status.toString()=="GOAL"),
                   float(transition.child.status.toString()=="ERROR"),
                   float(transition.child.status.toString()=="NONE")]
        return res
    def uncontrollable_neighborhood(self, transition):
        warnings.warn("Chequear que este bien")
        return [float(transition.state.uncontrollableUnexploredTransitions>0),
                float(transition.state.uncontrollableTransitions>0),
                float(transition.child is None or transition.child.uncontrollableUnexploredTransitions > 0),
                float(transition.child is None or transition.child.uncontrollableTransitions > 0)
                ]

    def explored_state_child(self, transition):
        return [float(len(self.composition.out_edges(transition.state))!= transition.state.unexploredTransitions),
                float(transition.child is not None and len(self.composition.out_edges(transition.child))!= transition.state.unexploredTransitions)]

    def isLastExpanded(self, transition):
        return [float(self.composition.getLastExpanded()==transition)]

    def remove_indices(self, transition_label : str):
        res = ""

        for c in transition_label:
            if not c.isdigit(): res += c

        return res
    def get_transition_features_size(self): return len(self.compute_features(self.composition.getFrontier()[0]))

    def compute_features(self, transition):
        res = []
        for feature_method in self._feature_methods:
            res += feature_method(transition)
        return res

class Environment:
    def __init__(self, contexts : CompositionAnalyzer, normalize_reward : bool = False):
        """Environment base class.
            TODO are contexts actually part of the concept of an RL environment?
            """
        self.contexts = contexts
        self.normalize_reward = normalize_reward

    def reset_from_copy(self):
        self.contexts = [CompositionAnalyzer(context.composition.reset_from_copy()) for context in self.contexts]
        return self

    def get_number_of_contexts(self):
        return len(self.contexts)
    def get_contexts(self):
        return self.contexts
    def step(self, action_idx, context_idx = 0):
        composition_graph = self.contexts[context_idx].composition
        composition_graph.expand(action_idx) # TODO refactor. Analyzer should not be the expansion medium
        Warning("HERE obs is not actions, but the featurization of the frontier actions")
        if not composition_graph._javaEnv.isFinished(): return self.frontier_features(), self.reward(), False, {}
        else: return None, self.reward(), True, self.get_results()
    def get_results(self, context_idx = 0):
        composition_dg = self.contexts[context_idx].composition
        return {
            "synthesis time(ms)": float(composition_dg._javaEnv.getSynthesisTime()),
            "expanded transitions": int(composition_dg._javaEnv.getExpandedTransitions()),
            "expanded states": int(composition_dg._javaEnv.getExpandedStates())
        }


    def reward(self):
        #TODO ?normalize like Learning-Synthesis?
        return -1
    def state(self):
        raise NotImplementedError
    def actions(self, context_idx=0):
        #TODO refactor
        return self.contexts[context_idx].composition.getFrontier()
    def frontier_features(self):
        #TODO you can parallelize this
        return [self.contexts[0].compute_features(trans) for trans in self.contexts[0].getFrontier()]

if __name__ == "__main__":
    d = CompositionGraph("AT", 3, 3)

    d.start_composition()
    da = CompositionAnalyzer(d)
    k = 0
    i = 100
    while(i and not d._javaEnv.isFinished()):
        d.expand(0)
        frontier_features = [(da.compute_features(trans)) for trans in d.getFrontier()]
        assert(d._expansion_order[-1] in [e[2]["action_with_features"] for e in d.edges(data=True)])

        #k+=sum([sum(da.isLastExpanded(trans[2]["action_with_features"])) for trans in d.edges(data=True)])
        #i-=1

    print(k)



