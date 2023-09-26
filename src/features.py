from util import *

class Feature:
    def __init__(self):
        raise NotImplementedError
    def compute(self, data):
        raise NotImplementedError
class GraphEmbedding(Feature):
    def __init__(self):
        raise NotImplementedError
    def compute(self, data):
        raise NotImplementedError


class FeatureExtractor:
    """class used to get Composition information, usable as hand-crafted features
        TODO this class will be replaced by object-oriented Feature class
    """

    def __init__(self, composition):
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

        Warning("Chequear que este bien")
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



