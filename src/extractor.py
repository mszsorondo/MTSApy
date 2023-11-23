import torch
from bidict import bidict
from torch_geometric.nn import GAE
from torch_geometric.utils import from_networkx
from util import remove_indices as util_remove_indices
from features import *

LEARNING_SYNTHESIS_BENCHMARK_FEATURES = [EventLabel, StateLabel, Controllable, MarkedSourceAndSinkStates, CurrentPhase,
                                         ChildNodeState, UncontrollableNeighborhood, ExploredStateChild, IsLastExpanded, ChildDeadlock]

class FeatureExtractor:
    """class used to get Composition information, usable as hand-crafted features
        Design:
        attrs: Features (list of classes)
        methods: .extract(featureClass, from = composition) .phi(composition)
    """

    def __init__(self, composition, transition_feature_classes = LEARNING_SYNTHESIS_BENCHMARK_FEATURES, global_feature_classes = [], node_feature_classes = []):
        #FIXME composition should be a parameter of phi, since FeatureExctractor works ...
        # for any context independently UNLESS there are trained features
        # in general, composition should be completely decoupled from extractor (passed always as a parameter)
        self.composition = composition
        assert (self.composition._started)

        self._no_indices_alphabet = list(set([self.remove_indices(str(e)) for e in composition._alphabet]))
        self._no_indices_alphabet.sort()
        self._fast_no_indices_alphabet_dict = dict()
        for i in range(len(self._no_indices_alphabet)): self._fast_no_indices_alphabet_dict[self._no_indices_alphabet[i]]=i
        self._fast_no_indices_alphabet_dict = bidict(self._fast_no_indices_alphabet_dict)


        self._global_feature_classes = global_feature_classes
        self._transition_feature_classes = transition_feature_classes
        self._node_feature_classes = node_feature_classes
        self._feature_classes = self._global_feature_classes + self._transition_feature_classes + self._node_feature_classes
    def phi(self):
        #TODO make this function a parameter
        return self.frontier_feature_vectors()

    def extract(self, transition, state)-> list[float]:
        res = []
        for feature in self._transition_feature_classes:
            res += feature.compute(state=state, transition=transition)
        return res
    def extract_global_features(self, state):
        res = []
        for feature in self._global_feature_classes:
            res += feature.compute(state)
        return res
    def _set_transition_type_bit(self, feature_vec_slice, transition):
        no_idx_label = self.remove_indices(transition.toString())
        feature_vec_slice_pos = self._fast_no_indices_alphabet_dict[no_idx_label]
        feature_vec_slice[feature_vec_slice_pos] = 1


    def remove_indices(self, transition_label : str):
        util_remove_indices(transition_label)
    def get_transition_features_size(self):
        if(len(self.composition.getFrontier())): return len(self.extract(self.composition.getFrontier()[0], self.composition)) + sum([feature.size for feature in self._global_feature_classes])
        elif (len(self.composition.getNonFrontier())): return len(self.extract(self.composition.getNonFrontier()[0], self.composition)) + sum([feature.size for feature in self._global_feature_classes])
        else: raise ValueError

    def includes(self, feature):
        return self._enabled_feature_classes[feature]
    def non_frontier_feature_vectors(self) -> dict[tuple,list[float]]:
        # TODO you can parallelize this (GPU etc)
        return {(trans.state,trans.child) : self.extract(trans, self.composition) for trans in self.composition.getNonFrontier()}

    def frontier_feature_vectors(self) -> dict[tuple,list[float]]:
        res = dict()
        #TODO you can parallelize this (GPU etc)
        node_features = self.extract_global_features(self.composition)
        for trans in list(self.composition.getFrontier()):
            source_state_idx = self.composition.composition_int_identifier[trans.state]
            res.update({(trans.state,trans.action) : self.extract(trans, self.composition) + node_features[source_state_idx].tolist()})

        return res
    def frontier_feature_vectors_as_batch(self):
        return np.array(list(self.frontier_feature_vectors().values()), dtype=np.float32)
    def set_static_node_features(self):
        #FIXME refactor this
        for node in self.composition.nodes:
            in_label_ohe = LabelsOHE.compute(self.composition, node, dir="in")
            out_label_ohe = LabelsOHE.compute(self.composition, node, dir="out")
            marked =  MarkedState.compute(self.composition, node)
            self.composition.nodes[node]["features"] = in_label_ohe + out_label_ohe + marked
            self.composition.nodes[node]["compostate"] = node.toString()




    def global_feature_vectors(self) -> dict:
        raise NotImplementedError

    def train_node2vec(self):
        raise NotImplementedError
    def train_watch_your_step(self):
        raise NotImplementedError
    def train_DGI(self):
        raise NotImplementedError
    def __str__(self):
        return "feature classes: " + str(self._enabled_feature_classes)
