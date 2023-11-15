import random

from util import *
from torch_geometric.nn import GCNConv, Sequential
from torch import nn
from composition import CompositionGraph, util_remove_indices
class Feature:
    def __call__(cls, data):
        raise NotImplementedError
    def compute(self, data):
        raise NotImplementedError

class NodeFeature(Feature):
    @classmethod
    def compute(cls, data):
        raise NotImplementedError

class GlobalFeature(Feature):

    @classmethod
    def compute(cls , data):
        raise NotImplementedError

    @classmethod
    def __call__(cls, data):
        return cls.compute(data)

class AutoencoderEmbeddings(GlobalFeature):

    @classmethod
    def compute(cls, data : CompositionGraph) -> dict:
        raise NotImplementedError
    def train(self, data : CompositionGraph, static_feature_extractor) -> torch.nn.Module:
        raise NotImplementedError
class Node2Vec(GlobalFeature):
    def compute(cls, data : CompositionGraph) -> dict:
        raise NotImplementedError
    def train(self, data : CompositionGraph, static_feature_extractor) -> torch.nn.Module:
        raise NotImplementedError
class TransitionFeature(Feature):

    @classmethod
    def __call__(cls, state: CompositionGraph, transition):
        return cls.compute(state, transition)
    @classmethod
    def compute(cls, state : CompositionGraph, transition):
        raise NotImplementedError
    @classmethod
    def _set_transition_type_bit(cls,feature_vec_slice, transition, _fast_no_indices_alphabet_dict):
        no_idx_label = util_remove_indices(transition.toString())
        feature_vec_slice_pos = _fast_no_indices_alphabet_dict[no_idx_label]
        feature_vec_slice[feature_vec_slice_pos] = 1

class RandomTransitionFeature(TransitionFeature):
    @classmethod
    def compute(cls, state: CompositionGraph, transition, size=500):
        return [float(i) for i in np.random.rand(size)]

class RandomOneHotTransitionFeature(TransitionFeature):
    @classmethod
    def compute(cls, state: CompositionGraph, transition, size=300):
        return [float(i) for i in (np.random.rand(size)>=0.5)]

class RandomNodeFeature(NodeFeature):
    @classmethod
    def compute(cls, state: CompositionGraph, node, size=500):
        return [float(i) for i in np.random.rand(size)]

class LabelsOHE(NodeFeature):
    @classmethod
    def compute(cls, state: CompositionGraph, node, dir = "in"):
        incoming = state.in_edges(node, data=True) if dir=="in" else state.out_edges(node, data=True)

        feature_vec_slice = [0 for _ in state._no_indices_alphabet]
        #arriving_to_s = transition.state.getParents()
        for edge in incoming:
            cls._set_transition_type_bit(feature_vec_slice, edge[2]["label"], state._fast_no_indices_alphabet_dict)
        return feature_vec_slice
    @classmethod
    def _set_transition_type_bit(cls, feature_vec_slice, transition, _fast_no_indices_alphabet_dict):
        no_idx_label = util_remove_indices(transition.toString() if type(transition) != str else transition)
        feature_vec_slice_pos = _fast_no_indices_alphabet_dict[no_idx_label]
        feature_vec_slice[feature_vec_slice_pos] = 1
class RandomOneHotNodeFeature(NodeFeature):
    @classmethod
    def compute(cls, state: CompositionGraph, node, size=300):
        return [float(i) for i in (np.random.rand(size)>=0.5)]

class GAEEmbedding(NodeFeature):
    def __init__(self, gae : nn.Module):
        self.gae = gae

    def compute(cls, state: CompositionGraph, node):
        raise NotImplementedError
class EventLabel(TransitionFeature):

    @classmethod
    def compute(cls, state : CompositionGraph, transition):
        """
            Determines the label of ℓ in A E p .
                """

        #TODO add another featurevec for salient transitions
        feature_vec_slice = [0 for _ in state._no_indices_alphabet]
        cls._set_transition_type_bit(feature_vec_slice, transition.action, state._fast_no_indices_alphabet_dict)
        # print(no_idx_label, feature_vec_slice)
        return [float(e) for e in feature_vec_slice]

class StateLabel(TransitionFeature):
    @classmethod
    def compute(cls, state: CompositionGraph, transition):
        """
            Determines the labels of the explored
                transitions that arrive at s.
            """
        feature_vec_slice = [0 for _ in state._no_indices_alphabet]
        arriving_to_s = transition.state.getParents()
        for trans in arriving_to_s: cls._set_transition_type_bit(feature_vec_slice, trans.getFirst(), state._fast_no_indices_alphabet_dict)
        return feature_vec_slice
class Controllable(TransitionFeature):
    @classmethod
    def compute(cls, state : CompositionGraph, transition):
        return [float(transition.action.isControllable())]

class MarkedSourceAndSinkStates(TransitionFeature):

    @classmethod
    def compute(cls, state : CompositionGraph, transition):
        return [float(transition.state.marked),float(transition.childMarked)]
class MarkedState(NodeFeature):
    @classmethod
    def compute(cls, state: CompositionGraph, node):
        return [float(node.marked)]

class CurrentPhase(TransitionFeature):
    @classmethod
    def compute(cls, state: CompositionGraph, transition):
        return [float(state._javaEnv.dcs.heuristic.goals_found > 0),
                float(state._javaEnv.dcs.heuristic.marked_states_found > 0),
                float(state._javaEnv.dcs.heuristic.closed_potentially_winning_loops > 0)]

class ChildNodeState(TransitionFeature):
    @classmethod
    def compute(cls, state: CompositionGraph, transition):
        res = [0, 0, 0]
        if (transition.child is not None):
            res = [float(transition.child.status.toString() == "GOAL"),
                   float(transition.child.status.toString() == "ERROR"),
                   float(transition.child.status.toString() == "NONE")]
        return res

class UncontrollableNeighborhood(TransitionFeature):
    @classmethod
    def compute(cls, state: CompositionGraph, transition):
        return [float(transition.state.uncontrollableUnexploredTransitions > 0),
                float(transition.state.uncontrollableTransitions > 0),
                float(transition.child is None or transition.child.uncontrollableUnexploredTransitions > 0),
                float(transition.child is None or transition.child.uncontrollableTransitions > 0)
                ]


class ExploredStateChild(TransitionFeature):
    @classmethod
    def compute(cls, state: CompositionGraph, transition):
        res = [float(transition.state.unexploredTransitions) != float(len(transition.state.getTransitions())),
                float(transition.child is not None and float(transition.child.unexploredTransitions) != float(len(transition.child.getTransitions())))]

        return [float(i) for i in res]

class IsLastExpanded(TransitionFeature):
    @classmethod
    def compute(cls, state: CompositionGraph, transition):
        return [float(transition.state==transition.heuristic.lastExpandedTo), float(transition.state==transition.heuristic.lastExpandedFrom)]

class ChildDeadlock(TransitionFeature):
    @classmethod
    def compute(cls, state : CompositionGraph, transition):
        return [float(transition.child is not None and len(transition.child.getTransitions()) == 0)]


class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        planes = 128
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.first_layer = Sequential('x, edge_index', [
            (GCNConv(in_channels, planes), 'x, edge_index -> x'),
            nn.ReLU(inplace=True),
        ])

        def conv_block():
            return Sequential('x, edge_index', [
                (GCNConv(planes, planes), 'x, edge_index -> x'),
                nn.ReLU(inplace=True)
            ])

        self.conv_blocks = Sequential('x, edge_index', [(conv_block(), 'x, edge_index -> x') for _ in range(7)])

        self.last_linear = nn.Linear(planes, out_channels)

    def forward(self, x, edge_index):
        x = self.first_layer(x, edge_index)
        x = self.conv_blocks(x, edge_index)
        x = self.last_linear(x)
        return x


def train(model, optimizer, features, train_pos_edge_label_index, train_neg_edge_label_index):
    model.train()
    optimizer.zero_grad()

    z = model.encode(features, train_pos_edge_label_index)
    #   breakpoint()
    loss = model.recon_loss(z, train_pos_edge_label_index, train_neg_edge_label_index)
    # if args.variational:
    #   loss = loss + (1 / data.num_nodes) * model.kl_loss()
    loss.backward()
    optimizer.step()
    return float(loss)

def test(model, test_pos_edge_index, test_neg_edge_index, features,train_pos_edge_label_index, train_neg_edge_label_index):
    model.eval()
    with torch.no_grad():
        #breakpoint()
        z = model.encode(features, train_pos_edge_label_index)
    return model.test(z, train_pos_edge_label_index, train_neg_edge_label_index)




