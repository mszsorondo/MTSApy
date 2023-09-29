from src.features.extractor import FeatureExtractor
from ..util import *
from torch_geometric.nn import GCNConv, Sequential
from torch import nn
from ..composition import CompositionGraph, util_remove_indices
class Feature:
    def __call__(cls, data):
        raise NotImplementedError
    def compute(self, data):
        raise NotImplementedError
class GraphEmbedding(Feature):

    @classmethod
    def compute(cls , data):
        raise NotImplementedError

    @classmethod
    def __call__(cls, data):
        return cls.compute(data)

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
        return feature_vec_slice

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

class MarkedState(TransitionFeature):
    def compute(cls, state : CompositionGraph, transition):
        return [float(transition.childMarked)]

class CurrentPhase(TransitionFeature):
    def compute(cls, state: CompositionGraph, transition):
        return [float(state._javaEnv.dcs.heuristic.goals_found > 0),
                float(state._javaEnv.dcs.heuristic.marked_states_found > 0),
                float(state._javaEnv.dcs.heuristic.closed_potentially_winning_loops > 0)]

class ChildNodeState(TransitionFeature):
    def compute(cls, state: CompositionGraph, transition):
        res = [0, 0, 0]
        if (transition.child is not None):
            res = [float(transition.child.status.toString() == "GOAL"),
                   float(transition.child.status.toString() == "ERROR"),
                   float(transition.child.status.toString() == "NONE")]
        return res

class UncontrollableNeighborhood(TransitionFeature):
    def compute(cls, state: CompositionGraph, transition):
        return [float(transition.state.uncontrollableUnexploredTransitions > 0),
                float(transition.state.uncontrollableTransitions > 0),
                float(transition.child is None or transition.child.uncontrollableUnexploredTransitions > 0),
                float(transition.child is None or transition.child.uncontrollableTransitions > 0)
                ]


class ExploredStateChild(TransitionFeature):
    def compute(cls, state: CompositionGraph, transition):
        return [float(len(state.out_edges(transition.state)) != transition.state.unexploredTransitions),
                float(transition.child is not None and len(
                    state.out_edges(transition.child)) != transition.state.unexploredTransitions)]

class IsLastExpanded(TransitionFeature):
    def compute(cls, state: CompositionGraph, transition):
        return [float(state.getLastExpanded().state==transition.state), float(state.getLastExpanded().child==transition.state)]



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


def train(model, optimizer, x_train, train_pos_edge_label_index):
    model.train()
    optimizer.zero_grad()
    z = model.encode(x_train, train_pos_edge_label_index)
    loss = model.recon_loss(z, train_pos_edge_label_index)
    # if args.variational:
    #   loss = loss + (1 / data.num_nodes) * model.kl_loss()
    loss.backward()
    optimizer.step()
    return float(loss)

def test(model, test_pos_edge_index, test_neg_edge_index, x_test,train_pos_edge_label_index):
    model.eval()
    with torch.no_grad():
        z = model.encode(x_test, train_pos_edge_label_index)
    return model.test(z, test_pos_edge_index, test_neg_edge_index)




class Environment:
    def __init__(self, contexts : FeatureExtractor, normalize_reward : bool = False):
        """Environment base class.
            TODO are contexts actually part of the concept of an RL environment?
            """
        self.contexts = contexts
        self.normalize_reward = normalize_reward

    def reset_from_copy(self):
        self.contexts = [FeatureExtractor(context.composition.reset_from_copy()) for context in self.contexts]
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
        return [self.contexts[0].extract(trans) for trans in self.contexts[0].getFrontier()]


if __name__=="__main__":
    d = CompositionGraph("AT", 3, 3)
    d.start_composition()
    da = FeatureExtractor(d)

    da.train_gae_on_full_graph()
    breakpoint()
    d.load()


