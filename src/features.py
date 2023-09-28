from util import *
from torch_geometric.nn import GCNConv, GAE, Sequential
from torch_geometric.utils import from_networkx
from torch import nn
from composition import CompositionGraph, util_remove_indices
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

class FeatureExtractor:
    """class used to get Composition information, usable as hand-crafted features
        TODO this class will be replaced by object-oriented Feature class

        Design:
        attrs: Features (list of classes)
        methods: .extract(featureClass, from = composition) .phi(composition)
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
        self._feature_classes = [EventLabel, ]

    def phi(self):
        return self.frontier_features()
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
        return [float(self.composition.getLastExpanded().state==transition.state), float(self.composition.getLastExpanded().child==transition.state)]

    def remove_indices(self, transition_label : str):
        util_remove_indices(transition_label)
    def get_transition_features_size(self):
        if(len(self.composition.getFrontier())): return len(self.compute_features(self.composition.getFrontier()[0]))
        elif (len(self.composition.getNonFrontier())): return len(self.compute_features(self.composition.getNonFrontier()[0]))
        else: raise ValueError

    def compute_features(self, transition):
        res = []
        for feature_method in self._feature_methods:
            res += feature_method(transition)
        return res

    def non_frontier_features(self):
        # TODO you can parallelize this (GPU etc)
        return {(trans.state,trans.child) : self.compute_features(trans) for trans in self.composition.getNonFrontier()}

    def frontier_features(self):
        #TODO you can parallelize this (GPU etc)
        return {(trans.state,trans.child) : self.compute_features(trans) for trans in self.composition.getFrontier()}

    def train_gae_on_full_graph(self):
        from torch_geometric.transforms import RandomLinkSplit

        self.composition.full_composition()
        edge_features = self.non_frontier_features()

        CG = self.composition

        # fill attrs with features:
        for ((s, t), features) in edge_features.items():
            CG[s][t]["features"] = features

        D = CG.to_pure_nx()

        data = from_networkx(CG.copy_with_nodes_as_ints(D),group_edge_attrs=["features"])
        transform = RandomLinkSplit(split_labels=True, add_negative_train_samples=True, neg_sampling_ratio=2.0, disjoint_train_ratio = 0.0)
        train_data, val_data, test_data = transform(data)

        out_channels = 2
        num_features = self.get_transition_features_size()
        epochs = 10000
        #TODO adapt for RandomLinkSplit, continue with tutorial structure
        # model
        model = GAE(GCNEncoder(num_features, out_channels))

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        x_train = train_data.edge_attr.to(device)
        x_test = test_data.edge_attr.to(device)

        train_pos_edge_label_index = train_data.pos_edge_label_index.to(device)

        train_neg_edge_label_index = train_data.neg_edge_label_index.to(device) #TODO .encode and add to loss and EVAL
        # inizialize the optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        for epoch in range(1, epochs + 1):

            loss = train(model, optimizer,x_train,train_pos_edge_label_index)
            auc, ap = test(model,test_data.pos_edge_label_index, test_data.neg_edge_label_index, x_test, train_pos_edge_label_index)
            print('Epoch: {:03d}, AUC: {:.4f}, AP: {:.4f}'.format(epoch, auc, ap))
    def train_node2vec(self):
        raise NotImplementedError
    def train_watch_your_step(self):
        raise NotImplementedError
    def train_DGI(self):
        raise NotImplementedError


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
        return [self.contexts[0].compute_features(trans) for trans in self.contexts[0].getFrontier()]


if __name__=="__main__":
    d = CompositionGraph("AT", 3, 3)
    d.start_composition()
    da = FeatureExtractor(d)

    da.train_gae_on_full_graph()
    breakpoint()
    d.load()


