import torch
from bidict import bidict
from torch_geometric.nn import GAE
from torch_geometric.utils import from_networkx
from ..util import remove_indices as util_remove_indices
from src.features.features import EventLabel, StateLabel, Controllable, MarkedState, CurrentPhase, ChildNodeState, \
    UncontrollableNeighborhood, ExploredStateChild, IsLastExpanded, GCNEncoder, train, test
from src.util import remove_indices


class FeatureExtractor:
    """class used to get Composition information, usable as hand-crafted features
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
        self._feature_classes = [EventLabel,StateLabel, Controllable, MarkedState,CurrentPhase,
                                 ChildNodeState,UncontrollableNeighborhood,ExploredStateChild,IsLastExpanded]

    def phi(self):
        return self.frontier_features()

    def extract(self, transition, state):
        res = []
        for feature in self._feature_classes:
            res += feature.compute(transition,state)
        return res

    def _set_transition_type_bit(self, feature_vec_slice, transition):
        no_idx_label = self.remove_indices(transition.toString())
        feature_vec_slice_pos = self._fast_no_indices_alphabet_dict[no_idx_label]
        feature_vec_slice[feature_vec_slice_pos] = 1


    def remove_indices(self, transition_label : str):
        util_remove_indices(transition_label)
    def get_transition_features_size(self):
        if(len(self.composition.getFrontier())): return len(self.extract(self.composition.getFrontier()[0], self.composition))
        elif (len(self.composition.getNonFrontier())): return len(self.extract(self.composition.getNonFrontier()[0], self.composition))
        else: raise ValueError
    def non_frontier_features(self):
        # TODO you can parallelize this (GPU etc)
        return {(trans.state,trans.child) : self.extract(trans) for trans in self.composition.getNonFrontier()}

    def frontier_features(self):
        #TODO you can parallelize this (GPU etc)
        return {(trans.state,trans.child) : self.extract(trans) for trans in self.composition.getFrontier()}

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
