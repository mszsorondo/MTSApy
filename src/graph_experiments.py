import torch
from torch.utils.tensorboard import SummaryWriter
from environment import *
import datetime
from torch_geometric.datasets import Planetoid
TRAINABLE_FEATURES = [AutoencoderEmbeddings]
class TrainableFeatureExtractor(FeatureExtractor):
    def __init__(self, composition, enabled_features_dict = None, feature_classes = LEARNING_SYNTHESIS_BENCHMARK_FEATURES + TRAINABLE_FEATURES):
        super().__init__(composition,enabled_features_dict)
        self._trainable_features = [feature_cls for feature_cls in  self._feature_classes if hasattr(feature_cls, "train")]

    def train(self, feature : Feature):
        assert(feature in self._trainable_features)
        feature.train(self.composition,self)

class DynamicEdgeSampler:
    """
    Motivation:
    Sparse matrices with large number of nodes can be a problem when generating a negative edge index.
    Explicit generation and storage of this indices is very costly in terms of memory and time.

    """
class NodePairSplitter:
    def __init__(self, data, split_labels=True, add_negative_train_samples=True, test_prop = 0.1, proportional=False):
        Warning("Sending split tensors to DEVICE may be convenient if using accelerators (TODO).")

        n_nodes = self.n_nodes = data.x.shape[0]
        n_edges = self.n_edges = data.edge_index.shape[1]
        n_neg_edges = (n_nodes ** 2) - n_edges if proportional else n_nodes

        test_edge_index_idx = np.random.randint(0,n_edges,int(test_prop * n_edges))
        train_edge_index_idx = [i for i in range(n_edges) if i not in test_edge_index_idx]

        #neg_edge_index = torch.tensor([(i,j) for i in range(n_nodes) for j in range(n_nodes) if torch.tensor((i,j)) not in data.edge_index.T])
        #assert(neg_edge_index.shape[0] == (n_edges ** 2)-n_edges)
        self.pos_training_edge_index = data.edge_index.T[train_edge_index_idx].tolist()
        self.pos_testing_edge_index = data.edge_index.T[test_edge_index_idx].tolist()

        """edge_index_as_list =  data.edge_index.T.tolist()
        edge_index_mtx = [[0 for i in range(n_nodes)] for j in range(n_nodes)]
        for k in edge_index_as_list: edge_index_mtx[k[0]][k[1]] = 1
        neg_edge_index = []
        for i in range(n_nodes):
            for j in range(n_nodes):
                print([i,j])
                if edge_index_mtx[i][j]==0: neg_edge_index.append([i,j])"""
        #neg_edge_index = [[i,j] for i in range(n_nodes) for j in range(n_nodes) if [i,j] not in data.edge_index.T.tolist()]

        edge_index_tensor = torch.tensor(data.edge_index, device="cpu")
        edge_index_mtx = torch.zeros((n_nodes, n_nodes), device="cpu")
        edge_index_mtx[edge_index_tensor[0], edge_index_tensor[1]] = 1
        edge_index_mtx_bool = edge_index_mtx == 0
        neg_edge_index = torch.nonzero(edge_index_mtx_bool, as_tuple=True)
        neg_edge_index = list(zip(neg_edge_index[0].tolist(), neg_edge_index[1].tolist()))

        neg_test_edge_index_idx = np.random.randint(0, len(neg_edge_index), int(test_prop * len(neg_edge_index)))
        #neg_train_edge_index_idx = [i for i in range(len(neg_edge_index)) if i not in neg_test_edge_index_idx]
        neg_train_edge_index_idx = np.setdiff1d(np.arange(len(neg_edge_index)), neg_test_edge_index_idx)

        self.neg_testing_edge_index = torch.tensor(neg_edge_index)[neg_test_edge_index_idx].T
        self.neg_training_edge_index = torch.tensor(neg_edge_index)[neg_train_edge_index_idx].T
        self.pos_training_edge_index = torch.tensor(self.pos_training_edge_index).T
        self.pos_testing_edge_index = torch.tensor(self.pos_testing_edge_index).T
        breakpoint()

    def get_split(self):
        return self.pos_training_edge_index, self.neg_training_edge_index, self.pos_testing_edge_index, self.neg_testing_edge_index






def train_gae_on_full_graph(self : FeatureExtractor, to_undirected = True, epochs = 5000, debug_graph = None):
    #FIXME this should be converted into a Feature class in the future
    #FIXME FIXME the inference is being performed purely on edges!!!!!!!!!!!
    #from torch_geometric.transforms import RandomLinkSplit
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    self.composition.full_composition()

    #writer = SummaryWriter(rf".runs/feature_trains/{str((self.composition._problem, self.composition._n, self.composition._k))}_at_{str(datetime.datetime.now())}", \
    #                       filename_suffix=f"{str((self.composition._problem, self.composition._n, self.composition._k))}_at_{str(datetime.datetime.now())}")
    #writer.add_text("training data", f"{str(self.composition)}"+f"{str(self)}")

    print(len(self.composition.nodes()), len(self.composition.edges()))
    edge_features = self.non_frontier_feature_vectors()

    node_features = self.static_node_features()
    CG = self.composition

    # fill attrs with features:
    for ((s, t), features) in edge_features.items():
        CG[s][t]["features"] = features


    D = CG.to_pure_nx()
    G = CG.copy_with_nodes_as_ints(D)
    if to_undirected: G = G.to_undirected() #FIXME what about double edges between nodes?

    breakpoint()
    data = from_networkx(G, group_node_attrs=["features"]) if debug_graph is None else debug_graph[0].to(device)

    Warning("We should use RandomNodeSplit")
    Warning("How are negative edge features obtained?")
    splitter = NodePairSplitter(data)
    p_tr, n_tr, p_test, n_test = splitter.get_split()


    out_channels = 2
    #breakpoint()
    num_features = data.x.shape[1]
    # TODO adapt for RandomLinkSplit, continue with tutorial structure
    # model
    model = GAE(GCNEncoder(num_features, out_channels))


    model = model.to(device)
    node_features = data.x.to(device)
    Warning("This features are only of connected nodes")
    #x_train = train_data.edge_attr.to(device)
    #x_test = test_data.edge_attr.to(device)

    #train_pos_edge_label_index = train_data.pos_edge_label_index.to(device)

    #FIXME how are neg edge features computed if inference is done on edge features and not node features?
    #train_neg_edge_label_index = train_data.neg_edge_label_index.to(device)  # TODO .encode and add to loss and EVAL
    # inizialize the optimizer

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        #, , p_test, n_test
        #breakpoint()
        loss = train(model, optimizer, node_features, p_tr, n_tr)
        auc, ap = test(model, p_test, n_test, node_features,
                       data.edge_index, n_tr)
        print('Epoch: {:03d}, AUC: {:.4f}, AP: {:.4f}'.format(epoch, auc, ap))
        #writer.add_scalar("losses/loss", loss, epoch)
        #writer.add_scalar("charts/SPS", int(epoch / (time.time() - start_time)), epoch)
        #writer.add_scalar("metrics/AUC", auc, epoch)
        #writer.add_scalar("metrics/AP", ap, epoch)
    #writer.close()

FeatureExtractor.train_gae_on_full_graph = train_gae_on_full_graph

if __name__=="__main__":
    ENABLED_PYTHON_FEATURES = {
        EventLabel: False,
        StateLabel: False,
        Controllable: False,
        MarkedSourceAndSinkStates: False,
        CurrentPhase: False,
        ChildNodeState: False,
        UncontrollableNeighborhood: False,
        ExploredStateChild: False,
        IsLastExpanded: False,
        RandomTransitionFeature : False,
        RandomOneHotTransitionFeature : True,
    }
    enable_first_n_values(ENABLED_PYTHON_FEATURES, 9)
    d = CompositionGraph("AT", 2, 2)
    d.start_composition()
    da = FeatureExtractor(d, ENABLED_PYTHON_FEATURES, feature_classes=ENABLED_PYTHON_FEATURES.keys())


    da.train_gae_on_full_graph(to_undirected=True, epochs=100000, debug_graph = Planetoid("./data", "Cora", split="full"))
    """for i in range(1,len(ENABLED_PYTHON_FEATURES.keys())):
        d = CompositionGraph("AT", 3, 3)
        d.start_composition()
        enable_first_n_values(ENABLED_PYTHON_FEATURES, i)
        print(ENABLED_PYTHON_FEATURES)
        da = FeatureExtractor(d,ENABLED_PYTHON_FEATURES, feature_classes=ENABLED_PYTHON_FEATURES.keys())

        da.train_gae_on_full_graph(to_undirected=True, epochs=200)
"""
    #d.load()
