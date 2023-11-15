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


    """
    class DynamicEdgeSampler:
    Motivation:
    Sparse matrices with large number of nodes can be a problem when generating a negative edge index.
    Explicit generation and storage of this indices is very costly in terms of memory and time.

    """
class NodePairSplitter:
    def __init__(self, data, split_labels=True, add_negative_train_samples=True, val_prop = 0.05,test_prop = 0.1, proportional=False):
        Warning("Sending split tensors to DEVICE may be convenient if using accelerators (TODO).")

        n_nodes = self.n_nodes = data.x.shape[0]
        n_edges = self.n_edges = data.edge_index.shape[1]
        n_neg_edges = (n_nodes ** 2) - n_edges if proportional else n_edges

        test_edge_index_idx = np.random.randint(0,n_edges,int(test_prop * n_edges))
        #val_edge_index_idx = np.random.randint(0, n_edges, int(val_prop * n_edges))
        train_edge_index_idx = [i for i in range(n_edges) if i not in test_edge_index_idx]

        #neg_edge_index = torch.tensor([(i,j) for i in range(n_nodes) for j in range(n_nodes) if torch.tensor((i,j)) not in data.edge_index.T])
        #assert(neg_edge_index.shape[0] == (n_edges ** 2)-n_edges)
        self.pos_training_edge_index = data.edge_index.T[train_edge_index_idx].tolist()
        self.pos_testing_edge_index = data.edge_index.T[test_edge_index_idx].tolist()

        self.neg_testing_edge_index = []
        self.neg_training_edge_index = []

        while(len(self.pos_training_edge_index)<len(self.neg_training_edge_index)):
            raise NotImplementedError
            #i = np.ra




        self.pos_training_edge_index = torch.tensor(self.pos_training_edge_index).T
        self.pos_testing_edge_index = torch.tensor(self.pos_testing_edge_index).T



    def get_split(self):
        return self.pos_training_edge_index, self.neg_training_edge_index, self.pos_testing_edge_index, self.neg_testing_edge_index

def train_vgae_official(file_name = "vgae.pt"):
    import pickle
    import sys
    sys.path.append("/home/marco/Desktop/dgl/dgl/examples/pytorch/vgae")
    import train_vgae
    import dgl
    from torch_geometric.utils import to_dgl
    for problem in ["AT", "DP","TA", "TL", "BW", "CM"]:
        d = CompositionGraph(problem, 2, 2)
        d.start_composition()
        d.full_composition()

        da = FeatureExtractor(d, ENABLED_PYTHON_FEATURES, feature_classes=ENABLED_PYTHON_FEATURES.keys())

        data, device = da.composition_to_nx()

        dgl_data = to_dgl(data)
        best_model = train_vgae.dgl_main(dgl_data) #TODO add parameters: graph, epochs, etc etc
        Warning("I'm not so sure the parameters are correctly loaded or if the parameters are from the best model (watch out running mean and variance etc)")
        torch.save(best_model, problem + file_name)

def train_gae_on_full_graph(self : FeatureExtractor, to_undirected = True, epochs = 5000, debug_graph = None):
    Warning("This function will be replaced by the official VGAE implementation from DGL")
    #FIXME this should be converted into a Feature class in the future
    #FIXME FIXME the inference is being performed purely on edges!!!!!!!!!!!
    #from torch_geometric.transforms import RandomLinkSplit
    data, device = self.composition_to_nx(debug_graph, to_undirected)

    Warning("We should use RandomNodeSplit")
    Warning("How are negative edge features obtained?")
    splitter = NodePairSplitter(data)
    p_tr, n_tr, p_test, n_test = splitter.get_split()


    out_channels = 2

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


def composition_to_nx(self, debug_graph=None, to_undirected=True, selected_transitions_to_inspect = []):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(len(self.composition.nodes()), len(self.composition.edges()))
    edge_features = self.non_frontier_feature_vectors()
    self.set_static_node_features()
    CG = self.composition
    # fill attrs with features:
    selected_actions_to_inspect = []
    for ((s, t), features) in edge_features.items():
        #if CG[s][t]["label"] in selected_transitions_to_inspect:
            #selected_actions_to_inspect.append((s.toString(),t.toString(),CG[s][t]["label"]))
        for edge in CG[s][t].values(): edge["features"] = features

    D = CG.to_pure_nx()

    G = CG.copy_with_nodes_as_ints(D)
    if to_undirected: G = G.to_undirected()  # FIXME what about double edges between nodes?

    data = from_networkx(G, group_node_attrs=["features"], group_edge_attrs=["label"]) if debug_graph is None else debug_graph[0].to(device)
    data.feat = data.x
    return data, device


FeatureExtractor.train_gae_on_full_graph = train_gae_on_full_graph
FeatureExtractor.composition_to_nx = composition_to_nx


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
        RandomOneHotTransitionFeature : False,
    }
    enable_first_n_values(ENABLED_PYTHON_FEATURES, 9)

    #PAPER IMPLEMENTATION
    train_vgae_official()

    breakpoint()
    #OUR IMPLEMENTATION
    d = CompositionGraph("TL", 2, 2)
    d.start_composition()
    i = 10
    while(i): d.expand(0);i-=1;
    fe = FeatureExtractor(d)
    fe.frontier_feature_vectors()
    breakpoint()

    d.full_composition()



    da = FeatureExtractor(d, ENABLED_PYTHON_FEATURES, feature_classes=ENABLED_PYTHON_FEATURES.keys())
    breakpoint()
    data,device = da.composition_to_nx()

    da.train_gae_on_full_graph(to_undirected=True, epochs=100000)
    """for i in range(1,len(ENABLED_PYTHON_FEATURES.keys())):
        d = CompositionGraph("AT", 3, 3)
        d.start_composition()
        enable_first_n_values(ENABLED_PYTHON_FEATURES, i)
        print(ENABLED_PYTHON_FEATURES)
        da = FeatureExtractor(d,ENABLED_PYTHON_FEATURES, feature_classes=ENABLED_PYTHON_FEATURES.keys())

        da.train_gae_on_full_graph(to_undirected=True, epochs=200)
"""
    #d.load()
