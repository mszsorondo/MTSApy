from torch.utils.tensorboard import SummaryWriter
from environment import *
import datetime
TRAINABLE_FEATURES = [AutoencoderEmbeddings]
class TrainableFeatureExtractor(FeatureExtractor):
    def __init__(self, composition, enabled_features_dict = None, feature_classes = LEARNING_SYNTHESIS_BENCHMARK_FEATURES + TRAINABLE_FEATURES):
        super().__init__(composition,enabled_features_dict)
        self._trainable_features = [feature_cls for feature_cls in  self._feature_classes if hasattr(feature_cls, "train")]
        breakpoint()
    def train(self, feature : Feature):
        assert(feature in self._trainable_features)
        feature.train(self.composition,self)
def train_gae_on_full_graph(self : FeatureExtractor, to_undirected = False, epochs = 5000):
    #FIXME this should be converted into a Feature class in the future
    #FIXME FIXME the inference is being performed purely on edges!!!!!!!!!!!
    Warning("Inference is being performed purely on edges!!!!!!!!!!!")
    from torch_geometric.transforms import RandomLinkSplit


    self.composition.full_composition()

    writer = SummaryWriter(f"runs/feature_trains/{str((self.composition._problem, self.composition._n, self.composition._k))}_at_{str(datetime.datetime.now())}", \
                           filename_suffix=f"{str((self.composition._problem, self.composition._n, self.composition._k))}_at_{str(datetime.datetime.now())}")
    writer.add_text("training data", f"{str(self.composition)}"+f"{str(self)}")

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
    data = from_networkx(G, group_edge_attrs=["features"])
    Warning("We should use RandomNodeSplit")
    Warning("How are negative edge features obtained?")
    transform = RandomLinkSplit(split_labels=True, add_negative_train_samples=True, neg_sampling_ratio=14.0,
                                disjoint_train_ratio=0.0)
    train_data, val_data, test_data = transform(data)

    out_channels = 2
    num_features = self.get_transition_features_size()
    # TODO adapt for RandomLinkSplit, continue with tutorial structure
    # model
    model = GAE(GCNEncoder(num_features, out_channels))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    Warning("This features are only of connected nodes")
    x_train = train_data.edge_attr.to(device)
    x_test = test_data.edge_attr.to(device)

    train_pos_edge_label_index = train_data.pos_edge_label_index.to(device)

    #FIXME how are neg edge features computed if inference is done on edge features and not node features?
    train_neg_edge_label_index = train_data.neg_edge_label_index.to(device)  # TODO .encode and add to loss and EVAL
    # inizialize the optimizer
    breakpoint()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        loss = train(model, optimizer, x_train, train_pos_edge_label_index, train_neg_edge_label_index)
        auc, ap = test(model, test_data.pos_edge_label_index, test_data.neg_edge_label_index, x_test,
                       train_pos_edge_label_index)
        print('Epoch: {:03d}, AUC: {:.4f}, AP: {:.4f}'.format(epoch, auc, ap))
        writer.add_scalar("losses/loss", loss, epoch)
        writer.add_scalar("charts/SPS", int(epoch / (time.time() - start_time)), epoch)
        writer.add_scalar("metrics/AUC", auc, epoch)
        writer.add_scalar("metrics/AP", ap, epoch)
    writer.close()

FeatureExtractor.train_gae_on_full_graph = train_gae_on_full_graph

if __name__=="__main__":
    d = CompositionGraph("AT", 3, 3)
    d.start_composition()
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
    d = CompositionGraph("DP", 2, 2)
    d.start_composition()
    da = FeatureExtractor(d, ENABLED_PYTHON_FEATURES, feature_classes=ENABLED_PYTHON_FEATURES.keys())

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
