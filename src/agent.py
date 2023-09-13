from util import *
from composition import CompositionGraph
import torch


class Feature:
    def __init__(self, str_id):
        self.str_id = str_id
        # TODO self.size ??
        # TODO self._mem may be useful for general features or for avoiding redundant computations
        raise NotImplementedError
    def compute(self, dataset : CompositionGraph):
        raise NotImplementedError
    def get_size(self, dataset : CompositionGraph):
        return len(self.compute(dataset))
    def __str__(self):
        return self.str_id


class DQNAgent:
    def __init__(self, args, nn_model : torch.nn.Module, save_file=None, verbose=False):
        Warning("TODO: refactor this initialization")
        assert nn_model is not None

        self.args = args
        self.model = nn_model

        self.target = None
        self.buffer = None

        self.save_file = save_file
        self.save_idx = 0

        self.training_start = None
        self.training_steps = 0
        self.epsilon = args.first_epsilon

        self.verbose = verbose

        self.training_data = []

        self.best_training_perf = {}
        self.last_best = None
        self.converged = False

    def set_model(self, features : list[Feature], context : CompositionGraph, prebuilt = None):
        if prebuilt is not None:
            self.model = prebuilt
            return
        print("Inferred default model should also be possible in the future")
        raise NotImplementedError
        input_size = sum([feature.get_size(context) for feature in features])




