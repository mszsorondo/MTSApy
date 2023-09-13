from util import *
import torch
class DQNAgent:
    def __init__(self, args, nn_model : torch.nn.Module, save_file=None, verbose=False):
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
