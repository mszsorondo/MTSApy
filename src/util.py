import networkx as nx
import jpype
import jpype.imports
import matplotlib.pyplot as plt
from bidict import bidict
import torch
import numpy as np
import argparse
import os, time
import onnx

import pickle
FSP_PATH = "./fsp"
BENCHMARK_PROBLEMS = ["AT", "BW", "CM", "DP","TA","TL"]


class Model:
    def __init__(self):
        pass

    def predict(self, s):
        raise NotImplementedError()

    def eval_batch(self, obss):
        raise NotImplementedError()

    def eval(self, s):
        raise NotImplementedError()

    def best(self, s):
        raise NotImplementedError()

    def current_loss(self):
        raise NotImplementedError()
class OnnxModel(Model):
    def __init__(self, model):
        super().__init__()
        assert model.has_learned_something

        self.onnx_model, self.session = model.to_onnx()

    def save(self, path):
        onnx.save(self.onnx_model, path + ".onnx")

    def predict(self, s):
        if s is None:
            return 0
        return self.session.run(None, {'X': s})[0]

    def eval_batch(self, ss):
        return np.array([self.eval(s) for s in ss])

    def eval(self, s):
        return np.max(self.predict(s))

    def current_loss(self):
        raise NotImplementedError()

class TorchModel(Model):

    def __init__(self, args, nfeatures, network=None):
        super().__init__()
        self.nfeatures = nfeatures
        self.n, self.k = None, None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using", self.device, "device")
        self.model = network
        print(self.model)
        print("Learning rate:", args.learning_rate)

        self.loss_fn = torch.nn.MSELoss()

        self.optimizer = torch.optim.SGD(self.model.parameters(),
                                         lr=args.learning_rate,
                                         momentum=args.momentum,
                                         nesterov=args.nesterov,
                                         weight_decay=args.weight_decay)

        self.has_learned_something = False

        self.losses = []

    def eval_batch(self, ss):
        return np.array([self.eval(s) for s in ss])

    def eval(self, s):
        if not self.has_learned_something or s is None:
            return 0
        return float(self.predict(s).max())

    def best(self, s):
        if not self.has_learned_something or s is None:
            return 0
        return int(self.predict(s).argmax())

    def predict(self, s):
        if not self.has_learned_something or s is None:
            return 0
        return self.model(torch.tensor(s).to(self.device))

    def single_update(self, s, value):
        return self.batch_update(np.array([s]), np.array([value]))

    def batch_update(self, ss, values):

        ss = torch.tensor(ss).to(self.device)
        values = torch.tensor(values, dtype=torch.float, device=self.device).reshape(len(ss), 1)

        self.optimizer.zero_grad()
        pred = self.model(ss)

        loss = self.loss_fn(pred, values)
        loss.backward()
        self.optimizer.step()

        self.losses.append(loss.item())
        self.has_learned_something = True

    def nfeatures(self):
        return self.nfeatures

    # should be called only at the end of each episode
    def current_loss(self):
        avg_loss = np.mean(self.losses)
        self.losses = []
        return avg_loss


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--problems", type=str,
                        help="The set of target problems: e.g. \"AT-BW-CM-DP-TA-TL\"",
                        default="AT-BW-CM-DP-TA-TL")

    parser.add_argument("--exp-path", type=str, default="test",
                        help="The path of this experiment inside results")

    parser.add_argument("--step-2-results", type=str, default="step_2_results.csv",
                        help="The filename for the results of step 2 inside the experiment path")

    parser.add_argument("--step-3-results", type=str, default="step_3_results.csv",
                        help="The filename for the results of step 3 inside the experiment path")

    parser.add_argument("--desc", type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help="A description for this experiment")

    parser.add_argument("--learning-rate", type=float, default=1e-5,
                        help="The learning rate of the optimizer")

    parser.add_argument("--first-epsilon", type=float, default=1.0,
                        help="The initial rate of exploration of the epsilon-greedy policy")

    parser.add_argument("--last-epsilon", type=float, default=0.01,
                        help="The final rate of exploration of the epsilon-greedy policy")

    parser.add_argument("--epsilon-decay-steps", type=int, default=250000,
                        help="The number of decay steps for the rate of exploration")

    parser.add_argument("--target-q", action=argparse.BooleanOptionalAction, default=True,
                        help="Toggle usage of Fixed Q-Target")

    parser.add_argument("--reset-target-freq", type=int, default=10000,
                        help="Number of steps between target function updates (if using Fixed Q-Targets)")

    parser.add_argument("--exp-replay", action=argparse.BooleanOptionalAction, default=True,
                        help="Toggle usage of Experience Replay")

    parser.add_argument("--buffer-size", type=int, default=10000,
                        help="Experience Replay buffer size")

    parser.add_argument("--batch-size", type=int, default=10,
                        help="Mini-batch size (if using Experience Replay)")

    parser.add_argument("--n-step", type=int, default=1,
                        help="Lookahead size for n-step Q-Learning")

    parser.add_argument("--nesterov", action=argparse.BooleanOptionalAction, default=True,
                        help="Toggle usage of Nesterov momentum with SGD")

    parser.add_argument("--momentum", type=float, default=0.9,
                        help="SGD momentum")

    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="SGD weight decay")

    parser.add_argument("--training-steps", type=int, default=500000,
                        help="Steps of the training algorithm. \
                        If using early stopping these are the minimum training steps.")

    parser.add_argument("--save-freq", type=int, default=5000,
                        help="The number of training steps between each set of weights saved.")

    parser.add_argument("--early-stopping", action=argparse.BooleanOptionalAction, default=True,
                        help="Toggle usage of early stopping (stoppping training when no improvement"
                             " are shown after a signicant number of steps)")

    parser.add_argument("--step-2-n", type=int, default=100,
                        help="Number of agents to be tested in step 2")

    parser.add_argument("--step-2-budget", type=int, default=5000,
                        help="Expansions budget for step 2")

    parser.add_argument("--step-3-budget", type=int, default=15000,
                        help="Expansions budget for step 3")

    parser.add_argument("--ra", action=argparse.BooleanOptionalAction, default=False,
                        help="Toggle usage of the ra feature")

    parser.add_argument("--labels", action=argparse.BooleanOptionalAction, default=True,
                        help="Toggle usage of the labels features")

    parser.add_argument("--context", action=argparse.BooleanOptionalAction, default=True,
                        help="Toggle usage of the context features")

    parser.add_argument("--state-labels", type=int, default=1,
                        help="Size of state labels feature history (0 to disable the feature)")

    parser.add_argument("--je", action=argparse.BooleanOptionalAction, default=True,
                        help="Toggle usage of the just explored feature.")

    parser.add_argument("--nk", action=argparse.BooleanOptionalAction, default=False,
                        help="Toggle usage of the nk feature.")

    parser.add_argument("--prop", action=argparse.BooleanOptionalAction, default=False,
                        help="Toggle usage of the prop feature.")

    parser.add_argument("--visits", action=argparse.BooleanOptionalAction, default=False,
                        help="Toggle usage of the visits feature.")

    parser.add_argument("--ltr", action=argparse.BooleanOptionalAction, default=False,
                        help="Toggle usage of the labelsThatReach feature.")

    parser.add_argument("--boolean", action=argparse.BooleanOptionalAction, default=True,
                        help="Toggle usage only boolean fatures.")

    parser.add_argument("--cbs", action=argparse.BooleanOptionalAction, default=False,
                        help="Toggle usage of the components by state feature.")

    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False,
                        help="Allows saving experiment on a path that already has results")

    parser.add_argument("--max-instance-size", type=int, default=15,
                        help="Maximum value for parameters (n, k) on steps 2 and 3")

    parser.add_argument("--exploration-graph", action=argparse.BooleanOptionalAction, default=False,
                        help="Enables the re-construction of the exploration graph in python, to be used by an agent.")

    parser.add_argument("--enable-autoencoder", action=argparse.BooleanOptionalAction, default=False,
                        help="Enables usage of graph embeddings as feature")

    args = parser.parse_args()

    return args

class NeuralNetwork(torch.nn.Module):
    def __init__(self, nfeatures, nnsize):
        super(NeuralNetwork, self).__init__()
        nnsize = list(nnsize)
        layers = [torch.nn.Linear(nfeatures, nnsize[0])]
        for i in range(len(nnsize)-1):
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Linear(nnsize[i], nnsize[i+1]))
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def reuse_onnx_model(self, onnx_path):
        raise NotImplementedError


def default_network(args, nfeatures, nn_size=[20]):
    nn = NeuralNetwork(nfeatures, nn_size).to("cpu")
    nn_model = TorchModel(args, nfeatures, network=nn)
    return nn_model

def remove_indices(transition_label : str):
    res = ""

    for c in transition_label:
        if not c.isdigit() and c!='.': res += c

    return res

LABELS_PATH = "./fsp/labels"
