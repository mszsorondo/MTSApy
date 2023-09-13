from util import *
from composition import CompositionGraph
from agent import *

class ExperimentStats
class Experiment:
    def __init__(self, args : argparse.Namespace, problem : str):
        self.args = args
        self.problem = problem
        self.results_path = "../experiments/" + args.exp_path + "/" + self.problem + "/"

class TrainingExperiment(Experiment):
    def __init__(self, args : argparse.Namespace, problem : str, context : tuple[int,int]):
        super.__init__(args,problem)
        self.training_contexts = [(problem, context[0], context[1])]
        self.partially_trained = False
        #we no longer use self.features nor self.nfeatures, self.env and self.agent will give compute this information

    def train(self): raise NotImplementedError

class RLTrainingExperiment(TrainingExperiment):
    def __init__(self, args : argparse.Namespace, problem : str, context : tuple[int,int], ):
        super.__init__(args, problem, context)
        self.env = [CompositionGraph(p, n, k) for p, n, k in self.training_contexts]
        self.agent = self.init_agent()
        #self.reward etc

    def init_agent(self, agent):
        agent = DQNAgent(self.args, save_file=self.results_path, verbose=False, nn_model=default_network()) if agent is not None else agent
        self.write_description()
        return agent
    def compute_step_state(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError