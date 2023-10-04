from composition import Environment
from src.extractor import FeatureExtractor
from agent import *

class Experiment:
    def __init__(self, args : argparse.Namespace, problem : str):
        self.args = args
        self.problem = problem
        self.results_path = "../experiments/" + args.exp_path + "/" + self.problem + "/"
    def write_info_to_file(self):
        raise NotImplementedError
class TrainingExperiment(Experiment):
    def __init__(self, args : argparse.Namespace, problem : str, context : tuple[int,int]):
        super().__init__(args,problem)
        self.training_contexts = [(problem, context[0], context[1])]
        self.partially_trained = False
        #we no longer use self.features nor self.nfeatures, self.env and self.agent will give compute this information

    def train(self): raise NotImplementedError

class RLTrainingExperiment(TrainingExperiment):
    def __init__(self, args : argparse.Namespace, problem : str, context : tuple[int,int], ):
        super().__init__(args, problem, context)
        self.env = Environment([FeatureExtractor(CompositionGraph(p, n, k).start_composition()) for p, n, k in self.training_contexts])
        self.agent = self.init_agent()
        #self.reward etc
    def init_agent(self, agent=None):
        nfeatures = self.env.contexts[0].get_transition_features_size() #TODO refactor, hardcoded
        agent = DQNAgent(self.args, save_file=self.results_path, verbose=False, nfeatures=nfeatures) if agent is None else agent
        self.write_description()
        return agent
    def compute_step_state(self):
        raise NotImplementedError

    def run(self):
        assert self.args.overwrite or not os.path.exists(self.results_path + "finished.txt"), \
            "Experiment is already fully trained, training would override" \
            " previous agents."

        self.partially_trained = True
        #TODO self.print_training_characteristics()

        #self.args.nfeatures = self.nfeatures

        if self.args.exp_replay:
            self.agent._initializeBuffer(self.env, self.args.buffer_size)
        self.agent.set_training_environment(self.env)

        self.agent.train(max_steps=self.args.training_steps,
                         save_freq=self.args.save_freq,
                         save_at_end=True,
                         early_stopping=self.args.early_stopping,
                         results_path=self.results_path)

        with open(self.results_path + "training_data.pkl", "wb") as f:
            pickle.dump((self.agent.training_data, self.args, self.env[self.training_contexts[0]].info), f)

        self.flag_as_fully_trained()

    def write_description(self):
        print("TODO write description")
    def __str__(self):
        raise NotImplementedError


if __name__ == "__main__":
    breakpoint()
    exp = RLTrainingExperiment(parse_args(), "AT", (2,2))
    exp.run()
    i = 0