from extractor import FeatureExtractor
from agent import *
from agent import TrainingSession
from environment import EnvironmentRefactored
class Experiment:
    def __init__(self, args : argparse.Namespace, problem : str):
        self.args = args
        self.problem = problem
        self.results_path = "./experiments/" + args.exp_path + "/" + self.problem + "/"
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
        self.args = args

        #self.env_refactor = EnvironmentRefactored(CompositionGraph(self.training_contexts))
        #self.reward etc
    def init_agent(self, agent=None):
        nfeatures = self.env.contexts[0].get_transition_features_size() #TODO refactor, hardcoded
        agent = DQNAgent(self.args, save_file=self.results_path, verbose=False, nfeatures=nfeatures) if agent is None else agent
        self.write_description()
        return agent
    def compute_step_state(self):
        raise NotImplementedError
    def run_refactored(self):
        # TrainingSession should have the summarywriter

        agent_refactor = DQNAgentRefactored(self.args,FeatureExtractor)

        session = TrainingSession(self.agent,self.env_refactor,n_agents_budget=150)

        session.run()


    def run(self):
        assert self.args.overwrite or not os.path.exists(self.results_path + "finished.txt"), \
            "Experiment is already fully trained, training would override" \
            " previous agents."

        self.partially_trained = True
        #TODO self.print_training_characteristics()

        #self.args.nfeatures = self.nfeatures

        self.agent.set_feature_extractor(self.env.contexts[0].composition) #FIXME this parameter pass is horrible haha
        if self.args.exp_replay:
            self.agent._initializeBuffer(self.env, self.args.buffer_size)
        self.agent.set_training_environment(self.env)

        self.agent.train(max_steps=self.args.training_steps,
                         save_freq=self.args.save_freq,
                         save_at_end=True,
                         early_stopping=self.args.early_stopping,
                         results_path=self.results_path + self.add_nk(),
                         )

        with open(self.results_path + "training_data.pkl", "wb") as f:
            #FIXME info dropped from saved tuple (see Learning-synthesis)
            pickle.dump((self.agent.training_data, self.args), f)

        self.flag_as_fully_trained()

    def add_nk(self):
        res = str(list(self.env.contexts[0].composition.info().values())[:2])
        return res

    def write_description(self):
        print("TODO write description")

    def flag_as_fully_trained(self):
        with open(self.results_path + "finished.txt", 'w') as f:
            f.write("Fully trained. This function should write a summary of training stats in the future.")  # FIXME
    def __str__(self):
        raise NotImplementedError

def debug_graph_inference():
    from features import GAEEmbeddings
    tcg = TrainingCompositionGraph("AT",2,2)
    e = GAEEmbeddings(None)
    tcg.start_composition()
    for i in range(4):
        tcg.expand(0)
        e.compute(tcg)


        breakpoint()
    breakpoint()

if __name__ == "__main__":

    debug_graph_inference()
    problems = ["AT","TL","BW","CM","DP","TA"]
    for problem in problems:
        exp = RLTrainingExperiment(parse_args(), problem, (2,2))
        exp.run()
