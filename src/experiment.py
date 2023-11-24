import random

import numpy as np
import torch

from extractor import FeatureExtractor
from agent import *
from agent import TrainingSession
from environment import EnvironmentRefactored
from features import *

def selection_phase(args, agents_path, problem = "AT"):

    exp = RLTestingExperiment(args, f"{agents_path}/{problem}/[2, 2]/", problem)
    instances = [(i,k) for i in range(1,16) for k in range(1,16)]
    exp.run(instances, logs_path=rf".runs/testing_torch_save_and_early_stopping_with_gae/{problem}")

def best_agent_testing_phase(args, agent_path, problem = "AT"):
    raise NotImplementedError
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
        self.env = Environment([FeatureExtractor(TrainingCompositionGraph(p, n, k).start_composition(), global_feature_classes=[GAEEmbeddings(problem=p)]) for p, n, k in self.training_contexts])
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


class RLTestingExperiment():
    def __init__(self,args : argparse.Namespace, agents_path, problem):
        self.args = args
        self.agents_path = agents_path
        self.problem = problem

    def run(self, instances, logs_path):
        #agent_idxs = np.random.randint(0,min(100,len(??)))

        agent_paths = [self.agents_path +"/"+ f for f in get_filenames_in_path(self.agents_path) if ".pt" in f]
        agent_paths = sorted(random.sample(agent_paths,min(100,len(agent_paths))))
        solved_by_agent = {path:0 for path in agent_paths}
        n = max(instances)[0]
        solved = [[False for _ in range(n+1)] for _ in range(n+1)]
        path_to_folder = rf"{logs_path}/{str((self.problem))}_selection_at_{str(datetime.datetime.now())}"
        writer = SummaryWriter(
            path_to_folder, \
            filename_suffix=f"{self.problem}_at_{str(datetime.datetime.now())}")
        writer.add_text("selection phase data", f"{str(self.problem)}")
        best_agent_path, best_solved = None,0
        #TODO correlation between expanded transitions at a benchmark subset and total solved instances
        for i,agent_path in enumerate(agent_paths):
            for instance in instances:
                n,k = instance[0], instance[1]
                if not ((n==1 or solved[n-1][k]) and (k==1 or solved[n][k-1])):
                    continue
                tcg = TrainingCompositionGraph(self.problem, instance[0], instance[1])

                env = Environment([FeatureExtractor(tcg.start_composition(),
                                   global_feature_classes = [GAEEmbeddings(problem=self.problem)])])


                nfeatures = env.contexts[0].get_transition_features_size()  # TODO refactor, hardcoded

                agent = DQNAgent(self.args, verbose=False,nfeatures=nfeatures)
                agent.nn_model = torch.load(agent_path,)
                agent.set_feature_extractor(env.contexts[0].composition)
                agent.set_training_environment(env)
                remaining = self.args.step_2_budget

                while(remaining>0 and not tcg._javaEnv.isFinished()):
                    obs = agent.frontier_feature_vectors_as_batch()
                    a = agent.get_action(obs, 0)
                    tcg.expand(a)
                    remaining-=1
                if (remaining==0):
                    solved[n][k] = False
                    print(f"Failed {n} , {k} with {self.args.step_2_budget} total expansions")
                else:
                    print(f"Agent {i} solved {self.problem} at {n} , {k} with {self.args.step_2_budget-remaining} expansions")
                    solved[n][k] = True
                    solved_by_agent[agent_path]+=1
            writer.add_scalar("Solved Instances ",solved_by_agent[agent_path], i)
            if solved_by_agent[agent_path]>best_solved:
                best_solved=solved_by_agent[agent_path]
                best_agent_path = agent_path
            print(f"{i}-th agent solved {solved_by_agent[agent_path]} instances")
        with open(f"{path_to_folder}/best_agent_info.txt","w") as f:
            f.write(f"{best_agent_path}\n")
            f.write(str(best_solved))
        return best_agent_path, best_solved



def debug_graph_inference(problem="AT"):
    from features import GAEEmbeddings
    import sys
    sys.path.append("/home/marco/Desktop/dgl/dgl/examples/pytorch/vgae")
    import train_vgae
    import model as mdl
    tcg = TrainingCompositionGraph(problem,2,2)
    breakpoint()


    e = GAEEmbeddings(problem=problem)
    tcg.start_composition()
    res = None

    for i in range(16):
        tcg.expand(0)
        res = e.compute(tcg)




        if i==15: breakpoint()
    breakpoint()


if __name__ == "__main__":

    args = parse_args()
    problems = ["AT", "BW", "CM", "DP", "TA", "TL"]
    for problem in problems:
        selection_phase(args, agents_path="/home/marco/Desktop/MTSApy/experiments/testing_torch_save_and_early_stopping_with_gae", problem=problem)

    #debug_graph_inference()

    """for problem in problems:
        exp = RLTrainingExperiment(args, problem, (2,2))
        exp.run()"""
