import copy

import numpy as np

from util import *
from composition import CompositionGraph, TrainingCompositionGraph
from environment import Environment
from replay_buffer import ReplayBuffer
import torch
import json
from extractor import FeatureExtractor
from torch.utils.tensorboard import SummaryWriter
import datetime
from features import GAEEmbeddings
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

class DQNAgentRefactored:
    def __init__(self, args, feature_extractor,nn_model : torch.nn.Module = None):
        self.args = args
        self.feature_extractor = feature_extractor
        self.model = nn_model

    def set_model(self, context : CompositionGraph, prebuilt = None):
        #TODO Initialize a neural network Q with random weights and input dimension d_e
        #initialize Q' as a copy of Q
        #initialize buffer B with observations from a random policy
        if prebuilt is not None:
            self.model = prebuilt
            return
        #input_size = sum([feature.get_size(context) for feature in features])





class DQNAgent:
    def __init__(self, args, nfeatures, nn_model : torch.nn.Module = None,save_file=None, verbose=False):
        Warning("TODO: refactor this initialization")
        assert nn_model is not None or nfeatures is not None

        self.feature_extractor = None
        self.current_training_environment = None
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
        self.nfeatures = nfeatures

        if nn_model is None: self.model = default_network(args,nfeatures) #TODO refactor this line as well



    def set_feature_extractor(self, composition : CompositionGraph):
        self.feature_extractor = FeatureExtractor(composition, global_feature_classes=[GAEEmbeddings(problem=composition._problem)])

    def _get_experience_from_random_policy(self, env : Environment, total_steps, nstep=1):
        """ TODO it is not ok for an agent to restart and execute the steps of the environment, refactor this
        A random policy is run for total_steps steps, saving the observations in the format of the replay buffer """
        states = []

        env.reset_from_copy()
        Warning("HERE obs is not actions, but the featurization of the frontier actions")
        obs = self.frontier_feature_vectors_as_batch()
        steps = 0

        last_steps = []
        for i in range(total_steps):
            action = np.random.randint(len(obs))
            last_steps.append(obs[action])

            obs2, reward, done, info = env.step(action)

            if done:
                for j in range(len(last_steps)):
                    states.append((last_steps[j], -len(last_steps) + j, None))
                last_steps = []
                env.reset_from_copy()
                Warning("HERE obs is not actions, but the featurization of the frontier actions")
                obs = self.frontier_feature_vectors_as_batch()
            else:
                if len(last_steps) >= nstep:
                    states.append((last_steps[0], -nstep, obs2))
                last_steps = last_steps[len(last_steps) - nstep + 1:]
                obs = obs2
            steps += 1
        return states

    def _initializeBuffer(self, env : Environment, buffer_size):
        """ Initialize replay buffer uniformly with experiences from a set of environments """
        exp_per_instance = buffer_size // env.get_number_of_contexts()

        print("Initializing buffer with", exp_per_instance, "observations per instance, and", env.get_number_of_contexts(), "instances.")

        self.buffer = ReplayBuffer(buffer_size)
        random_experience = self._get_experience_from_random_policy(env,total_steps=exp_per_instance,nstep=self.args.n_step)
        for action_features, reward, obs2 in random_experience:
            self.buffer.add(action_features, reward, obs2)
        print("Done.")

    def train(self,seconds=None, max_steps=None, max_eps=None, save_freq=200000, last_obs=None,
              early_stopping=False, save_at_end=False, results_path=None, n_agents_budget=100):
        assert self.current_training_environment is not None
        assert self.model is not None

        composition = self.current_training_environment.contexts[0].composition
        comp_info = composition.info()
        writer = SummaryWriter(rf".runs/gae_agent_trains/{str((comp_info))}_at_{str(datetime.datetime.now())}", \
                               filename_suffix=f"{str((comp_info['problem'], comp_info['n'], comp_info['k']))}_at_{str(datetime.datetime.now())}")
        writer.add_text("training data", f"{str(composition)}")

        session = TrainingSession(self, self.current_training_environment, seconds, max_steps, max_eps, save_freq, last_obs,
              early_stopping, save_at_end, results_path, n_agents_budget)
        if (last_obs is None):
            self.current_training_environment.reset_from_copy()

        current_reward, reward_list, episode_number = 0, [], 1
        obs = self.frontier_feature_vectors_as_batch() if (last_obs is None) else last_obs
        while(not session.finished()):
            a = self.get_action(obs, self.epsilon)
            session.last_steps.append(obs[a])
            obs2, reward, done, step_info = self.current_training_environment.step(a)
            current_reward -= 1
            #TODO refactor with DQN class and DQNExperienceReplay class or decorator with ex                                                                        perience replay or similar
            #also modifying session.last_steps this way looks horrible
            if self.args.exp_replay:
                if done:
                    for j in range(len(session.last_steps)):
                        self.buffer.add(session.last_steps[j], -len(session.last_steps) + j, None)
                    session.last_steps = []
                else:
                    if len(session.last_steps) >= self.args.n_step:
                        self.buffer.add(session.last_steps[0], -self.args.n_step, obs2)
                    session.last_steps = session.last_steps[len(session.last_steps) - self.args.n_step + 1:]
                #TODO how is this? see Learning-Synthesis agent.batch_update

                loss = self.batch_update()
                writer.add_scalar("Loss", loss, self.training_steps)
            else:
                self.update(obs, a, reward, obs2)

            if not done: obs = obs2
            else:
                #info = session.compute_final_info(step_info)
                self.current_training_environment.reset_from_copy()
                obs = self.frontier_feature_vectors_as_batch()

                reward_list.append(current_reward)
                current_reward = 0
                if (episode_number % 10 == 0): writer.add_scalar("Reward", sum(reward_list[-10:]) / 10, episode_number)
                episode_number += 1
            if self.training_steps % save_freq == 0 and results_path is not None:
                session.save_as_torch(self)
                #self.save(self.current_training_environment.info, path=results_path)
                session.n_agents_budget -= 1

            if self.training_steps % 10000 == 0:
                self.target = OnnxModel(self.model)
            self.training_steps += 1

            session.steps+=1

            if self.epsilon>self.args.last_epsilon+ 1e-10:
                writer.add_scalar("epsilon", self.epsilon, self.training_steps)
                self.epsilon -= session.epsilon_step
        writer.close()



    def set_training_environment(self, env: Environment):
        self.current_training_environment = env

    def from_pretrained(self):
        raise NotImplementedError
    def get_frontier_features(self):

        features = [self.current_training_environment.contexts[0].test_features_on_transition(transition) for transition in
                    self.current_training_environment.contexts[0].composition.getFrontier()]
        return np.asarray(features.copy())#TODO why copy? Memory cost of this?
    def frontier_feature_vectors(self) -> dict[tuple,list]:
        Warning("Values should be np.arrays or torch tensors (faster)")
        return self.feature_extractor.frontier_feature_vectors()

    def frontier_feature_vectors_as_batch(self):
        return self.feature_extractor.frontier_feature_vectors_as_batch()
    def get_action(self, s, epsilon):
        """ Gets epsilon-greedy action using self.model """
        if np.random.rand() <= epsilon:
            return np.random.randint(len(s))
        else:
            return self.model.best(s)
    def batch_update(self):
        action_featuress, rewards, obss2 = self.buffer.sample(self.args.batch_size)
        if self.target is not None:
            values = self.target.eval_batch(obss2)
        else:
            values = self.model.eval_batch(obss2)

        if self.verbose:
            print("Batch update. Values:", rewards+values)

        return self.model.batch_update(np.array(action_featuress), rewards + values)
    def update(self, obs, action, reward, obs2):
        """ Gets epsilon-greedy action using self.model """
        if self.target is not None:
            value = self.target.eval(obs2)
        else:

            value = self.model.eval(obs2)

        self.model.single_update(obs[action], value+reward)

        if self.verbose:
            print("Single update. Value:", value+reward)




class TrainingSession:
    """ TODO abstraction to be implemented in order to pause learning, transfer learning and multiple training sessions in multiple environments if needed"""

    def __init__(self, agent: DQNAgentRefactored, env: Environment, seconds=None,
                 max_steps=None, max_eps=None, save_freq=200000,
                 last_obs=None,
                 early_stopping=False, save_at_end=False, results_path=None,
                 n_agents_budget=1000):

        if agent.training_start is None:
            agent.training_start = time.time()
            agent.last_best = 0

        self.agent = agent
        self.env = env
        self.steps, self.eps = 0, 0
        self.seconds = seconds
        self.max_steps = max_steps
        self.max_eps = max_eps
        self.save_freq = save_freq
        self.early_stopping=early_stopping
        self.save_at_end=save_at_end
        self.results_path=results_path
        self.n_agents_budget=n_agents_budget
        self.converged = False
        self.epsilon_step = (agent.args.first_epsilon - agent.args.last_epsilon)
        self.epsilon_step /= agent.args.epsilon_decay_steps

        print("Warning: self.model being overwritten by hand, remember to refactor")
        self.last_steps = []

    def run(self):
        self.agent.set_model(self.env)
        self.env.reset()
        while(self.not_finished()):
            obs = self.agent.observe(self.env)
            action_index = self.agent.get_action_based_on_last_observation()
            s_t_1 = self.env.expand_and_propagate(action_index)
            self.agent.add_experience(s_t_1)
            self.agent.experience_replay()

    def run_imperative(self, E = ("AT",2,2), T=0):
        Env = CompositionGraph(E[0], E[1], E[2]).start_composition(no_tau=True)
        feature_extractor = FeatureExtractor(Env)
        Q = default_network(n_features = feature_extractor.get_transition_features_size())
        Qp = copy.deepcopy(Q)
        B = ReplayBuffer(10000)

        #self.random

        S0 = Env.reset
        for i in range(T):
            pass
            #at
    def pause(self):
        raise NotImplementedError

    def save(self, agent):
        os.makedirs(self.results_path, exist_ok=True)
        OnnxModel(agent.model).save(self.results_path  + "/" + str(agent.save_idx))

        with open(self.results_path + "/" + str(agent.save_idx) + ".json", "w") as f:
            info = {
                "training time": time.time() - agent.training_start,
                "training steps": agent.training_steps,
            }
            info.update(vars(agent.args))
            info.update(self.env.contexts[0].composition.info())
            info.update({'expansion_budget_exceeded': 'false'})
            json.dump(info, f)

        print("Agent", agent.save_idx, "saved. Training time:", time.time() - agent.training_start, "Training steps:",
              agent.training_steps)
        agent.save_idx += 1
    def save_as_torch(self, agent):
        os.makedirs(self.results_path, exist_ok=True)
        agent.model.save(self.results_path  + "/" + str(agent.save_idx))

        with open(self.results_path + "/" + str(agent.save_idx) + ".json", "w") as f:
            info = {
                "training time": time.time() - agent.training_start,
                "training steps": agent.training_steps,
            }
            info.update(vars(agent.args))
            info.update(self.env.contexts[0].composition.info())
            info.update({'expansion_budget_exceeded': 'false'})
            json.dump(info, f)

        print("Agent", agent.save_idx, "saved. Training time:", time.time() - agent.training_start, "Training steps:",
              agent.training_steps)
        agent.save_idx += 1

    def finished(self):
        cond1 = self.max_steps>self.steps and not self.early_stopping
        cond2 = self.n_agents_budget>0
        cond3 = self.max_eps is not None and self.eps>=self.max_eps
        if(self.max_steps is not None and self.training_steps > self.max_steps
                and (self.training_steps - self.last_best) / self.training_steps > 0.33):
            self.converged=True
        cond4 = self.early_stopping and self.converged
        return cond1 or cond2 or cond3 or cond4

    def compute_final_info(self, info):
        instance = (self.env.info["problem"], self.env.info["n"], self.env.info["k"])
        if instance not in self.best_training_perf.keys() or \
                info["expanded transitions"] < self.best_training_perf[instance]:
            self.best_training_perf[instance] = info["expanded transitions"]
            print("New best at instance " + str(instance) + "!", self.best_training_perf[instance], "Steps:",
                  self.training_steps)
            self.last_best = self.training_steps
        info.update({
            "training time": time.time() - self.training_start,
            "training steps": self.training_steps,
            "instance": instance,
            "loss": self.model.current_loss(),

        })
        self.training_data.append(info)
