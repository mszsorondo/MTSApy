from util import *
from composition import CompositionGraph, Environment
from replay_buffer import ReplayBuffer
import torch
import json

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
    def __init__(self, args, nfeatures, nn_model : torch.nn.Module = None,save_file=None, verbose=False):
        Warning("TODO: refactor this initialization")
        assert nn_model is not None or nfeatures is not None

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

    def set_model(self, features : list[Feature], context : CompositionGraph, prebuilt = None):
        if prebuilt is not None:
            self.model = prebuilt
            return
        print("Inferred default model should also be possible in the future")
        raise NotImplementedError
        input_size = sum([feature.get_size(context) for feature in features])
    def _get_experience_from_random_policy(self, env : Environment, total_steps, nstep=1):
        """ TODO it is not ok for an agent to restart and execute the steps of the environment, refactor this
        A random policy is run for total_steps steps, saving the observations in the format of the replay buffer """
        states = []

        env.reset_from_copy()
        Warning("HERE obs is not actions, but the featurization of the frontier actions")
        obs = env.frontier_features()
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
                obs = env.frontier_features()
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
              early_stopping=False, save_at_end=False, results_path=None, n_agents_budget=1000):
        assert self.current_training_environment is not None
        assert self.model is not None
        session = TrainingSession(self, self.current_training_environment, seconds, max_steps, max_eps, save_freq, last_obs,
              early_stopping, save_at_end, results_path, n_agents_budget)
        if (last_obs is None): self.current_training_environment.reset_from_copy()

        obs = self.get_frontier_features() if (last_obs is None) else last_obs

        while(session.not_finished()):
            a = self.get_action(obs, self.epsilon)
            session.last_steps.append(obs[a])
            obs2, reward, done, step_info = self.current_training_environment.step(a)

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
                self.batch_update()
            else:
                self.update(obs, a, reward, obs2)

            if not done: obs = obs2
            else:
                info = session.compute_final_info(step_info)
                obs = self.current_training_environment.reset_from_copy()

            if self.training_steps % save_freq == 0 and results_path is not None:
                self.save(self.current_training_environment.info, path=results_path)
                n_agents_budget -= 1
            breakpoint()

    def set_training_environment(self, env: Environment):
        self.current_training_environment = env

    def from_pretrained(self):
        raise NotImplementedError
    def get_frontier_features(self):

        features = [self.current_training_environment.contexts[0].test_features_on_transition(transition) for transition in
                    self.current_training_environment.contexts[0].composition.getFrontier()]
        return np.asarray(features.copy())#TODO why copy? Memory cost of this?
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
        breakpoint()
        self.model.batch_update(np.array(action_featuress), rewards + values)
    def update(self, obs, action, reward, obs2):
        """ Gets epsilon-greedy action using self.model """
        if self.target is not None:
            value = self.target.eval(obs2)
        else:
            value = self.model.eval(obs2)

        self.model.single_update(obs[action], value+reward)

        if self.verbose:
            print("Single update. Value:", value+reward)
    def save(self, env_info, path):
        os.makedirs(path, exist_ok=True)
        OnnxModel(self.model).save(path + "/" + str(self.save_idx))

        with open(path + "/" + str(self.save_idx) + ".json", "w") as f:
            info = {
                "training time": time.time() - self.training_start,
                "training steps": self.training_steps,
            }
            info.update(vars(self.args))
            info.update(env_info)
            json.dump(info, f)

        print("Agent", self.save_idx, "saved. Training time:", time.time() - self.training_start, "Training steps:", self.training_steps)
        self.save_idx += 1




class TrainingSession:
    """ TODO abstraction to be implemented in order to pause learning, transfer learning and multiple training sessions in multiple environments if needed"""

    def __init__(self, agent: DQNAgent, env: Environment, seconds=None, max_steps=None, max_eps=None, save_freq=200000,
                 last_obs=None,
                 early_stopping=False, save_at_end=False, results_path=None, n_agents_budget=1000):

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

        self.epsilon_step = (agent.args.first_epsilon - agent.args.last_epsilon)
        self.epsilon_step /= agent.args.epsilon_decay_steps
        print("Warning: self.model being overwritten by hand, remember to refactor")
        self.last_steps = []


    def pause(self):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError

    def not_finished(self):
        return self.n_agents_budget>0
        raise NotImplementedError

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