from util import *
from composition import CompositionGraph
from replay_buffer import ReplayBuffer
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
    def __init__(self, args, nfeatures, nn_model : torch.nn.Module = None,save_file=None, verbose=False):
        Warning("TODO: refactor this initialization")
        assert nn_model is not None or nfeatures is not None

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

        if nn_model is None: self.nn_model = default_network(nfeatures) #TODO refactor this line as well

    def set_model(self, features : list[Feature], context : CompositionGraph, prebuilt = None):
        if prebuilt is not None:
            self.model = prebuilt
            return
        print("Inferred default model should also be possible in the future")
        raise NotImplementedError
        input_size = sum([feature.get_size(context) for feature in features])
    def _get_experience_from_random_policy(self, env, total_steps, nstep=1):
        """ TODO it is not ok for an agent to restart and execute the steps of the environment, refactor this
        A random policy is run for total_steps steps, saving the observations in the format of the replay buffer """
        states = []

        env.reset_from_copy()
        obs = env.actions()
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
                obs = env.reset()
            else:
                if len(last_steps) >= nstep:
                    states.append((last_steps[0], -nstep, obs2))
                last_steps = last_steps[len(last_steps) - nstep + 1:]
                obs = obs2
            steps += 1
        return states

    def initializeBuffer(self, env, buffer_size):
        """ Initialize replay buffer uniformly with experiences from a set of environments """
        exp_per_instance = buffer_size // env.get_number_of_contexts()

        print("Initializing buffer with", exp_per_instance, "observations per instance, and", env.get_number_of_contexts(), "instances.")

        self.buffer = ReplayBuffer(buffer_size)
        random_experience = self._get_experience_from_random_policy(env,total_steps=exp_per_instance,nstep=self.args.n_step)
        for action_features, reward, obs2 in random_experience:
            self.buffer.add(action_features, reward, obs2)
        print("Done.")



