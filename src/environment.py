from extractor import *
from composition import CompositionGraph
from features import *
class Environment:
    def __init__(self, contexts : FeatureExtractor, normalize_reward : bool = False):
        """Environment base class.
            TODO are contexts actually part of the concept of an RL environment?
            FIXME FeatureExtractor is not something from the environment
            """
        self.contexts = contexts
        self.normalize_reward = normalize_reward

    def reset_from_copy(self):
        self.contexts = [FeatureExtractor(context.composition.reset_from_copy()) for context in self.contexts]
        return self

    def get_number_of_contexts(self):
        return len(self.contexts)
    def get_contexts(self):
        return self.contexts
    def step(self, action_idx, context_idx = 0):
        composition_graph = self.contexts[context_idx].composition
        composition_graph.expand(action_idx) # TODO refactor. Analyzer should not be the expansion medium
        Warning("HERE obs is not actions, but the featurization of the frontier actions")
        if not composition_graph._javaEnv.isFinished(): return self.frontier_features(), self.reward(), False, {}
        else: return None, self.reward(), True, self.get_results()
    def get_results(self, context_idx = 0):
        composition_dg = self.contexts[context_idx].composition
        return {
            "synthesis time(ms)": float(composition_dg._javaEnv.getSynthesisTime()),
            "expanded transitions": int(composition_dg._javaEnv.getExpandedTransitions()),
            "expanded states": int(composition_dg._javaEnv.getExpandedStates())
        }


    def reward(self):
        #TODO ?normalize like Learning-Synthesis?
        return -1
    def state(self):
        raise NotImplementedError
    def actions(self, context_idx=0):
        #TODO refactor
        return self.contexts[context_idx].composition.getFrontier()
    def frontier_features(self):
        #TODO you can parallelize this
        return [self.contexts[0].extract(trans) for trans in self.contexts[0].getFrontier()]
def enable_first_n_values(enabler, n):
    for k,v in enabler.items():
        enabler[k] = n>0
        n-=1
if __name__=="__main__":
    d = CompositionGraph("AT", 3, 3)
    d.start_composition()
    ENABLED_PYTHON_FEATURES = {
        EventLabel: True,
        StateLabel: False,
        Controllable: False,
        MarkedSourceAndSinkStates: False,
        CurrentPhase: False,
        ChildNodeState: False,
        UncontrollableNeighborhood: False,
        ExploredStateChild: False,
        IsLastExpanded: False
    }
    for i in range(1,len(ENABLED_PYTHON_FEATURES.keys())):
        d = CompositionGraph("AT", 3, 3)
        d.start_composition()
        enable_first_n_values(ENABLED_PYTHON_FEATURES, i)
        print(ENABLED_PYTHON_FEATURES)
        da = FeatureExtractor(d,ENABLED_PYTHON_FEATURES)

        da.train_gae_on_full_graph(to_undirected=True, epochs=2000)

    d.load()
