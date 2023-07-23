from model import *

class CompositionGraphTests:
    def __init__(self):
        self.problems = BENCHMARK_PROBLEMS
        self.instances = [(1,1),(1,2),(2,1),(2,2),(2,3)]
        self.max_num_expansions = 100
    def write_benchmark_feature_outputs(self):
        for problem in self.problems:
            for ins in self.instances:
                d = CompositionGraph(problem, ins[0], ins[1])
                d.start_composition()
                da = CompositionAnalyzer(d)
                i = self.max_num_expansions
                while (i and not d._javaEnv.isFinished()):
                    raise NotImplementedError
                    d.expand(0)
                    #[(da.compute_features(trans)) for trans in d.getFrontier()]
                    #assert (d._expansion_order[-1] in [e[2]["action_with_features"] for e in d.edges(data=True)])
                    i-=1
    def test_expansions_are_added(self):
        for problem in self.problems:
            for ins in self.instances:
                d = CompositionGraph(problem, ins[0], ins[1])
                d.start_composition()
                da = CompositionAnalyzer(d)
                i = self.max_num_expansions
                while (i and not d._javaEnv.isFinished()):
                    d.expand(0)
                    [(da.compute_features(trans)) for trans in d.getFrontier()]
                    assert (d._expansion_order[-1] in [e[2]["action_with_features"] for e in d.edges(data=True)]), "Last expansion was not added to the composition graph."
                    i-=1

        print("PASSED")

if __name__ == "__main__":

    CompositionGraphTests().test_expansions_are_added()
    """d = CompositionGraph("AT", 3, 3)

    d.start_composition()
    da = CompositionAnalyzer(d)
    while(i and not d._javaEnv.isFinished()):
        d.expand(0)
        [(da.compute_features(trans)) for trans in d.getFrontier()]
        assert(d._expansion_order[-1] in [e[2]["action_with_features"] for e in d.edges(data=True)])"""

        #k+=sum([sum(da.isLastExpanded(trans[2]["action_with_features"])) for trans in d.edges(data=True)])
        #i-=1


    #breakpoint()
    #print(k)


