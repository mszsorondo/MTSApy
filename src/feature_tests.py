import random

from features import *


#REMEMBER this is only useful for testing features in the python implementation.
# Once all tests pass, this test suite will no longer be that much useful
# unless you do internal refactors to MTSA that impact on current implementation
def test_event_label_on_problem(problem,n,k, post_mortem_debug):
    cg = CompositionGraph(problem,n,k)
    cg.start_composition(no_tau=True)
    javaEnv = cg.get_jvm()
    featureMaker = javaEnv.featureMaker
    featureMaker.readLabelsOrdered(f"{LABELS_PATH}/{problem}.txt") #readLabels
    tr = cg.getFrontier()[0]
    while (not cg.finished()):
        frontier_features = [EventLabel.compute(cg, trans)==[i for i in featureMaker.alfComputeSlice(trans)] for trans in cg.getFrontier()]
        cg.expand(random.randint(0,len(cg.getFrontier())-1))
        try: assert(all(frontier_features))
        except AssertionError:
            if post_mortem_debug:
                print("Post-mortem debug")
                breakpoint()
    print(f"{problem} , {n}, {k} EVENT LABEL PASSED")

def test_event_label(post_mortem_debug=False):
    problems = ["AT", "BW", "DP", "TA", "TL"]
    ns = [2]
    ks = [2,3]
    for problem in problems:
        for n in ns:
            for k in ks:
                test_event_label_on_problem(problem,n,k,post_mortem_debug)
if __name__ == "__main__":
    test_event_label(post_mortem_debug=True)