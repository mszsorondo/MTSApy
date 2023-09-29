import random

from features import *


#REMEMBER this is only useful for testing features in the python implementation.
# Once all tests pass, this test suite will no longer be that much useful
# unless you do internal refactors to MTSA that impact on current implementation

JAVA_METHOD_TO_PYTHON_METHOD = {
    #TODO generalize testing functions for all features, put parameters
    "alfComputeSlice" : EventLabel,
    "slfComputeSlice" : StateLabel,
    "isControllableComputeSlice" : Controllable,
    "isMarkedComputeSlice": MarkedState,
    "contextComputeSlice":CurrentPhase,
    "childStatusComputeSlice" : ChildNodeState,
    "uncontrollableComputeSlice" : UncontrollableNeighborhood,
    "exploredComputeSlice" : ExploredStateChild,
    "justExploredComputeSlice" : IsLastExpanded
}
def test_event_label_on_problem(problem,n,k, post_mortem_debug):
    cg = CompositionGraph(problem,n,k)
    cg.start_composition(no_tau=True)
    javaEnv = cg.get_jvm()
    featureMaker = javaEnv.featureMaker
    featureMaker.readLabelsOrdered(f"{LABELS_PATH}/{problem}.txt") #readLabels
    tr = cg.getFrontier()[0]
    while (not cg.finished()):
        for trans in cg.getFrontier():
            pyf = EventLabel.compute(cg, trans)
            jarf = [i for i in featureMaker.alfComputeSlice(trans)]

            try:
                assert (pyf == jarf)
            except AssertionError:
                if post_mortem_debug:
                    print("Test failed. Post-mortem debug")
                    print(f"Java features: {jarf} ; Python features: {pyf}")
                    breakpoint()
                    o = 0
        cg.expand(random.randint(0, len(cg.getFrontier()) - 1))
    print(f"{problem} , {n}, {k} EVENT LABEL PASSED")

def test_state_label_on_problem(problem,n,k, post_mortem_debug):
    cg = CompositionGraph(problem,n,k)
    cg.start_composition(no_tau=True)
    javaEnv = cg.get_jvm()
    featureMaker = javaEnv.featureMaker
    featureMaker.readLabelsOrdered(f"{LABELS_PATH}/{problem}.txt") #readLabels
    tr = cg.getFrontier()[0]
    while (not cg.finished()):
        for trans in cg.getFrontier():
            pyf = StateLabel.compute(cg, trans)
            jarf = [i for i in featureMaker.slfComputeSlice(trans)]

            try: assert(pyf==jarf)
            except AssertionError:
                if post_mortem_debug:
                    print("Test failed. Post-mortem debug")
                    print(f"Java features: {jarf} ; Python features: {pyf}")
                    breakpoint()
                    o = 0
        cg.expand(random.randint(0, len(cg.getFrontier()) - 1))
    print(f"{problem} , {n}, {k} EVENT LABEL PASSED")

def test_event_label(post_mortem_debug=False):
    problems = ["AT", "BW", "DP", "TA", "TL"]
    ns = [2]
    ks = [2,3]
    for problem in problems:
        for n in ns:
            for k in ks:
                test_event_label_on_problem(problem,n,k,post_mortem_debug)

def test_state_label(post_mortem_debug=False):
    problems = ["AT", "BW", "DP", "TA", "TL"]
    ns = [2]
    ks = [2,3]
    for problem in problems:
        for n in ns:
            for k in ks:
                test_state_label_on_problem(problem,n,k,post_mortem_debug)
if __name__ == "__main__":
    test_event_label(post_mortem_debug=True)
    test_state_label(post_mortem_debug=True)