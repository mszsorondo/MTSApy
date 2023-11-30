import random

from features import *


#REMEMBER this is only useful for testing features in the python implementation.
# Once all tests pass, this test suite will no longer be that much useful
# unless you do internal refactors to MTSA that impact on current implementation

JAVA_METHOD_TO_PYTHON_METHOD = {
    "alfComputeSlice" : EventLabel,
    "slfComputeSlice" : StateLabel,
    "isControllableComputeSlice" : Controllable,
    "isMarkedComputeSlice": MarkedSourceAndSinkStates,
    "contextComputeSlice":CurrentPhase,
    "childStatusComputeSlice" : ChildNodeState,
    "uncontrollableComputeSlice" : UncontrollableNeighborhood,
    "exploredComputeSlice" : ExploredStateChildFromJava,
    "justExploredComputeSlice" : IsLastExpanded,
    "childDeadlockComputeSlice": ChildDeadlock,
}
JAVA_METHOD_TO_PYTHON_METHOD = bidict(JAVA_METHOD_TO_PYTHON_METHOD)
def test_feature_on_problem(java_feature_name : str,problem,n,k,post_mortem_debug):
    cg = CompositionGraph(problem, n, k)
    cg.start_composition(no_tau=True)
    javaEnv = cg.get_jvm()
    featureMaker = javaEnv.featureMaker
    featureMaker.readLabelsOrdered(f"{LABELS_PATH}/{problem}.txt")  # readLabels
    tr = cg.getFrontier()[0]
    java_feature = getattr(featureMaker, java_feature_name)
    while (not cg.finished()):
        for trans in cg.getFrontier():
            pyf_result = JAVA_METHOD_TO_PYTHON_METHOD[java_feature_name].compute(cg, trans)
            jarf_result = [i for i in java_feature(trans)]
            try:
                assert (pyf_result == jarf_result)
            except AssertionError:
                if post_mortem_debug:
                    print(f"Test failed for {java_feature_name} . Post-mortem debug")
                    print(f"Java features: {jarf_result} ; Python features: {pyf_result}")
                    breakpoint()
                    pass
        cg.expand(random.randint(0, len(cg.getFrontier()) - 1))
    print(f"{problem} , {n}, {k} {java_feature_name} PASSED")

def test_feature(feature_name, post_mortem_debug=False):
    problems = ["AT", "BW", "DP", "TA", "TL"] #FIXME testing CM takes too long
    ns = [2]
    ks = [2, 3]
    for problem in problems:
        for n in ns:
            for k in ks:
                test_feature_on_problem(feature_name, problem, n, k, post_mortem_debug)

if __name__ == "__main__":
    #test_event_label(post_mortem_debug=True)
    #test_state_label(post_mortem_debug=True)
    for feature_name in JAVA_METHOD_TO_PYTHON_METHOD.keys():
        test_feature(feature_name,True)
    print("All python features work.")