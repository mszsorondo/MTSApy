from util import *


class CompositionGraph(nx.DiGraph):
    def __init__(self, problem, n, k):
        super().__init__()
        self._problem, self._n, self._k = problem, n , k
        self._initial_state = None
        self._machines = []
        self._frontier = []
        self._completed = False
        self._javaEnv = None
    def start_composition(self, mtsa_version_path = 'mtsa.jar'):
        assert(self._initial_state is None)
        if not jpype.isJVMStarted(): jpype.startJVM(classpath=[mtsa_version_path])
        from MTSTools.ac.ic.doc.mtstools.model.operations.DCS.nonblocking import DirectedControllerSynthesisNonBlocking, FeatureBasedExplorationHeuristic
        from ltsa.dispatcher import TransitionSystemDispatcher

        c = FeatureBasedExplorationHeuristic.compileFSP(f"{FSP_PATH}/{self._problem}/{self._problem}-{self._n}-{self._k}.fsp")

        self._javaEnv = TransitionSystemDispatcher.hcsInteractive(c.getFirst(), c.getSecond())
        assert(self._javaEnv is not None)
        print("see startSynthesis from java.DCSForPython and use a similar functionality to complete this method")
        raise NotImplementedError




if __name__ == "__main__":
    d = CompositionGraph("AT", 2, 2)

    d.start_composition()
