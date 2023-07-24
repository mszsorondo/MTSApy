# MTSApy
A python API to facilitate the usage of MTSA Java implementation.  

<span style="color: red;">This tool is still under development and may present some bugs. It supports non-blocking + safety control problems.</span>

This API aims to better integration between [LAFHIS' Modal Transition System Analyzer](https://mtsa.dc.uba.ar/) and modern learning-based systems, in an effort to develop AI-guided directed control heuristics.  
It gives a graph-like representation of state machines and its composition. It also provides access to the usage of SOTA controller synthesis algorithms.


### 

## Basic Usage
#### Make sure you installed the DCSNonBlockingForRLPythonAPI branch, compiled the resulting .jar and moved it to the root directory of this repo as mtsa.jar
For further setup-installation questions regarding the MTSA engine, email msorondo@live.com.ar

Setup minimal composition configuration:
```python
c = CompositionGraph(problem_name, n, k)
```
Given that CompositionGraph class inherits directly from Networkx's DiGraph class, you can navigate it as such and see the basic transition and state characteristics as edge and node attributes respectively.

Start the OTF synthesis and composition process:
```python
c.start_composition()
```
Get current exploration frontier list:
```python
c.getFrontier()
```
Expand i-th transition from the frontier list:
```python
c.expand(i)
```

