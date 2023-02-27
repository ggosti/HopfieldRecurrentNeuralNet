# HopfieldRecurrentNeuralNet

Implementation of a discrete-time recurrent neural network with binary neurons.

Requirements
------------

The code uses Python2.7 and the following Python open source packages:
- NumPy
- Matplotlib
- NetworkX

You may install the Python requirements with `pip install -r requirements.txt`.

This implementation considers a discrete-time recurrent neural network with binary neurons.
The number of neurons `N` can be set, and the neurons can have activation state either {0,1} if `typ = 0`,
or {-1,1} if `typ = 1`. The activation function can only be a step function with a certain threshold thr.

In the following text it is assumed that the following packages are loaded:
```python
import learningRNN as lrnn
import numpy as np
import matplotlib.pyplot as plt
```

It is useful to set the following parameters:
```python
N= 16 # Number of neurons
typ = 0 # typ = 0 neurons with binary activation states {0,1}, typ = 1  neurons with states {-1,1}.
        # typ=1 --> {-1,1}    typ=0 --> {0,1} 
thr = 0 # activation function threshold
```

```python
initial_state = lrnn.stateIndex2stateVec(17,N,typ)
```

We can get back to the index:
```python
initial_state_index = lrnn.stateVec2stateIndex(initial_state, N, typ)
```

Get the next state given the objective network and an `initial_state`
```python
transition = lrnn.transPy(initial_state,objective, N, typ, thr)
```

Given a list of initial states `initial_state_list`, get a list with the corresponding transition
```python
initial_state_list = lrnn.stateIndex2stateVecSeq([19,2001,377], N, typ)
transition_list = lrnn.transPy(initial_state_list,objective, N, typ, thr)
```
